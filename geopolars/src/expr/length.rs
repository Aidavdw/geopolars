//! How long a linestring is along the ellipsoid, for geometries that declare a CRS.
//!
//! Geometries without a CRS are measured in plain Polars expressions instead, on the Python side.

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use super::distance::GeodesicMetric;
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// The CRS of a column that has a length. Type operation only.
fn crs_of_curve(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        matches!(geo.kind, Kind::LineString | Kind::MultiLineString),
        SchemaMismatch: "a length is measured on a `{}` or `{}` column, got: {}",
        Kind::LineString.name(), Kind::MultiLineString.name(), field.dtype()
    );
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geo.metadata.crs()
}

/// `output_type_func` for [`length_geodesic`]: an `f64` in metres,
/// or a list of them for a multilinestring.
fn metres(input_fields: &[Field]) -> PolarsResult<Field> {
    crs_of_curve(&input_fields[0])?;
    let dtype = match describe(input_fields[0].dtype())?.kind {
        Kind::MultiLineString => DataType::List(Box::new(DataType::Float64)),
        _ => DataType::Float64,
    };
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// One chunk of linestrings, each a list of coordinates.
struct Lines<'a> {
    lines: &'a ListArray<i64>,
    coords: CoordsView<'a>,
    /// The height of every coordinate, if the line has one.
    z: Option<&'a PrimitiveArray<f64>>,
}

impl<'a> Lines<'a> {
    fn new(lines: &'a ListArray<i64>) -> PolarsResult<Self> {
        let coords = CoordsView::new(lines.values().as_ref())?;
        let z = heights(lines.values().as_ref())?;
        Ok(Self { lines, coords, z })
    }

    /// The length of line `i`, `None` if it or any of its coordinates is missing.
    fn length(&self, i: usize, metric: &GeodesicMetric) -> PolarsResult<Option<f64>> {
        if self.lines.is_null(i) {
            return Ok(None);
        }
        let (start, end) = self.lines.offsets().start_end(i);
        // Decided once per line, not per vertex: 2D lines take a loop that never reads `z`.
        match self.z {
            None => metric.length((start..end).map(|j| self.coords.xy(j))),
            Some(z) => metric.length_with_height((start..end).map(|j| {
                let xy = self.coords.xy(j)?;
                z.is_valid(j).then(|| (xy, z.value(j)))
            })),
        }
    }
}

/// The `z` field of a chunk of coordinates, `None` for a dimension without one.
fn heights(coords: &dyn Array) -> PolarsResult<Option<&PrimitiveArray<f64>>> {
    let coords: &StructArray = downcast(coords, "a coordinate struct")?;
    coords
        .fields()
        .iter()
        .position(|field| field.name.as_str() == "z")
        .map(|i| downcast(coords.values()[i].as_ref(), "f64 `z` coordinates"))
        .transpose()
}

fn linestrings(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.list()?.downcast_iter() {
        let lines = Lines::new(chunk)?;
        for i in 0..chunk.len() {
            out.push(lines.length(i, metric)?);
        }
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// The length of each part of every multilinestring, in order.
/// A missing part (or one with a missing coordinate) gets a missing length,
/// without taking its siblings with it.
fn multilinestrings(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<ListChunked> {
    let name = storage.name().clone();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|chunk| -> PolarsResult<_> {
            let lines = Lines::new(downcast(chunk.values().as_ref(), "a list of linestrings")?)?;
            // Only the parts this chunk points at: it may be a slice of a larger one.
            let (first, last) = (
                *chunk.offsets().first() as usize,
                *chunk.offsets().last() as usize,
            );
            let parts = (first..last)
                .map(|line| lines.length(line, metric))
                .collect::<PolarsResult<PrimitiveArray<f64>>>()?;
            let offsets = Offsets::try_from_lengths(chunk.offsets().lengths())?;
            Ok(ListArray::<i64>::new(
                ListArray::<i64>::default_datatype(ArrowDataType::Float64),
                offsets.into(),
                parts.boxed(),
                chunk.validity().cloned(),
            ))
        });
    ListChunked::try_from_chunk_iter(name, chunks)
}

/// See `length`.
#[polars_expr(output_type_func=metres)]
fn length_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let metric = GeodesicMetric::of(&crs_of_curve(&inputs[0].field())?)?;
    let storage = inputs[0].ext()?.storage();

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::LineString => linestrings(storage, &metric)?.into_series(),
        Kind::MultiLineString => multilinestrings(storage, &metric)?.into_series(),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint) => polars_bail!(
            SchemaMismatch: "a `{}` has no length", kind.name()
        ),
    };
    Ok(out.with_name(inputs[0].name().clone()))
}
