//! How long a linestring is along the ellipsoid, for geometries that declare a CRS.
//!
//! Geometries without a CRS are measured in plain Polars expressions instead, on the Python side.

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use pyo3_polars::derive::polars_expr;

use super::distance::GeodesicMetric;
use crate::geoarrow::crs::ExtensionMetadata;
use crate::geoarrow::to_rsgeo::{downcast, Coords};
use crate::geoarrow::{describe, Kind};

/// The CRS of a column that has a length. Type operation only.
fn crs_of_curve(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        matches!(geo.kind, Kind::LineString | Kind::MultiLineString),
        SchemaMismatch: "a length is measured on a `{}` or `{}` column, got: {}",
        Kind::LineString.name(), Kind::MultiLineString.name(), field.dtype()
    );
    let metadata = ExtensionMetadata::parse(geo.typ.serialize_metadata().as_deref())?;
    polars_ensure!(
        metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    metadata.crs()
}

/// `output_type_func` for [`length_geodesic`]: an `f64`, in metres.
fn metres(input_fields: &[Field]) -> PolarsResult<Field> {
    crs_of_curve(&input_fields[0])?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

/// One chunk of linestrings, each a list of coordinates.
struct Lines<'a> {
    lines: &'a ListArray<i64>,
    coords: Coords<'a>,
    /// The height of every coordinate, if the line has one.
    z: Option<&'a PrimitiveArray<f64>>,
}

impl<'a> Lines<'a> {
    fn new(lines: &'a ListArray<i64>) -> PolarsResult<Self> {
        let coords = Coords::new(lines.values().as_ref())?;
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

/// The sum of each multilinestring's parts.
/// A missing part makes the whole missing, as GeoArrow has no nulls inside a geometry.
fn multilinestrings(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.list()?.downcast_iter() {
        let lines = Lines::new(downcast(chunk.values().as_ref(), "a list of linestrings")?)?;
        for i in 0..chunk.len() {
            if chunk.is_null(i) {
                out.push(None);
                continue;
            }
            let (start, end) = chunk.offsets().start_end(i);
            // Stops at the first error, or the first missing part.
            let total: PolarsResult<Option<f64>> =
                (start..end).map(|line| lines.length(line, metric)).sum();
            out.push(total?);
        }
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// The length of every linestring along its CRS's ellipsoid, in metres.
/// A multilinestring is as long as its parts together.
#[polars_expr(output_type_func=metres)]
fn length_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let metric = GeodesicMetric::new(&crs_of_curve(&inputs[0].field())?)?;
    let storage = inputs[0].ext()?.storage();

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::LineString => linestrings(storage, &metric)?,
        Kind::MultiLineString => multilinestrings(storage, &metric)?,
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint) => polars_bail!(
            SchemaMismatch: "a `{}` has no length", kind.name()
        ),
    };
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}
