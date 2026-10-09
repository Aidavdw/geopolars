//! How long a linestring is:
//! - in the space its coordinates lie in ([`length_planar`]),
//! - along the ellipsoid of its CRS ([`length_geodesic`]).

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use super::distance::GeodesicMetric;
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// The CRS of a column to measure a length in. Type operation only.
fn crs_of(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geo.metadata.crs()
}

/// Only linestrings and multilinestrings have a length: other kinds are refused
/// rather than given a length of `0.0`.
fn has_no_length(kind: Kind) -> PolarsError {
    polars_err!(
        SchemaMismatch: "a length is measured on a `{}` or `{}` column, got: `{}`",
        Kind::LineString.name(), Kind::MultiLineString.name(), kind.name()
    )
}

/// An `f64` per geometry, or a list of them (one per part) for a multilinestring.
fn length_dtype(field: &Field) -> PolarsResult<Field> {
    let dtype = match describe(field.dtype())?.kind {
        Kind::LineString => DataType::Float64,
        Kind::MultiLineString => DataType::List(Box::new(DataType::Float64)),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            return Err(has_no_length(kind))
        }
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// `output_type_func` for [`length_planar`]: in the unit of the coordinates.
fn lengths_planar(input_fields: &[Field]) -> PolarsResult<Field> {
    length_dtype(&input_fields[0])
}

/// `output_type_func` for [`length_geodesic`]: in the unit of the CRS.
fn lengths(input_fields: &[Field]) -> PolarsResult<Field> {
    // So a CRS PROJ cannot measure in fails while the plan is built.
    GeodesicMetric::of(&crs_of(&input_fields[0])?)?;
    length_dtype(&input_fields[0])
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

    /// The length of line `i` in space, by Pythagoras,
    /// `None` if it or any of its coordinates is missing.
    fn planar_length(&self, i: usize) -> PolarsResult<Option<f64>> {
        if self.lines.is_null(i) {
            return Ok(None);
        }
        let (start, end) = self.lines.offsets().start_end(i);
        // Decided once per line, not per vertex: 2D lines take a loop that never reads `z`.
        Ok(match self.z {
            None => path(
                (start..end).map(|j| self.coords.xy(j)),
                |(ax, ay), (bx, by)| (bx - ax).hypot(by - ay),
            ),
            Some(z) => path(
                (start..end).map(|j| {
                    let (x, y) = self.coords.xy(j)?;
                    z.is_valid(j).then(|| (x, y, z.value(j)))
                }),
                |(ax, ay, az), (bx, by, bz)| {
                    let (dx, dy, dz) = (bx - ax, by - ay, bz - az);
                    (dx * dx + dy * dy + dz * dz).sqrt()
                },
            ),
        })
    }

    /// The length of line `i` along the ellipsoid,
    /// `None` if it or any of its coordinates is missing.
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

/// The sum of `segment` between every pair of consecutive coordinates,
/// `None` if any of them is missing. Fewer than two coordinates have a length of `0.0`.
fn path<C: Copy>(
    coords: impl IntoIterator<Item = Option<C>>,
    segment: impl Fn(C, C) -> f64,
) -> Option<f64> {
    let mut total = 0.0;
    let mut previous = None;
    for coord in coords {
        let here = coord?;
        if let Some(previous) = previous {
            total += segment(previous, here);
        }
        previous = Some(here);
    }
    Some(total)
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

/// The length of every linestring.
fn linestrings(
    storage: &Series,
    measure: impl Fn(&Lines, usize) -> PolarsResult<Option<f64>>,
) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.list()?.downcast_iter() {
        let lines = Lines::new(chunk)?;
        for i in 0..chunk.len() {
            out.push(measure(&lines, i)?);
        }
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// The length of each part of every multilinestring.
/// A missing part (or one with a missing coordinate) gets a missing length,
/// without taking its siblings with it.
fn multilinestrings(
    storage: &Series,
    measure: impl Fn(&Lines, usize) -> PolarsResult<Option<f64>>,
) -> PolarsResult<ListChunked> {
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
                .map(|line| measure(&lines, line))
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

/// See `length_planar`.
#[polars_expr(output_type_func=lengths_planar)]
fn length_planar(inputs: &[Series]) -> PolarsResult<Series> {
    let storage = inputs[0].ext()?.storage();
    let measure = |lines: &Lines, i| lines.planar_length(i);

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::LineString => linestrings(storage, measure)?.into_series(),
        Kind::MultiLineString => multilinestrings(storage, measure)?.into_series(),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            return Err(has_no_length(kind))
        }
    };
    Ok(out.with_name(inputs[0].name().clone()))
}

/// See `length_geodesic`.
#[polars_expr(output_type_func=lengths)]
fn length_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let metric = GeodesicMetric::of(&crs_of(&inputs[0].field())?)?;
    let storage = inputs[0].ext()?.storage();
    let measure = |lines: &Lines, i| lines.length(i, &metric);

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::LineString => linestrings(storage, measure)?.into_series(),
        Kind::MultiLineString => multilinestrings(storage, measure)?.into_series(),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            return Err(has_no_length(kind))
        }
    };
    Ok(out.with_name(inputs[0].name().clone()))
}
