//! How long a linestring is:
//! - in the space its coordinates lie in ([`length_planar`]),
//! - along the ellipsoid of its CRS ([`length_geodesic`]).

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray};
use pyo3_polars::derive::polars_expr;

use super::distance::{GeodesicMetric, Metric, Planar};
use crate::geoarrow::storage::{downcast, Axes};
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

/// Length always returns an `f64`.
/// Multilinestring: sum
fn length_dtype(field: &Field) -> PolarsResult<Field> {
    match describe(field.dtype())?.kind {
        Kind::LineString | Kind::MultiLineString => {
            Ok(Field::new(field.name().clone(), DataType::Float64))
        }
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            Err(has_no_length(kind))
        }
    }
}

/// For [`length_planar`]: in the unit of the coordinates.
fn kernel_output_type_length_planar(input_fields: &[Field]) -> PolarsResult<Field> {
    length_dtype(&input_fields[0])
}

/// For [`length_geodesic`]: in the unit of the CRS.
fn kernel_output_type_length_geodesic(input_fields: &[Field]) -> PolarsResult<Field> {
    // So a CRS PROJ cannot measure in fails while the plan is built.
    GeodesicMetric::of(&crs_of(&input_fields[0])?)?;
    length_dtype(&input_fields[0])
}

/// The length of the line through `vertices`: fewer than two have a length of `0.0`.
#[inline(always)]
fn line_length<const Z: bool, M: Metric>(
    metric: &M,
    axes: &Axes,
    mut vertices: Range<usize>,
) -> PolarsResult<f64> {
    let Some(first) = vertices.next() else {
        return Ok(0.0);
    };
    let mut previous = metric.vertex(axes.position::<Z>(first))?;
    let mut total = 0.0;
    for i in vertices {
        let here = metric.vertex(axes.position::<Z>(i))?;
        total += metric.segment::<Z>(previous, here);
        previous = here;
    }
    Ok(total)
}

/// The length of every row of a chunk: a linestring,
/// or with `MULTI`, a multilinestring as the sum of its parts.
fn lengths<const Z: bool, const MULTI: bool>(
    metric: &impl Metric,
    rows: &ListArray<i64>,
    lines: &ListArray<i64>,
    axes: &Axes,
) -> PolarsResult<PrimitiveArray<f64>> {
    let line = |i: usize| {
        let (start, end) = lines.offsets().start_end(i);
        line_length::<Z, _>(metric, axes, start..end)
    };
    let values = (0..rows.len())
        .map(|i| {
            // A missing row's offsets point anywhere, so it is not measured.
            if rows.is_null(i) {
                Ok(0.0)
            } else if MULTI {
                let (start, end) = rows.offsets().start_end(i);
                (start..end).map(line).sum()
            } else {
                line(i)
            }
        })
        .collect::<PolarsResult<Vec<f64>>>()?;
    Ok(PrimitiveArray::from_vec(values).with_validity(rows.validity().cloned()))
}

/// The length of every geometry in `geometries`, as `metric` measures it.
fn measure(metric: &impl Metric, geometries: &Series) -> PolarsResult<Series> {
    let multi = match describe(geometries.dtype())?.kind {
        Kind::LineString => false,
        Kind::MultiLineString => true,
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            return Err(has_no_length(kind))
        }
    };
    let storage = geometries.ext()?.storage();
    let chunks = storage.list()?.downcast_iter().map(|rows| {
        let lines = if multi {
            downcast(rows.values().as_ref(), "a list of linestrings")?
        } else {
            rows
        };
        let (axes, dim) = Axes::new(lines.values().as_ref())?;
        with_const_flags!(lengths(metric, rows, lines, &axes); dim.has_z(), multi)
    });
    Ok(Float64Chunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}

/// See `length_planar`.
#[polars_expr(output_type_func=kernel_output_type_length_planar)]
fn length_planar(inputs: &[Series]) -> PolarsResult<Series> {
    measure(&Planar, &inputs[0])
}

/// See `length_geodesic`.
#[polars_expr(output_type_func=kernel_output_type_length_geodesic)]
fn length_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let metric = GeodesicMetric::of(&crs_of(&inputs[0].field())?)?;
    measure(&*metric, &inputs[0])
}
