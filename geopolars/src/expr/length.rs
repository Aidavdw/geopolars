//! How long a linestring is:
//! - in the space its coordinates lie in ([`DistanceMethod::Planar`]),
//! - along the ellipsoid of its CRS ([`DistanceMethod::Geodesic`]).

use std::fmt::Display;

use geoarrow_abstr::native::LengthMetric;
use geoarrow_abstr::{GeoArrowCoordsShared, LineStringArray, MultiLineStringArray, Planar};
use polars::prelude::*;
use polars_arrow::array::PrimitiveArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::distance::{geodesic_metric, DistanceMethod};
use crate::geoarrow::storage::{downcast, F64Array, ListView, ValidCoordsView, ValidListView};
use crate::geoarrow::{describe, Kind};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LengthKwargs {
    method: DistanceMethod,
    /// Skip vertices with a NaN, rather than letting the length become NaN.
    ignore_nan: bool,
}

/// The CRS of a column to measure a length in. Type operation only.
fn crs_of(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geo.metadata.crs().map(str::to_owned)
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

/// Output for length:
/// - Planar is in the unit of the coordinates
/// - geodesic in the unit of the CRS
fn kernel_output_type_length(input_fields: &[Field], kwargs: LengthKwargs) -> PolarsResult<Field> {
    if let DistanceMethod::Geodesic = kwargs.method {
        // So a CRS PROJ cannot measure in fails while the plan is built.
        geodesic_metric(&crs_of(&input_fields[0])?)?;
    }
    length_dtype(&input_fields[0])
}

/// [geoarrow_abstr::native::line_string_lengths], into a polars array.
/// `with_const_flags!` cannot name the metric and output types itself, so it calls this.
fn line_string_lengths<const Z: bool, const IGNORE_NAN: bool>(
    metric: &impl LengthMetric<Error: Display>,
    array: &LineStringArray<ListView, ValidCoordsView>,
) -> PolarsResult<PrimitiveArray<f64>> {
    let F64Array(lengths) =
        geoarrow_abstr::native::line_string_lengths::<Z, IGNORE_NAN, _, F64Array>(metric, array)
            .map_err(|e| polars_err!(ComputeError: "{e}"))?;
    Ok(lengths)
}

/// [geoarrow_abstr::native::multi_line_string_lengths], into a polars array.
/// `with_const_flags!` cannot name the metric and output types itself, so it calls this.
fn multi_line_string_lengths<const Z: bool, const IGNORE_NAN: bool>(
    metric: &impl LengthMetric<Error: Display>,
    array: &MultiLineStringArray<ListView, ValidListView, ValidCoordsView>,
) -> PolarsResult<PrimitiveArray<f64>> {
    let F64Array(lengths) =
        geoarrow_abstr::native::multi_line_string_lengths::<Z, IGNORE_NAN, _, F64Array>(
            metric, array,
        )
        .map_err(|e| polars_err!(ComputeError: "{e}"))?;
    Ok(lengths)
}

/// The length of every geometry in `geometries`, as `metric` measures it.
fn measure(
    metric: &impl LengthMetric<Error: Display>,
    geometries: &Series,
    ignore_nan: bool,
) -> PolarsResult<Series> {
    let multi = match describe(geometries.dtype())?.kind {
        Kind::LineString => false,
        Kind::MultiLineString => true,
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            return Err(has_no_length(kind))
        }
    };
    let storage = geometries.ext()?.storage();
    let chunks = storage.list()?.downcast_iter().map(|rows| {
        if multi {
            let lines = downcast(rows.values().as_ref(), "a list of linestrings")?;
            let array = MultiLineStringArray {
                rows: ListView::new(rows),
                lines: ValidListView::new(lines),
                coords: ValidCoordsView::new(lines.values().as_ref())?,
            };
            let z = array.coords.dim().has_z();
            with_const_flags!(multi_line_string_lengths(metric, &array); z, ignore_nan)
        } else {
            let array = LineStringArray {
                rows: ListView::new(rows),
                coords: ValidCoordsView::new(rows.values().as_ref())?,
            };
            let z = array.coords.dim().has_z();
            with_const_flags!(line_string_lengths(metric, &array); z, ignore_nan)
        }
    });
    Ok(Float64Chunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}

/// See `length`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_length)]
fn length(inputs: &[Series], kwargs: LengthKwargs) -> PolarsResult<Series> {
    match kwargs.method {
        DistanceMethod::Planar => measure(&Planar, &inputs[0], kwargs.ignore_nan),
        DistanceMethod::Geodesic => {
            let metric = geodesic_metric(&crs_of(&inputs[0].field())?)?;
            measure(&*metric, &inputs[0], kwargs.ignore_nan)
        }
    }
}
