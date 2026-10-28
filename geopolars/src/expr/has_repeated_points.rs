//! Whether a geometry has two points that coincide,
//! through `geopolars_compute::native::*_have_repeated_points`.
//! See `has_repeated_points` in Python for the rules.

use geopolars_compute::native::{
    line_strings_have_repeated_points, multi_line_strings_have_repeated_points,
    multi_points_have_repeated_points, multi_polygons_have_repeated_points,
    points_have_repeated_points, polygons_have_repeated_points,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, Tolerance,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HasRepeatedPointsKwargs {
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
    unique_z: bool,
    unique_m: bool,
}

/// `None` when neither tolerance is given; otherwise the one not given is `0.0`.
pub(super) fn tolerance(abs_tol: Option<f64>, rel_tol: Option<f64>) -> Option<Tolerance> {
    (abs_tol.is_some() || rel_tol.is_some()).then(|| Tolerance {
        abs_tol: abs_tol.unwrap_or(0.0),
        rel_tol: rel_tol.unwrap_or(0.0),
    })
}

/// [`has_repeated_points`] for one chunk of the storage of a `kind`.
fn chunk_has_repeated_points(
    chunk: &dyn Array,
    kind: Kind,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_have_repeated_points(&PointArray::new(Coords::new(chunk)?)),
        Kind::LineString => line_strings_have_repeated_points(
            &LineStringArray::new(rows()?)?,
            tolerance,
            unique_z,
            unique_m,
        ),
        Kind::MultiPoint => multi_points_have_repeated_points(
            &MultiPointArray::new(rows()?)?,
            tolerance,
            unique_z,
            unique_m,
        ),
        Kind::MultiLineString => multi_line_strings_have_repeated_points(
            &MultiLineStringArray::new(rows()?)?,
            tolerance,
            unique_z,
            unique_m,
        ),
        Kind::Polygon => polygons_have_repeated_points(
            &PolygonArray::new(rows()?)?,
            tolerance,
            unique_z,
            unique_m,
        ),
        Kind::MultiPolygon => multi_polygons_have_repeated_points(
            &MultiPolygonArray::new(rows()?)?,
            tolerance,
            unique_z,
            unique_m,
        ),
    })
}

/// For [`has_repeated_points`]: a `bool` per geometry.
fn kernel_output_type_has_repeated_points(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `has_repeated_points`.
#[polars_expr(output_type_func=kernel_output_type_has_repeated_points)]
fn has_repeated_points(inputs: &[Series], kwargs: HasRepeatedPointsKwargs) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let kind = describe(geometries.dtype())?.kind;
    let tolerance = tolerance(kwargs.abs_tol, kwargs.rel_tol);
    let storage = geometries.ext()?.storage();
    let chunks = storage.chunks().iter().map(|chunk| {
        chunk_has_repeated_points(
            chunk.as_ref(),
            kind,
            tolerance,
            kwargs.unique_z,
            kwargs.unique_m,
        )
    });
    Ok(BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)?.into_series())
}
