//! Whether a geometry is valid, through `geopolars_compute::validity`.
//! See `is_valid` in Python for the rules.

use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_points, propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::Bounds;
use geopolars_compute::validity::{
    line_strings_are_valid, multi_line_strings_are_valid, multi_points_are_valid,
    multi_polygons_are_valid, points_are_valid, polygons_are_valid, CoordinateChecks,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, UncheckedStorage,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use crate::geoarrow::{describe, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsValidKwargs {
    bounds: Option<Bounds>,
    check_missing_coordinates: bool,
    check_finite_coordinates: bool,
}

impl Default for IsValidKwargs {
    fn default() -> Self {
        Self {
            bounds: None,
            check_missing_coordinates: true,
            check_finite_coordinates: true,
        }
    }
}

impl IsValidKwargs {
    fn checks(&self) -> CoordinateChecks {
        CoordinateChecks {
            bounds: self.bounds,
            check_finite_coordinates: self.check_finite_coordinates,
        }
    }
}

/// [`are_valid`] for one chunk of the storage of a `kind`.
///
/// If `check_missing_coordinates=true`, every missing value below a row is first moved out to the row
/// (see [geopolars_compute::inner_nulls]), so that the views can trust the rest.
fn chunk_is_valid(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanArray> {
    let propagate = kwargs.check_missing_coordinates;
    let unchecked = || UncheckedStorage::new(chunk);
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    let checks = kwargs.checks();

    let valid = match kind {
        Kind::Point => {
            let array = if propagate {
                propagate_inner_nulls_of_points(unchecked()?)
            } else {
                PointArray::new(Coords::new(chunk)?)
            };
            points_are_valid(&array, &checks)
        }
        Kind::LineString => {
            let array = if propagate {
                propagate_inner_nulls_of_line_strings(unchecked()?)
            } else {
                LineStringArray::new(rows()?)?
            };
            line_strings_are_valid(&array, &checks)
        }
        Kind::MultiPoint => {
            let array = if propagate {
                propagate_inner_nulls_of_multi_points(unchecked()?)
            } else {
                MultiPointArray::new(rows()?)?
            };
            multi_points_are_valid(&array, &checks)
        }
        Kind::MultiLineString => {
            let array = if propagate {
                propagate_inner_nulls_of_multi_line_strings(unchecked()?)
            } else {
                MultiLineStringArray::new(rows()?)?
            };
            multi_line_strings_are_valid(&array, &checks)
        }
        Kind::Polygon => {
            let array = if propagate {
                propagate_inner_nulls_of_polygons(unchecked()?)
            } else {
                PolygonArray::new(rows()?)?
            };
            polygons_are_valid(&array, &checks)
        }
        Kind::MultiPolygon => {
            let array = if propagate {
                propagate_inner_nulls_of_multi_polygons(unchecked()?)
            } else {
                MultiPolygonArray::new(rows()?)?
            };
            multi_polygons_are_valid(&array, &checks)
        }
    };
    // A missing row stays missing, and an incomplete one is invalid:
    // both are `false` in `valid`'s values.
    Ok(valid.with_validity(chunk.validity().cloned()))
}

pub(crate) fn are_valid(
    geometries: &Series,
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_valid(chunk.as_ref(), kind, kwargs));
    BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)
}

/// For [`is_valid`]: a `bool` per geometry.
fn kernel_output_type_is_valid(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_valid`.
#[polars_expr(output_type_func=kernel_output_type_is_valid)]
fn is_valid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    Ok(are_valid(&inputs[0], &kwargs)?.into_series())
}

/// `geometries`, missing where `keep` is `false`.
/// `keep`'s values alone say what stays: a missing row has to be `false` in them.
/// On a column where every row stays, it copies nothing.
pub(crate) fn keep_rows(geometries: &Series, keep: &BooleanChunked) -> PolarsResult<Series> {
    if keep.all() {
        return Ok(geometries.clone());
    }
    let geo = describe(geometries.dtype())?;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .zip(keep.downcast_iter())
        .map(|(chunk, keep)| chunk.with_validity(Some(keep.values().clone())))
        .collect();
    // SAFETY: only the outer validity changed, so every chunk is still the storage.
    let storage = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };
    Ok(storage.into_extension(geo.typ.clone()))
}

/// See `null_out_invalid`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn null_out_invalid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    // A missing row is `false` in the values.
    keep_rows(&inputs[0], &are_valid(&inputs[0], &kwargs)?)
}
