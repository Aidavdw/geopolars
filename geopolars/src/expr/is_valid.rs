//! Whether a geometry is valid, through `geopolars_compute::validity_checking`.
//! See `is_valid` in Python for the rules.

use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_points, propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::Bounds;
use geopolars_compute::validity_checking::{
    line_strings_are_valid, multi_line_strings_are_valid, multi_points_are_valid,
    multi_polygons_are_valid, points_are_valid, polygons_are_valid, ValidityChecks,
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
    check_line_string_zero_length_sections: bool,
    check_ring_self_intersect: bool,
    check_polygon_self_overlap: bool,
    check_multi_polygon_self_overlap: bool,
}

impl Default for IsValidKwargs {
    fn default() -> Self {
        Self {
            bounds: None,
            check_missing_coordinates: true,
            check_finite_coordinates: true,
            check_line_string_zero_length_sections: true,
            check_ring_self_intersect: true,
            check_polygon_self_overlap: true,
            check_multi_polygon_self_overlap: true,
        }
    }
}

impl IsValidKwargs {
    fn checks(&self) -> ValidityChecks {
        ValidityChecks {
            bounds: self.bounds,
            check_finite_coordinates: self.check_finite_coordinates,
            check_line_string_zero_length_sections: self.check_line_string_zero_length_sections,
            check_ring_self_intersect: self.check_ring_self_intersect,
            check_polygon_self_overlap: self.check_polygon_self_overlap,
            check_multi_polygon_self_overlap: self.check_multi_polygon_self_overlap,
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

/// See `null_out_invalid`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn null_out_invalid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let valid = are_valid(geometries, &kwargs)?;
    // Missing rows do not count, and stay missing.
    if valid.all() {
        return Ok(geometries.clone());
    }

    let geo = describe(geometries.dtype())?;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .zip(valid.downcast_iter())
        .map(|(chunk, valid)| {
            // A missing row is `false` in `valid`'s values, so they alone say what stays.
            chunk.with_validity(Some(valid.values().clone()))
        })
        .collect();
    // SAFETY: only the outer validity changed, so every chunk is still the storage.
    let storage = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };
    Ok(storage.into_extension(geo.typ.clone()))
}
