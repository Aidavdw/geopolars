//! Whether a geometry is valid, by running the checks of
//! `geopolars_compute::native` and `geopolars_compute::rsgeo` one after another.
//! See `is_valid` in Python for the rules.

use geopolars_compute::native::{
    line_strings_have_valid_coordinates, line_strings_have_valid_shape,
    multi_line_strings_have_valid_coordinates, multi_line_strings_have_valid_shape,
    multi_points_have_valid_coordinates, multi_polygons_have_valid_coordinates,
    multi_polygons_have_valid_shape, points_have_valid_coordinates,
    polygons_have_valid_coordinates, polygons_have_valid_shape, Bounds,
};
use geopolars_compute::rsgeo::{multi_polygons_fit, polygons_fit, FitChecks};
use geopolars_core::{
    complete_rows, downcast, Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ValidCoords,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
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
    // Relating rings needs them finite, so with the fit checks on, they always are checked.
    fn has_finite_check_for(&self, kind: Kind) -> bool {
        self.check_finite_coordinates || self.has_fit_checks_for(kind)
    }

    fn has_fit_checks_for(&self, kind: Kind) -> bool {
        let polygon = self.check_ring_self_intersect || self.check_polygon_self_overlap;
        match kind {
            Kind::Polygon => polygon,
            Kind::MultiPolygon => polygon || self.check_multi_polygon_self_overlap,
            Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => false,
        }
    }

    fn fit_checks(&self) -> FitChecks {
        FitChecks {
            check_ring_self_intersect: self.check_ring_self_intersect,
            check_polygon_self_overlap: self.check_polygon_self_overlap,
            check_multi_polygon_self_overlap: self.check_multi_polygon_self_overlap,
        }
    }
}

/// A missing row is `false` in `passed`'s values, so they alone say what stays.
fn keep_rows_that_passed(rows: &List, passed: BooleanArray) -> List {
    rows.clone().with_validity(Some(passed.values().clone()))
}

fn rows_still_valid_or_null_where_chunk_is(
    still_valid: Option<Bitmap>,
    chunk: &dyn Array,
) -> BooleanArray {
    let values = still_valid.unwrap_or_else(|| Bitmap::new_with_value(true, chunk.len()));
    BooleanArray::new(ArrowDataType::Boolean, values, chunk.validity().cloned())
}

/// [`are_valid`] for one chunk of the storage of a `kind`.
///
/// With `check_missing_coordinates`, a row missing anything below it is first set missing
/// (see [complete_rows]), so that the views can trust the rest.
/// Then every check that is on runs as its own pass, over the rows that passed so far.
fn chunk_is_valid(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanArray> {
    let complete = if kwargs.check_missing_coordinates {
        complete_rows(chunk, kind.nesting())?
    } else {
        chunk.validity().cloned()
    };
    let (finite, bounds) = (kwargs.has_finite_check_for(kind), kwargs.bounds.as_ref());
    let coordinates = finite || bounds.is_some();
    let zero = kwargs.check_line_string_zero_length_sections;
    let fit = kwargs.has_fit_checks_for(kind);

    if kind == Kind::Point {
        if !coordinates {
            return Ok(rows_still_valid_or_null_where_chunk_is(complete, chunk));
        }
        let array = PointArray::new(Coords::from_valid(ValidCoords::new(chunk)?, complete));
        let valid = points_have_valid_coordinates(&array, finite, bounds);
        return Ok(valid.with_validity(chunk.validity().cloned()));
    }

    let rows: &ListArray<i64> = downcast(chunk, "a list of parts")?;
    let rows = match kind {
        Kind::Point => unreachable!("handled above"),
        Kind::LineString => {
            let mut array = LineStringArray::new(rows)?;
            array.rows = array.rows.with_validity(complete);
            if coordinates {
                array.rows = keep_rows_that_passed(
                    &array.rows,
                    line_strings_have_valid_coordinates(&array, finite, bounds),
                );
            }
            array.rows =
                keep_rows_that_passed(&array.rows, line_strings_have_valid_shape(&array, zero));
            array.rows
        }
        Kind::MultiPoint => {
            let mut array = MultiPointArray::new(rows)?;
            array.rows = array.rows.with_validity(complete);
            if coordinates {
                array.rows = keep_rows_that_passed(
                    &array.rows,
                    multi_points_have_valid_coordinates(&array, finite, bounds),
                );
            }
            array.rows
        }
        Kind::MultiLineString => {
            let mut array = MultiLineStringArray::new(rows)?;
            array.rows = array.rows.with_validity(complete);
            if coordinates {
                array.rows = keep_rows_that_passed(
                    &array.rows,
                    multi_line_strings_have_valid_coordinates(&array, finite, bounds),
                );
            }
            array.rows = keep_rows_that_passed(
                &array.rows,
                multi_line_strings_have_valid_shape(&array, zero),
            );
            array.rows
        }
        Kind::Polygon => {
            let mut array = PolygonArray::new(rows)?;
            array.rows = array.rows.with_validity(complete);
            if coordinates {
                array.rows = keep_rows_that_passed(
                    &array.rows,
                    polygons_have_valid_coordinates(&array, finite, bounds),
                );
            }
            array.rows =
                keep_rows_that_passed(&array.rows, polygons_have_valid_shape(&array, zero));
            if fit {
                array.rows =
                    keep_rows_that_passed(&array.rows, polygons_fit(&array, &kwargs.fit_checks()));
            }
            array.rows
        }
        Kind::MultiPolygon => {
            let mut array = MultiPolygonArray::new(rows)?;
            array.rows = array.rows.with_validity(complete);
            if coordinates {
                array.rows = keep_rows_that_passed(
                    &array.rows,
                    multi_polygons_have_valid_coordinates(&array, finite, bounds),
                );
            }
            array.rows =
                keep_rows_that_passed(&array.rows, multi_polygons_have_valid_shape(&array, zero));
            if fit {
                array.rows = keep_rows_that_passed(
                    &array.rows,
                    multi_polygons_fit(&array, &kwargs.fit_checks()),
                );
            }
            array.rows
        }
    };
    // A missing row stays missing, and an incomplete one is invalid.
    Ok(rows_still_valid_or_null_where_chunk_is(
        rows.validity().cloned(),
        chunk,
    ))
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
