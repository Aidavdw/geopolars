//! Whether each geometry is valid: GeoArrow's rules, ours on top of them, and the coordinate checks.
//! See `is_valid` in Python for the rules.
//!
//! The coordinates are checked first, in one pass that does not know the geometry.
//! Then every row is checked against that, and against the shape rules of its kind.
//! Each gives a `bool` per row, `false` for a missing row.
//!
//! Missing values below the rows are not looked at here:
//! the views have to be built with them moved out to the rows (see [crate::inner_nulls]).

use std::ops::Range;

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ValidCoords,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::native::{
    coordinate_is_valid, line_string_shape_is_valid, polygon_shape_is_valid, Bounds,
};
use crate::per_row::per_row_bool;

/// The coordinate checks to run. See `is_valid` in Python.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateChecks {
    /// Where x and y have to lie, if anywhere.
    pub bounds: Option<Bounds>,
    pub check_finite_coordinates: bool,
}

impl CoordinateChecks {
    pub fn check_crs_bounds(&self) -> bool {
        self.bounds.is_some()
    }
}

/// See [coordinate_is_valid] for the flags.
fn coordinates_are_valid_with<
    const CHECK_FINITE_COORDINATES: bool,
    const CHECK_CRS_BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    bounds: &Bounds,
) -> Bitmap {
    Bitmap::from_iter((0..coords.len()).map(|i| {
        coordinate_is_valid::<CHECK_FINITE_COORDINATES, CHECK_CRS_BOUNDS, Z, M>(coords, i, bounds)
    }))
}

/// Which coordinates of `coords` pass `checks`.
/// `None` when there is nothing to check.
fn coordinates_are_valid(coords: &ValidCoords, checks: &CoordinateChecks) -> Option<Bitmap> {
    if !checks.check_finite_coordinates && !checks.check_crs_bounds() {
        return None;
    }
    let dim = coords.dim();
    let bounds = checks.bounds.unwrap_or_default();
    Some(with_const_flags!(
        coordinates_are_valid_with(coords, &bounds);
        checks.check_finite_coordinates, checks.check_crs_bounds(), dim.has_z(), dim.has_m()
    ))
}

/// Whether every coordinate in `span` passed, by [coordinates_are_valid].
#[inline(always)]
fn all_coordinates_valid(valid: Option<&Bitmap>, span: Range<usize>) -> bool {
    valid.is_none_or(|valid| valid.null_count_range(span.start, span.len()) == 0)
}

pub fn points_are_valid(array: &PointArray, checks: &CoordinateChecks) -> BooleanArray {
    let coords = &array.coords;
    let len = coords.ignoring_validity().len();
    let valid = coordinates_are_valid(coords.ignoring_validity(), checks)
        .unwrap_or_else(|| Bitmap::new_with_value(true, len));
    // A missing row is `false` in the values too.
    let values = match coords.validity() {
        Some(rows) => &valid & rows,
        None => valid,
    };
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn multi_points_are_valid(array: &MultiPointArray, checks: &CoordinateChecks) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    per_row_bool(&array.rows, |i| {
        all_coordinates_valid(valid.as_ref(), array.rows.range(i))
    })
}

pub fn line_strings_are_valid(array: &LineStringArray, checks: &CoordinateChecks) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    per_row_bool(&array.rows, |i| {
        let vertices = array.rows.range(i);
        line_string_shape_is_valid(vertices.clone())
            && all_coordinates_valid(valid.as_ref(), vertices)
    })
}

pub fn multi_line_strings_are_valid(
    array: &MultiLineStringArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    per_row_bool(&array.rows, |i| {
        let parts = array.rows.range(i);
        parts
            .clone()
            .all(|line| line_string_shape_is_valid(array.lines.range(line)))
            && all_coordinates_valid(valid.as_ref(), array.lines.span(parts))
    })
}

/// See [polygon_shape_is_valid] for `Z`.
fn polygons_are_valid_with<const Z: bool>(
    array: &PolygonArray,
    valid: Option<&Bitmap>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        polygon_shape_is_valid::<Z>(&array.get(i))
            && all_coordinates_valid(valid, array.rings.span(array.rows.range(i)))
    })
}

pub fn polygons_are_valid(array: &PolygonArray, checks: &CoordinateChecks) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    let z = array.coords.dim().has_z();
    with_const_flags!(polygons_are_valid_with(array, valid.as_ref()); z)
}

/// See [polygon_shape_is_valid] for `Z`.
fn multi_polygons_are_valid_with<const Z: bool>(
    array: &MultiPolygonArray,
    valid: Option<&Bitmap>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.rings.span(array.polygons.span(array.rows.range(i)));
        array
            .get(i)
            .polygons()
            .all(|polygon| polygon_shape_is_valid::<Z>(&polygon))
            && all_coordinates_valid(valid, vertices)
    })
}

pub fn multi_polygons_are_valid(
    array: &MultiPolygonArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    let z = array.coords.dim().has_z();
    with_const_flags!(multi_polygons_are_valid_with(array, valid.as_ref()); z)
}
