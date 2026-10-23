//! Whether each geometry is valid by the GeoArrow spec, and passes the coordinate checks.
//! See `is_valid_geoarrow` in Python for the rules.
//!
//! The coordinates are checked first, in one pass that does not know the geometry.
//! Then every row is checked against that, and every ring for being closed.
//! Each gives a `bool` per row, `false` for a missing row.
//!
//! Missing values below the rows are not looked at here:
//! the views have to be built with them moved out to the rows (see [crate::inner_nulls]).
//!
//! # Const generic flags
//!
//! See [crate::native] (`is_valid`) for those of the coordinate checks.
//!
//! ## `Z`
//!
//! The polygon kernels are generic over `Z`.
//! If `Z=true`, a ring has to close in z as well.

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::native::{
    all_coordinates_valid, coordinates_are_valid, polygon_rings_are_closed, CoordinateChecks,
};
use crate::per_row::per_row_bool;

pub fn points_are_valid_geoarrow(array: &PointArray, checks: &CoordinateChecks) -> BooleanArray {
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

pub fn multi_points_are_valid_geoarrow(
    array: &MultiPointArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    per_row_bool(&array.rows, |i| {
        all_coordinates_valid(valid.as_ref(), array.rows.range(i))
    })
}

pub fn line_strings_are_valid_geoarrow(
    array: &LineStringArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    per_row_bool(&array.rows, |i| {
        all_coordinates_valid(valid.as_ref(), array.rows.range(i))
    })
}

pub fn multi_line_strings_are_valid_geoarrow(
    array: &MultiLineStringArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    per_row_bool(&array.rows, |i| {
        all_coordinates_valid(valid.as_ref(), array.lines.span(array.rows.range(i)))
    })
}

fn polygons_are_valid_geoarrow_with<const Z: bool>(
    array: &PolygonArray,
    valid: Option<&Bitmap>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        polygon_rings_are_closed::<Z>(&array.get(i))
            && all_coordinates_valid(valid, array.rings.span(array.rows.range(i)))
    })
}

pub fn polygons_are_valid_geoarrow(
    array: &PolygonArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    let z = array.coords.dim().has_z();
    with_const_flags!(polygons_are_valid_geoarrow_with(array, valid.as_ref()); z)
}

fn multi_polygons_are_valid_geoarrow_with<const Z: bool>(
    array: &MultiPolygonArray,
    valid: Option<&Bitmap>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.rings.span(array.polygons.span(array.rows.range(i)));
        array
            .get(i)
            .polygons()
            .all(|polygon| polygon_rings_are_closed::<Z>(&polygon))
            && all_coordinates_valid(valid, vertices)
    })
}

pub fn multi_polygons_are_valid_geoarrow(
    array: &MultiPolygonArray,
    checks: &CoordinateChecks,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, checks);
    let z = array.coords.dim().has_z();
    with_const_flags!(multi_polygons_are_valid_geoarrow_with(array, valid.as_ref()); z)
}
