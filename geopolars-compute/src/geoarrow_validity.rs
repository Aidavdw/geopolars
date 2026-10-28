//! Whether each geometry is valid by the GeoArrow spec, and holds no selected coordinate.
//! See `is_valid_geoarrow` in Python for the rules.
//!
//! The coordinates are selected first ([selected_coords]), in one pass that does not know the geometry.
//! Then every row is checked for holding one, and every ring for being closed.
//! Each gives a `bool` per row, `false` for a null row.
//!
//! Null values below the rows are not looked at here:
//! the views have to be built with them moved out to the rows (see [crate::inner_nulls]).
//!
//! # Const generic flags
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

use crate::native::{has_selected, polygon_rings_are_closed, selected_coords, CoordinateSelection};
use crate::per_row::per_row_bool;

pub fn points_are_valid_geoarrow(
    array: &PointArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let coords = &array.coords;
    let len = coords.ignoring_validity().len();
    let valid = selected_coords(coords.ignoring_validity(), selection)
        .map_or_else(|| Bitmap::new_with_value(true, len), |selected| !&selected);
    // A null row is `false` in the values too.
    let values = match coords.validity() {
        Some(rows) => &valid & rows,
        None => valid,
    };
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn multi_points_are_valid_geoarrow(
    array: &MultiPointArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let selected = selected_coords(&array.coords, selection);
    per_row_bool(&array.rows, |i| {
        !has_selected(selected.as_ref(), array.rows.range(i))
    })
}

pub fn line_strings_are_valid_geoarrow(
    array: &LineStringArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let selected = selected_coords(&array.coords, selection);
    per_row_bool(&array.rows, |i| {
        !has_selected(selected.as_ref(), array.rows.range(i))
    })
}

pub fn multi_line_strings_are_valid_geoarrow(
    array: &MultiLineStringArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let selected = selected_coords(&array.coords, selection);
    per_row_bool(&array.rows, |i| {
        !has_selected(selected.as_ref(), array.lines.span(array.rows.range(i)))
    })
}

fn polygons_are_valid_geoarrow_with<const Z: bool>(
    array: &PolygonArray,
    selected: Option<&Bitmap>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        polygon_rings_are_closed::<Z>(&array.get(i))
            && !has_selected(selected, array.rings.span(array.rows.range(i)))
    })
}

pub fn polygons_are_valid_geoarrow(
    array: &PolygonArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let selected = selected_coords(&array.coords, selection);
    let z = array.coords.dim().has_z();
    with_const_flags!(polygons_are_valid_geoarrow_with(array, selected.as_ref()); z)
}

fn multi_polygons_are_valid_geoarrow_with<const Z: bool>(
    array: &MultiPolygonArray,
    selected: Option<&Bitmap>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.rings.span(array.polygons.span(array.rows.range(i)));
        array
            .get(i)
            .polygons()
            .all(|polygon| polygon_rings_are_closed::<Z>(&polygon))
            && !has_selected(selected, vertices)
    })
}

pub fn multi_polygons_are_valid_geoarrow(
    array: &MultiPolygonArray,
    selection: &CoordinateSelection,
) -> BooleanArray {
    let selected = selected_coords(&array.coords, selection);
    let z = array.coords.dim().has_z();
    with_const_flags!(multi_polygons_are_valid_geoarrow_with(array, selected.as_ref()); z)
}
