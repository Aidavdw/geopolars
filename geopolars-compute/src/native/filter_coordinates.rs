//! Emptying, dropping or nulling the coordinates that are not finite, or empty.
//! See `filter_coordinates` in Python for the rules.
//!
//! Every kernel marks the selected coordinates ([selected_coords]),
//! then acts on them as in [super::selected_coords].

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ValidCoords,
};

use super::selected_coords::{
    line_strings_with_selected, multi_line_strings_with_selected, multi_points_with_selected,
    multi_polygons_with_selected, points_with_selected, polygons_with_selected, SelectedAction,
};
use super::too_short::TooShort;

/// See `NonFinite` in Python.
#[derive(Debug, Clone, Copy)]
pub struct NonFinite {
    pub select_non_finite: bool,
    pub select_empty: bool,
    pub ignore_z: bool,
    pub ignore_m: bool,
    pub action: SelectedAction,
    pub too_short: TooShort,
}

/// Whether `filter` selects each of `coords`.
fn selected_coords(coords: &ValidCoords, filter: &NonFinite) -> Vec<bool> {
    let (x, y) = (coords.x(), coords.y());
    let mut non_finite: Vec<bool> = x
        .iter()
        .zip(y)
        .map(|(x, y)| !(x.is_finite() & y.is_finite()))
        .collect();
    let mut empty: Vec<bool> = x
        .iter()
        .zip(y)
        .map(|(x, y)| x.is_nan() & y.is_nan())
        .collect();
    for (axis, ignored) in [(coords.z(), filter.ignore_z), (coords.m(), filter.ignore_m)] {
        let Some(axis) = axis else { continue };
        for ((non_finite, empty), value) in non_finite.iter_mut().zip(&mut empty).zip(axis) {
            *non_finite |= !ignored & !value.is_finite();
            *empty &= value.is_nan();
        }
    }
    let NonFinite {
        select_non_finite,
        select_empty,
        ..
    } = *filter;
    non_finite
        .iter()
        .zip(&empty)
        .map(|(&non_finite, &empty)| {
            (select_non_finite & non_finite & !empty) | (select_empty & empty)
        })
        .collect()
}

pub fn points_with_filtered_coordinates(array: &PointArray, filter: &NonFinite) -> PointArray {
    let selected = selected_coords(array.coords.ignoring_validity(), filter);
    points_with_selected(array, &selected, filter.action)
}

pub fn line_strings_with_filtered_coordinates(
    array: &LineStringArray,
    filter: &NonFinite,
) -> LineStringArray {
    let selected = selected_coords(&array.coords, filter);
    line_strings_with_selected(array, &selected, filter.action, filter.too_short)
}

pub fn multi_points_with_filtered_coordinates(
    array: &MultiPointArray,
    filter: &NonFinite,
) -> MultiPointArray {
    let selected = selected_coords(&array.coords, filter);
    multi_points_with_selected(array, &selected, filter.action, filter.too_short)
}

pub fn polygons_with_filtered_coordinates(
    array: &PolygonArray,
    filter: &NonFinite,
) -> PolygonArray {
    let selected = selected_coords(&array.coords, filter);
    polygons_with_selected(array, &selected, filter.action, filter.too_short)
}

pub fn multi_line_strings_with_filtered_coordinates(
    array: &MultiLineStringArray,
    filter: &NonFinite,
) -> MultiLineStringArray {
    let selected = selected_coords(&array.coords, filter);
    multi_line_strings_with_selected(array, &selected, filter.action, filter.too_short)
}

pub fn multi_polygons_with_filtered_coordinates(
    array: &MultiPolygonArray,
    filter: &NonFinite,
) -> MultiPolygonArray {
    let selected = selected_coords(&array.coords, filter);
    multi_polygons_with_selected(array, &selected, filter.action, filter.too_short)
}
