//! Whether a geometry has two points that coincide.
//! See `has_repeated_points` in Python for the rules.
//!
//! # Const generic flags
//!
//! ## `TOLERANCE_COINCIDENCE`
//!
//! Every kernel here is generic over `TOLERANCE_COINCIDENCE`.
//! If `TOLERANCE_COINCIDENCE=true`, two points coincide when they lie at the same place within `tolerance`
//! ([same_place_within]).
//! If `TOLERANCE_COINCIDENCE=false`, they have to lie at exactly the same place ([same_place]).
//!
//! ## `Z`
//!
//! Every kernel here is generic over `Z`.
//! If `Z=true`, two points only coincide when their z values do too.

use std::ops::Range;

use geopolars_core::{
    same_place, same_place_within, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ScalarPolygon, Tolerance, ValidCoords,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::per_row::per_row_bool;

#[inline(always)]
fn coincide<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    if TOLERANCE_COINCIDENCE {
        same_place_within::<Z>(coords, i, j, tolerance)
    } else {
        same_place::<Z>(coords, i, j)
    }
}

/// Whether two vertices in a row of `vertices` [coincide].
#[inline(always)]
fn has_repeated_vertices<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    tolerance: Tolerance,
) -> bool {
    (vertices.start + 1..vertices.end)
        .any(|i| coincide::<TOLERANCE_COINCIDENCE, Z>(coords, i - 1, i, tolerance))
}

/// Whether any two of `points` [coincide].
#[inline(always)]
pub(crate) fn has_repeated_points<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    coords: &ValidCoords,
    points: Range<usize>,
    tolerance: Tolerance,
) -> bool {
    points.clone().any(|i| {
        (i + 1..points.end).any(|j| coincide::<TOLERANCE_COINCIDENCE, Z>(coords, i, j, tolerance))
    })
}

/// Whether two vertices in a row of one of the rings of `polygon` [coincide].
#[inline(always)]
fn polygon_has_repeated_vertices<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    polygon: &ScalarPolygon,
    tolerance: Tolerance,
) -> bool {
    polygon.rings().any(|ring| {
        has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(ring.coords, ring.vertices, tolerance)
    })
}

/// A point has nothing to coincide with, so this is `false` for every point that is there.
pub fn points_have_repeated_points(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::new_zeroed(coords.ignoring_validity().len());
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

fn multi_points_have_repeated_points_with<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    array: &MultiPointArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        has_repeated_points::<TOLERANCE_COINCIDENCE, Z>(
            &array.coords,
            array.rows.range(i),
            tolerance,
        )
    })
}

pub fn multi_points_have_repeated_points(
    array: &MultiPointArray,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_points_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), z
    )
}

fn line_strings_have_repeated_points_with<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    array: &LineStringArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(
            &array.coords,
            array.rows.range(i),
            tolerance,
        )
    })
}

pub fn line_strings_have_repeated_points(
    array: &LineStringArray,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        line_strings_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), z
    )
}

fn multi_line_strings_have_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const Z: bool,
>(
    array: &MultiLineStringArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array.rows.range(i).any(|line| {
            has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(
                &array.coords,
                array.lines.range(line),
                tolerance,
            )
        })
    })
}

pub fn multi_line_strings_have_repeated_points(
    array: &MultiLineStringArray,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_line_strings_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), z
    )
}

fn polygons_have_repeated_points_with<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    array: &PolygonArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        polygon_has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(&array.get(i), tolerance)
    })
}

pub fn polygons_have_repeated_points(
    array: &PolygonArray,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        polygons_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), z
    )
}

fn multi_polygons_have_repeated_points_with<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    array: &MultiPolygonArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array.get(i).polygons().any(|polygon| {
            polygon_has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(&polygon, tolerance)
        })
    })
}

pub fn multi_polygons_have_repeated_points(
    array: &MultiPolygonArray,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_polygons_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), z
    )
}
