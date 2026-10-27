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
//! ## `UNIQUE_Z`
//!
//! Every kernel here is generic over `UNIQUE_Z`.
//! If `UNIQUE_Z=true`, two points only coincide when their z values do too.
//! A runtime `unique_z` is ignored for coordinates without a z.
//!
//! ## `UNIQUE_M`
//!
//! Every kernel here is generic over `UNIQUE_M`.
//! If `UNIQUE_M=true`, two points only coincide when their m values do too.
//! A runtime `unique_m` is ignored for coordinates without an m.

use std::ops::Range;

use geopolars_core::{
    is_close, same_place, same_place_within, LineStringArray, MultiLineStringArray,
    MultiPointArray, MultiPolygonArray, PointArray, PolygonArray, ScalarPolygon, Tolerance,
    ValidCoords,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::per_row::per_row_bool;

#[inline(always)]
pub(super) fn coincide<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    let m_coincides = || {
        let m = coords.m().expect("an m, with `UNIQUE_M`");
        if TOLERANCE_COINCIDENCE {
            is_close(m[i], m[j], tolerance)
        } else {
            m[i] == m[j]
        }
    };
    let place_coincides = if TOLERANCE_COINCIDENCE {
        same_place_within::<UNIQUE_Z>(coords, i, j, tolerance)
    } else {
        same_place::<UNIQUE_Z>(coords, i, j)
    };
    place_coincides && (!UNIQUE_M || m_coincides())
}

/// `unique_z` and `unique_m`, only for an axis `coords` has.
pub(super) fn unique_axes(coords: &ValidCoords, unique_z: bool, unique_m: bool) -> (bool, bool) {
    let dim = coords.dim();
    (unique_z && dim.has_z(), unique_m && dim.has_m())
}

/// Whether two vertices in a row of `vertices` [coincide].
#[inline(always)]
fn has_repeated_vertices<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    tolerance: Tolerance,
) -> bool {
    (vertices.start + 1..vertices.end)
        .any(|i| coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(coords, i - 1, i, tolerance))
}

/// Whether any two of `points` [coincide].
#[inline(always)]
pub(crate) fn has_repeated_points<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    coords: &ValidCoords,
    points: Range<usize>,
    tolerance: Tolerance,
) -> bool {
    points.clone().any(|i| {
        (i + 1..points.end)
            .any(|j| coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(coords, i, j, tolerance))
    })
}

/// Whether two vertices in a row of one of the rings of `polygon` [coincide].
#[inline(always)]
fn polygon_has_repeated_vertices<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    polygon: &ScalarPolygon,
    tolerance: Tolerance,
) -> bool {
    polygon.rings().any(|ring| {
        has_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            ring.coords,
            ring.vertices,
            tolerance,
        )
    })
}

/// A point has nothing to coincide with, so this is `false` for every point that is there.
pub fn points_have_repeated_points(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::new_zeroed(coords.ignoring_validity().len());
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

fn multi_points_have_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiPointArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        has_repeated_points::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.coords,
            array.rows.range(i),
            tolerance,
        )
    })
}

pub fn multi_points_have_repeated_points(
    array: &MultiPointArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_points_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), unique_z, unique_m
    )
}

fn line_strings_have_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &LineStringArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        has_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.coords,
            array.rows.range(i),
            tolerance,
        )
    })
}

pub fn line_strings_have_repeated_points(
    array: &LineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        line_strings_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), unique_z, unique_m
    )
}

fn multi_line_strings_have_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiLineStringArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array.rows.range(i).any(|line| {
            has_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
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
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_line_strings_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), unique_z, unique_m
    )
}

fn polygons_have_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &PolygonArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        polygon_has_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.get(i),
            tolerance,
        )
    })
}

pub fn polygons_have_repeated_points(
    array: &PolygonArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        polygons_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), unique_z, unique_m
    )
}

fn multi_polygons_have_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiPolygonArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array.get(i).polygons().any(|polygon| {
            polygon_has_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
                &polygon, tolerance,
            )
        })
    })
}

pub fn multi_polygons_have_repeated_points(
    array: &MultiPolygonArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> BooleanArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_polygons_have_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), unique_z, unique_m
    )
}
