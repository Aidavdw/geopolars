//! Whether each geometry is degenerate: points repeated, empty, or rings that do not fit together.
//! See `is_degenerate` in Python for the rules.
//!
//! The views have to hold valid geometries (see [crate::validity]).
//! Each gives a `bool` per row, `true` where the geometry is degenerate, and `false` for a missing row.

use std::ops::Range;

use geopolars_core::{
    same_place, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, ValidCoords,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::native::is_empty_point;
use crate::per_row::per_row_bool;
use crate::rsgeo::{multi_polygon_topology_is_valid, polygon_topology_is_valid};
pub use crate::rsgeo::{MultiPolygonTopologyChecks, PolygonTopologyChecks};

/// Whether two vertices in a row of `vertices` lie at the [same place](same_place).
#[inline(always)]
fn has_repeated_vertices<const Z: bool>(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    (vertices.start + 1..vertices.end).any(|i| same_place::<Z>(coords, i - 1, i))
}

/// Whether any two of `points` lie at the [same place](same_place).
#[inline(always)]
fn has_repeated_points<const Z: bool>(coords: &ValidCoords, points: Range<usize>) -> bool {
    points
        .clone()
        .any(|i| (i + 1..points.end).any(|j| same_place::<Z>(coords, i, j)))
}

/// Whether every one of `vertices` is the empty point, which also holds when there are none.
#[inline(always)]
fn only_empty_points(coords: &ValidCoords, mut vertices: Range<usize>) -> bool {
    vertices.all(|i| is_empty_point(coords, i))
}

/// If `CHECK_EMPTY=true`, the empty point is degenerate.
fn points_are_degenerate_with<const CHECK_EMPTY: bool>(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::from_iter((0..coords.ignoring_validity().len()).map(|i| {
        CHECK_EMPTY && coords.is_valid(i) && is_empty_point(coords.ignoring_validity(), i)
    }));
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn points_are_degenerate(array: &PointArray, check_empty: bool) -> BooleanArray {
    with_const_flags!(points_are_degenerate_with(array); check_empty)
}

/// If `CHECK_REPEATED_POINTS=true`, two points at the same place are degenerate.
/// If `CHECK_EMPTY=true`, so is a multipoint of only empty points, or none.
/// See [same_place] for `Z`.
fn multi_points_are_degenerate_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const Z: bool,
>(
    array: &MultiPointArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let points = array.rows.range(i);
        (CHECK_REPEATED_POINTS && has_repeated_points::<Z>(&array.coords, points.clone()))
            || (CHECK_EMPTY && only_empty_points(&array.coords, points))
    })
}

pub fn multi_points_are_degenerate(
    array: &MultiPointArray,
    check_repeated_points: bool,
    check_empty: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(multi_points_are_degenerate_with(array); check_repeated_points, check_empty, z)
}

/// If `CHECK_REPEATED_POINTS=true`, two vertices in a row at the same place are degenerate.
/// If `CHECK_EMPTY=true`, so is a linestring of only empty points, or none.
/// See [same_place] for `Z`.
fn line_strings_are_degenerate_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const Z: bool,
>(
    array: &LineStringArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.rows.range(i);
        (CHECK_REPEATED_POINTS && has_repeated_vertices::<Z>(&array.coords, vertices.clone()))
            || (CHECK_EMPTY && only_empty_points(&array.coords, vertices))
    })
}

pub fn line_strings_are_degenerate(
    array: &LineStringArray,
    check_repeated_points: bool,
    check_empty: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(line_strings_are_degenerate_with(array); check_repeated_points, check_empty, z)
}

/// [line_strings_are_degenerate_with] for every part.
/// If `CHECK_EMPTY=true`, it is degenerate when every part is empty, or there are none.
fn multi_line_strings_are_degenerate_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const Z: bool,
>(
    array: &MultiLineStringArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let parts = array.rows.range(i);
        (CHECK_REPEATED_POINTS
            && parts
                .clone()
                .any(|line| has_repeated_vertices::<Z>(&array.coords, array.lines.range(line))))
            || (CHECK_EMPTY && only_empty_points(&array.coords, array.lines.span(parts)))
    })
}

pub fn multi_line_strings_are_degenerate(
    array: &MultiLineStringArray,
    check_repeated_points: bool,
    check_empty: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_line_strings_are_degenerate_with(array);
        check_repeated_points, check_empty, z
    )
}

/// If `CHECK_REPEATED_POINTS=true`, two vertices in a row of a ring at the same place are degenerate.
/// If `CHECK_EMPTY=true`, so is a polygon of only empty rings, or none.
/// So is a polygon that fails one of the `topology` checks, if there are any.
/// See [same_place] for `Z`.
fn polygons_are_degenerate_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const Z: bool,
>(
    array: &PolygonArray,
    topology: Option<&PolygonTopologyChecks>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let polygon = array.get(i);
        let vertices = array.rings.span(array.rows.range(i));
        (CHECK_REPEATED_POINTS
            && polygon
                .rings()
                .any(|ring| has_repeated_vertices::<Z>(ring.coords, ring.vertices)))
            || (CHECK_EMPTY && only_empty_points(&array.coords, vertices))
            || topology.is_some_and(|checks| !polygon_topology_is_valid(&polygon, checks))
    })
}

pub fn polygons_are_degenerate(
    array: &PolygonArray,
    check_repeated_points: bool,
    check_empty: bool,
    topology: &PolygonTopologyChecks,
) -> BooleanArray {
    let topology = topology.checks_anything().then_some(topology);
    let z = array.coords.dim().has_z();
    with_const_flags!(
        polygons_are_degenerate_with(array, topology);
        check_repeated_points, check_empty, z
    )
}

/// [polygons_are_degenerate_with] for every part.
/// If `CHECK_EMPTY=true`, it is degenerate when every part is empty, or there are none.
/// So is a multipolygon that fails one of the `topology` checks, if there are any.
fn multi_polygons_are_degenerate_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const Z: bool,
>(
    array: &MultiPolygonArray,
    topology: Option<&MultiPolygonTopologyChecks>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let multi = array.get(i);
        let vertices = array.rings.span(array.polygons.span(array.rows.range(i)));
        (CHECK_REPEATED_POINTS
            && multi.polygons().any(|polygon| {
                polygon
                    .rings()
                    .any(|ring| has_repeated_vertices::<Z>(ring.coords, ring.vertices))
            }))
            || (CHECK_EMPTY && only_empty_points(&array.coords, vertices))
            || topology.is_some_and(|checks| !multi_polygon_topology_is_valid(&multi, checks))
    })
}

pub fn multi_polygons_are_degenerate(
    array: &MultiPolygonArray,
    check_repeated_points: bool,
    check_empty: bool,
    topology: &MultiPolygonTopologyChecks,
) -> BooleanArray {
    let topology = topology.checks_anything().then_some(topology);
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_polygons_are_degenerate_with(array, topology);
        check_repeated_points, check_empty, z
    )
}
