//! Whether each geometry is malformed: points repeated, empty, or rings that do not fit together.
//! See `is_malformed` in Python for the rules.
//!
//! The views have to hold valid geometries (see [crate::validity]).
//! Each gives a `bool` per row, `true` where the geometry is malformed, and `false` for a missing row.
//! The polygon kernels error when GEOS does.
//! With every topology check on, GEOS's `is_valid` runs them all at once.
//! Otherwise, each check runs on its own.
//!
//! # Const generic flags
//!
//! ## `CHECK_EMPTY`
//!
//! Every kernel here is generic over `CHECK_EMPTY`.
//! If `CHECK_EMPTY=true`, an empty geometry is malformed.
//! A point is empty when it is the empty point (NaN in x and y).
//! A multipoint or linestring is empty when it has only empty points, or none.
//! A polygon is empty when it has only empty rings, or none.
//! A multilinestring or multipolygon is empty when every part is, or it has none.
//!
//! ## `CHECK_REPEATED_POINTS`
//!
//! The kernels of every kind but points are generic over `CHECK_REPEATED_POINTS`.
//! If `CHECK_REPEATED_POINTS=true`, a geometry with two points that coincide is malformed.
//! In a multipoint, those are any two of its points.
//! In a linestring, those are two vertices in a row.
//! In a multilinestring, those are two vertices in a row of one part.
//! In a polygon or multipolygon, those are two vertices in a row of one ring.
//!
//! ## `TOLERANCE_COINCIDENCE`
//!
//! The kernels of every kind but points are generic over `TOLERANCE_COINCIDENCE`.
//! If `TOLERANCE_COINCIDENCE=true`, two points coincide when they lie at the same place within `tolerance`
//! ([same_place_within]).
//! If `TOLERANCE_COINCIDENCE=false`, they have to lie at exactly the same place ([same_place]).
//!
//! ## `Z`
//!
//! The kernels of every kind but points are generic over `Z`.
//! If `Z=true`, two points only coincide when their z values do too.

use std::ops::Range;

use geopolars_core::{
    same_place, same_place_within, LineStringArray, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ScalarMultiPolygon, ScalarPolygon, Tolerance,
    ValidCoords,
};
use geos::{Geom, Geometry};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::geos::{meet_only_at_points_and_no_overlap, polygon_rings_fit, ToGeos};
use crate::native::{is_empty_point, only_empty_points};
use crate::per_row::{per_row_bool, try_per_row_bool};
use crate::GeoComputeResult;

/// The topology checks to run on a polygon. See `is_malformed` in Python.
#[derive(Debug, Clone, Copy)]
pub struct PolygonTopologyChecks {
    pub check_ring_self_intersect: bool,
    pub check_polygon_self_overlap: bool,
}

impl PolygonTopologyChecks {
    pub fn checks_anything(&self) -> bool {
        self.check_ring_self_intersect || self.check_polygon_self_overlap
    }

    pub fn checks_everything(&self) -> bool {
        self.check_ring_self_intersect && self.check_polygon_self_overlap
    }
}

/// The topology checks to run on a multipolygon. See `is_malformed` in Python.
#[derive(Debug, Clone, Copy)]
pub struct MultiPolygonTopologyChecks {
    /// Every part is checked for these individually
    pub polygons: PolygonTopologyChecks,
    pub check_multi_polygon_self_overlap: bool,
}

impl MultiPolygonTopologyChecks {
    pub fn checks_anything(&self) -> bool {
        self.polygons.checks_anything() || self.check_multi_polygon_self_overlap
    }

    pub fn checks_everything(&self) -> bool {
        self.polygons.checks_everything() && self.check_multi_polygon_self_overlap
    }
}

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
fn has_repeated_points<const TOLERANCE_COINCIDENCE: bool, const Z: bool>(
    coords: &ValidCoords,
    points: Range<usize>,
    tolerance: Tolerance,
) -> bool {
    points.clone().any(|i| {
        (i + 1..points.end).any(|j| coincide::<TOLERANCE_COINCIDENCE, Z>(coords, i, j, tolerance))
    })
}

fn xy_is_finite(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    let (x, y) = (&coords.x()[vertices.clone()], &coords.y()[vertices]);
    x.iter().chain(y).all(|value| value.is_finite())
}

fn is_empty_polygon(polygon: &ScalarPolygon) -> bool {
    polygon
        .rings()
        .next()
        .is_none_or(|exterior| exterior.vertices.is_empty())
}

fn needs_whole_polygon_in_geos(checks: &PolygonTopologyChecks) -> bool {
    checks.checks_everything() || checks.check_ring_self_intersect
}

fn whole_polygon_passes(
    geometry: &Geometry,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    Ok(if checks.checks_everything() {
        geometry.is_valid()?
    } else {
        !checks.check_ring_self_intersect || geometry.is_simple()?
    })
}

fn rings_fit_if_checked_separately(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    // A lone exterior has nothing to fit with.
    let fit = checks.check_polygon_self_overlap && !checks.checks_everything();
    Ok(!fit || polygon.parts.len() < 2 || polygon_rings_fit(polygon)?)
}

/// [polygon_topology_is_valid] for a `polygon` that is not empty, and finite in x and y.
fn finite_polygon_topology_is_valid(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    if needs_whole_polygon_in_geos(checks) && !whole_polygon_passes(&polygon.to_geos()?, checks)? {
        return Ok(false);
    }
    rings_fit_if_checked_separately(polygon, checks)
}

fn polygon_topology_is_valid(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    if is_empty_polygon(polygon) {
        return Ok(true);
    }
    if !xy_is_finite(polygon.coords, polygon.rings.span(polygon.parts.clone())) {
        return Ok(false);
    }
    finite_polygon_topology_is_valid(polygon, checks)
}

/// Uses [polygon_topology_is_valid] for every part.
/// if `check_multi_polygon_self_overlap=true`, uses [meet_only_at_points_and_no_overlap] for its parts.
fn multi_polygon_topology_is_valid(
    multi: &ScalarMultiPolygon,
    checks: &MultiPolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    let vertices = multi.rings.span(multi.polygons.span(multi.parts.clone()));
    if !xy_is_finite(multi.coords, vertices) {
        return Ok(false);
    }
    if checks.checks_everything() {
        return Ok(multi.to_geos()?.is_valid()?);
    }
    let polygons = multi
        .polygons()
        .filter(|polygon| !is_empty_polygon(polygon));
    if !checks.check_multi_polygon_self_overlap {
        for polygon in polygons {
            if !finite_polygon_topology_is_valid(&polygon, &checks.polygons)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    // Every part is built once, for its own checks as well as to relate to the others.
    let mut parts = Vec::with_capacity(multi.parts.len());
    for polygon in polygons {
        let geometry = polygon.to_geos()?;
        if !whole_polygon_passes(&geometry, &checks.polygons)?
            || !rings_fit_if_checked_separately(&polygon, &checks.polygons)?
        {
            return Ok(false);
        }
        parts.push(geometry);
    }
    meet_only_at_points_and_no_overlap(&parts)
}

fn points_are_malformed_with<const CHECK_EMPTY: bool>(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::from_iter((0..coords.ignoring_validity().len()).map(|i| {
        CHECK_EMPTY && coords.is_valid(i) && is_empty_point(coords.ignoring_validity(), i)
    }));
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn points_are_malformed(array: &PointArray, check_empty: bool) -> BooleanArray {
    with_const_flags!(points_are_malformed_with(array); check_empty)
}

fn multi_points_are_malformed_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const TOLERANCE_COINCIDENCE: bool,
    const Z: bool,
>(
    array: &MultiPointArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let points = array.rows.range(i);
        (CHECK_REPEATED_POINTS
            && has_repeated_points::<TOLERANCE_COINCIDENCE, Z>(
                &array.coords,
                points.clone(),
                tolerance,
            ))
            || (CHECK_EMPTY && only_empty_points(&array.coords, points))
    })
}

pub fn multi_points_are_malformed(
    array: &MultiPointArray,
    check_repeated_points: bool,
    check_empty: bool,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_points_are_malformed_with(array, tolerance.unwrap_or_default());
        check_repeated_points, check_empty, tolerance.is_some(), z
    )
}

fn line_strings_are_malformed_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const TOLERANCE_COINCIDENCE: bool,
    const Z: bool,
>(
    array: &LineStringArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.rows.range(i);
        (CHECK_REPEATED_POINTS
            && has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(
                &array.coords,
                vertices.clone(),
                tolerance,
            ))
            || (CHECK_EMPTY && only_empty_points(&array.coords, vertices))
    })
}

pub fn line_strings_are_malformed(
    array: &LineStringArray,
    check_repeated_points: bool,
    check_empty: bool,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        line_strings_are_malformed_with(array, tolerance.unwrap_or_default());
        check_repeated_points, check_empty, tolerance.is_some(), z
    )
}

/// [line_strings_are_malformed_with] for every part.
fn multi_line_strings_are_malformed_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const TOLERANCE_COINCIDENCE: bool,
    const Z: bool,
>(
    array: &MultiLineStringArray,
    tolerance: Tolerance,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let parts = array.rows.range(i);
        (CHECK_REPEATED_POINTS
            && parts.clone().any(|line| {
                has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(
                    &array.coords,
                    array.lines.range(line),
                    tolerance,
                )
            }))
            || (CHECK_EMPTY && only_empty_points(&array.coords, array.lines.span(parts)))
    })
}

pub fn multi_line_strings_are_malformed(
    array: &MultiLineStringArray,
    check_repeated_points: bool,
    check_empty: bool,
    tolerance: Option<Tolerance>,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_line_strings_are_malformed_with(array, tolerance.unwrap_or_default());
        check_repeated_points, check_empty, tolerance.is_some(), z
    )
}

/// A polygon that fails one of the `topology` checks is malformed, if there are any.
fn polygons_are_malformed_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const TOLERANCE_COINCIDENCE: bool,
    const Z: bool,
>(
    array: &PolygonArray,
    topology: Option<&PolygonTopologyChecks>,
    tolerance: Tolerance,
) -> GeoComputeResult<BooleanArray> {
    try_per_row_bool(&array.rows, |i| {
        let polygon = array.get(i);
        let vertices = array.rings.span(array.rows.range(i));
        Ok((CHECK_REPEATED_POINTS
            && polygon.rings().any(|ring| {
                has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(
                    ring.coords,
                    ring.vertices,
                    tolerance,
                )
            }))
            || (CHECK_EMPTY && only_empty_points(&array.coords, vertices))
            || match topology {
                Some(checks) => !polygon_topology_is_valid(&polygon, checks)?,
                None => false,
            })
    })
}

pub fn polygons_are_malformed(
    array: &PolygonArray,
    check_repeated_points: bool,
    check_empty: bool,
    topology: &PolygonTopologyChecks,
    tolerance: Option<Tolerance>,
) -> GeoComputeResult<BooleanArray> {
    let topology = topology.checks_anything().then_some(topology);
    let z = array.coords.dim().has_z();
    with_const_flags!(
        polygons_are_malformed_with(array, topology, tolerance.unwrap_or_default());
        check_repeated_points, check_empty, tolerance.is_some(), z
    )
}

/// [polygons_are_malformed_with] for every part.
/// A multipolygon that fails one of the `topology` checks is malformed too, if there are any.
fn multi_polygons_are_malformed_with<
    const CHECK_REPEATED_POINTS: bool,
    const CHECK_EMPTY: bool,
    const TOLERANCE_COINCIDENCE: bool,
    const Z: bool,
>(
    array: &MultiPolygonArray,
    topology: Option<&MultiPolygonTopologyChecks>,
    tolerance: Tolerance,
) -> GeoComputeResult<BooleanArray> {
    try_per_row_bool(&array.rows, |i| {
        let multi = array.get(i);
        let vertices = array.rings.span(array.polygons.span(array.rows.range(i)));
        Ok((CHECK_REPEATED_POINTS
            && multi.polygons().any(|polygon| {
                polygon.rings().any(|ring| {
                    has_repeated_vertices::<TOLERANCE_COINCIDENCE, Z>(
                        ring.coords,
                        ring.vertices,
                        tolerance,
                    )
                })
            }))
            || (CHECK_EMPTY && only_empty_points(&array.coords, vertices))
            || match topology {
                Some(checks) => !multi_polygon_topology_is_valid(&multi, checks)?,
                None => false,
            })
    })
}

pub fn multi_polygons_are_malformed(
    array: &MultiPolygonArray,
    check_repeated_points: bool,
    check_empty: bool,
    topology: &MultiPolygonTopologyChecks,
    tolerance: Option<Tolerance>,
) -> GeoComputeResult<BooleanArray> {
    let topology = topology.checks_anything().then_some(topology);
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_polygons_are_malformed_with(array, topology, tolerance.unwrap_or_default());
        check_repeated_points, check_empty, tolerance.is_some(), z
    )
}
