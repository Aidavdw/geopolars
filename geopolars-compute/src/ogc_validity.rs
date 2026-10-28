//! Whether each geometry is valid by OGC Simple Features, as GEOS's `is_valid` has it.
//! See `is_valid_ogc` in Python for the rules.
//!
//! Every geometry is held to the checks of [crate::geoarrow_validity] first,
//! with every coordinate finite: only a point or multipoint may hold the empty point.
//! Then its shape is checked natively, and its topology through GEOS.
//! Each gives a `bool` per row, `false` for a null row.
//! The polygon kernels error when GEOS does.
//! With every topology check on, GEOS's `is_valid` runs them all at once.
//! Otherwise, each check runs on its own.
//!
//! Null values below the rows are not looked at here, as in [crate::geoarrow_validity].
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! The polygon kernels are generic over `Z`.
//! If `Z=true`, a ring has to close in z as well.

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarMultiPolygon, ScalarPolygon,
};
use geos::{Geom, Geometry};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;

use crate::geoarrow_validity::{multi_points_are_valid_geoarrow, points_are_valid_geoarrow};
use crate::geos::{meet_only_at_points_and_no_overlap, polygon_rings_fit, ToGeos};
use crate::native::{
    all_coordinates_valid, coordinates_are_valid, line_string_shape_is_valid,
    polygon_rings_are_closed, polygon_shape_is_valid, Bounds, CoordinateChecks,
};
use crate::per_row::{per_row_bool, try_per_row_bool};
use crate::GeoComputeResult;

/// The topology checks to run on a polygon. See `is_valid_ogc` in Python.
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

/// The topology checks to run on a multipolygon. See `is_valid_ogc` in Python.
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

fn finite_coordinate_checks(bounds: Option<Bounds>, allow_empty_points: bool) -> CoordinateChecks {
    CoordinateChecks {
        bounds,
        check_finite_coordinates: true,
        allow_empty_points,
    }
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

/// Whether `polygon` passes the topology `checks`.
/// It has to have a valid shape, and be finite in x and y.
fn polygon_topology_is_valid(
    polygon: &ScalarPolygon,
    checks: &PolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    if is_empty_polygon(polygon) {
        return Ok(true);
    }
    if needs_whole_polygon_in_geos(checks) && !whole_polygon_passes(&polygon.to_geos()?, checks)? {
        return Ok(false);
    }
    rings_fit_if_checked_separately(polygon, checks)
}

/// Uses [polygon_topology_is_valid] for every part.
/// if `check_multi_polygon_self_overlap=true`, uses [meet_only_at_points_and_no_overlap] for its parts.
fn multi_polygon_topology_is_valid(
    multi: &ScalarMultiPolygon,
    checks: &MultiPolygonTopologyChecks,
) -> GeoComputeResult<bool> {
    if checks.checks_everything() {
        return Ok(multi.to_geos()?.is_valid()?);
    }
    let polygons = multi
        .polygons()
        .filter(|polygon| !is_empty_polygon(polygon));
    if !checks.check_multi_polygon_self_overlap {
        for polygon in polygons {
            if !polygon_topology_is_valid(&polygon, &checks.polygons)? {
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

pub fn points_are_valid_ogc(array: &PointArray, bounds: Option<Bounds>) -> BooleanArray {
    points_are_valid_geoarrow(array, &finite_coordinate_checks(bounds, true))
}

pub fn multi_points_are_valid_ogc(array: &MultiPointArray, bounds: Option<Bounds>) -> BooleanArray {
    multi_points_are_valid_geoarrow(array, &finite_coordinate_checks(bounds, true))
}

pub fn line_strings_are_valid_ogc(array: &LineStringArray, bounds: Option<Bounds>) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, &finite_coordinate_checks(bounds, false));
    per_row_bool(&array.rows, |i| {
        let vertices = array.rows.range(i);
        all_coordinates_valid(valid.as_ref(), vertices.clone())
            && line_string_shape_is_valid(&array.coords, vertices)
    })
}

pub fn multi_line_strings_are_valid_ogc(
    array: &MultiLineStringArray,
    bounds: Option<Bounds>,
) -> BooleanArray {
    let valid = coordinates_are_valid(&array.coords, &finite_coordinate_checks(bounds, false));
    per_row_bool(&array.rows, |i| {
        let parts = array.rows.range(i);
        all_coordinates_valid(valid.as_ref(), array.lines.span(parts.clone()))
            && parts
                .clone()
                .all(|line| line_string_shape_is_valid(&array.coords, array.lines.range(line)))
    })
}

/// Whether `polygon` passes the coordinate checks, is closed and has a valid shape.
#[inline(always)]
fn polygon_is_valid_without_topology<const Z: bool>(
    polygon: &ScalarPolygon,
    valid: Option<&Bitmap>,
) -> bool {
    all_coordinates_valid(valid, polygon.rings.span(polygon.parts.clone()))
        && polygon_rings_are_closed::<Z>(polygon)
        && polygon_shape_is_valid(polygon)
}

fn polygons_are_valid_with<const Z: bool>(
    array: &PolygonArray,
    valid: Option<&Bitmap>,
    topology: Option<&PolygonTopologyChecks>,
) -> GeoComputeResult<BooleanArray> {
    try_per_row_bool(&array.rows, |i| {
        let polygon = array.get(i);
        if !polygon_is_valid_without_topology::<Z>(&polygon, valid) {
            return Ok(false);
        }
        topology.map_or(Ok(true), |checks| {
            polygon_topology_is_valid(&polygon, checks)
        })
    })
}

/// A polygon that fails one of the `topology` checks is invalid, if there are any.
pub fn polygons_are_valid_ogc(
    array: &PolygonArray,
    bounds: Option<Bounds>,
    topology: &PolygonTopologyChecks,
) -> GeoComputeResult<BooleanArray> {
    let valid = coordinates_are_valid(&array.coords, &finite_coordinate_checks(bounds, false));
    let topology = topology.checks_anything().then_some(topology);
    let z = array.coords.dim().has_z();
    with_const_flags!(polygons_are_valid_with(array, valid.as_ref(), topology); z)
}

fn multi_polygons_are_valid_with<const Z: bool>(
    array: &MultiPolygonArray,
    valid: Option<&Bitmap>,
    topology: Option<&MultiPolygonTopologyChecks>,
) -> GeoComputeResult<BooleanArray> {
    try_per_row_bool(&array.rows, |i| {
        let multi = array.get(i);
        if !multi
            .polygons()
            .all(|polygon| polygon_is_valid_without_topology::<Z>(&polygon, valid))
        {
            return Ok(false);
        }
        topology.map_or(Ok(true), |checks| {
            multi_polygon_topology_is_valid(&multi, checks)
        })
    })
}

/// [polygons_are_valid_ogc] for every part.
/// A multipolygon that fails one of the `topology` checks is invalid too, if there are any.
pub fn multi_polygons_are_valid_ogc(
    array: &MultiPolygonArray,
    bounds: Option<Bounds>,
    topology: &MultiPolygonTopologyChecks,
) -> GeoComputeResult<BooleanArray> {
    let valid = coordinates_are_valid(&array.coords, &finite_coordinate_checks(bounds, false));
    let topology = topology.checks_anything().then_some(topology);
    let z = array.coords.dim().has_z();
    with_const_flags!(multi_polygons_are_valid_with(array, valid.as_ref(), topology); z)
}
