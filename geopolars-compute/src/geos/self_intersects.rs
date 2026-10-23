//! Wrappers around GEOS's `is_simple`.
//! This tests only for self-intersection (it is not a combined operation like `is_valid`).
//! Checks whether a linestring crosses or touches itself in x and y.
//! A polygon also intersects itself when two of its holes overlap.

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray, ScalarLineString,
    ScalarPolygon,
};
use geos::{Geom, Geometry, PreparedGeometry};
use polars_arrow::array::{BooleanArray, ListArray};

use super::linestrings::line_string_closed_within_is_close_to_geos;
use super::polygons::{polygon_enclosed_by_ring_to_geos, polygon_to_geos};
use super::rings_overlap::{enclosed_areas_overlap, ring_is_finite};
use super::{axes, xy_is_finite_or_error};
use crate::per_row::{per_part_optional_bool, per_row_optional_bool};
use crate::{row_or_missing, GeoComputeResult};

/// `None` if one of its vertices is not finite in x or y.
fn line_self_intersects(line: &ScalarLineString) -> GeoComputeResult<Option<bool>> {
    let axes = axes(line.coords);
    let (x, y, _, _) = axes;
    let vertices = line.vertices.clone();
    if xy_is_finite_or_error(&x[vertices.clone()], &y[vertices.clone()]).is_err() {
        return Ok(None);
    }
    // A single segment can't cross itself, and GEOS can't build a line of one vertex.
    if vertices.len() < 3 {
        return Ok(Some(false));
    }
    let line = line_string_closed_within_is_close_to_geos(&axes, vertices)?;
    Ok(Some(!line.is_simple()?))
}

fn any_two_holes_overlap(holes: &[Geometry]) -> GeoComputeResult<bool> {
    for (i, hole) in holes.iter().enumerate() {
        let hole = PreparedGeometry::new(hole)?;
        for other in &holes[i + 1..] {
            if enclosed_areas_overlap::<false>(&hole, other)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether `polygon` intersects itself:
/// one of its rings crosses itself, or two of its holes (every ring after the first) overlap.
/// `None` if one of its vertices is not finite in x or y.
/// Errors if one of its rings is not a ring.
fn polygon_self_intersects(polygon: &ScalarPolygon) -> GeoComputeResult<Option<bool>> {
    // Every ring is checked before GEOS looks at any,
    // so a ring crossing itself does not hide a later one that is not finite, or not a ring.
    for ring in polygon.rings() {
        if !ring_is_finite(&ring)? {
            return Ok(None);
        }
    }
    let axes = axes(polygon.coords);
    // GEOS checks every ring on its own here, not how they sit against each other.
    if !polygon_to_geos(&axes, polygon.rings, polygon.parts.clone())?.is_simple()? {
        return Ok(Some(true));
    }
    // Only with two holes is there anything to relate.
    if polygon.parts.len() < 3 {
        return Ok(Some(false));
    }
    let holes = polygon
        .parts
        .clone()
        .skip(1)
        .map(|j| polygon_enclosed_by_ring_to_geos(&axes, polygon.rings.range(j)))
        .collect::<GeoComputeResult<Vec<_>>>()?;
    any_two_holes_overlap(&holes).map(Some)
}

/// Whether each linestring intersects itself, missing where it is not finite in x or y.
pub fn line_strings_self_intersect(array: &LineStringArray) -> GeoComputeResult<BooleanArray> {
    per_row_optional_bool(&array.rows, |i| line_self_intersects(&array.get(i)))
}

/// [line_strings_self_intersect], as a list per row of a chunk, one per part.
pub fn multi_line_strings_self_intersect(
    array: &MultiLineStringArray,
) -> GeoComputeResult<ListArray<i64>> {
    per_part_optional_bool(&array.rows, &array.lines, |part| {
        line_self_intersects(&ScalarLineString {
            coords: &array.coords,
            vertices: array.lines.range(part),
        })
    })
}

/// Whether each polygon [intersects itself](polygon_self_intersects).
/// If `ignore_per_row_errors=true`, a polygon with a ring that is not one is missing;
/// otherwise that is the error.
pub fn polygons_self_intersect(
    array: &PolygonArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<BooleanArray> {
    per_row_optional_bool(&array.rows, |i| {
        row_or_missing(
            polygon_self_intersects(&array.get(i)),
            ignore_per_row_errors,
        )
    })
}

/// [polygons_self_intersect], as a list per row of a chunk, one per part.
/// If `ignore_per_row_errors=true`, only the part with a ring that is not one is missing.
pub fn multi_polygons_self_intersect(
    array: &MultiPolygonArray,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<ListArray<i64>> {
    per_part_optional_bool(&array.rows, &array.polygons, |part| {
        let polygon = ScalarPolygon {
            rings: &array.rings,
            coords: &array.coords,
            parts: array.polygons.range(part),
        };
        row_or_missing(polygon_self_intersects(&polygon), ignore_per_row_errors)
    })
}
