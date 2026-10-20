//! Convex hulls, forwarded to GEOS.
//!
//! A hull only depends on the points, not on the kind of geometry.
//! We use this to our advantage, and make every row go to GEOS the same way:
//! as one linestring (through its coordinates).
//! For a polygon or a multipolygon, these are only the coordinates of the exteriors.
//! A valid polygon will not change in size if you add or remove interior sections after all.
//!
//! Some degenerate cases can be caught in advance:
//! - A row of one coordinate encloses nothing, so it never goes to GEOS.
//! - An set of coordinates cannot enclose anything
//! - A two-point line can only enclose a line, so not a polygon -> also reject early
//!
//! GEOS has clockwise winding order,
//! so the user gets to choose if we want to reverse it.
//!
//! This trick almost always gives the same hull as GEOS on the geometry itself,
//! but there are some edge cases:
//! - GEOS ignores the holes of a polygon, but not those of a multipolygon.
//!   Such a multipolygon is invalid, so we take it to be a GEOS bug, and ignore holes for both.
//! - GEOS switches algorithm above 50 coordinates (`TUNING_REDUCE_SIZE`),
//!   counting duplicates and the closing vertex.
//!   Holes add to that count for a multipolygon in GEOS, but not here.
//!   So near that boundary, one of the two can take the other path.
//!   Then nearly collinear input can give a different (sometimes non-convex) sliver hull,
//!   and NaN input an error instead of a hull, or the other way around.
//! - No rings are built.
//!   GEOS can't build a ring starting with NaN, because its closing NaN ≠ NaN.
//!   Here, that ring is hulled like any other NaN input.
//!
//! NaN coordinates are passed to GEOS as they are.
//! What happends to them depends on where they are placed.
//! - only empty points (x and y NaN) -> a `POINT (NaN NaN)`, so the row is missing
//! - an empty point first, then real ones -> also a `POINT (NaN NaN)`,
//!   so the row is silently missing, even with `ignore_per_row_errors=false`;
//! - an empty point after a real one -> it is left out, giving the hull of the others
//! - a vertex with just one NaN axis -> an error ("orientationIndex encountered NaN/Inf numbers").
//!   So the error is returned, or with `ignore_per_row_errors=true` the row is missing.
//!
//! Leaving NaN vertices out before going to GEOS would make this consistent.

use std::ops::Range;

use geopolars_core::output::{polygon_array, OwnedAxes};
use geopolars_core::{
    GeoDimension, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray, ValidCoords,
};
use geos::{CoordSeq, Geom, Geometry, GeometryTypes};
use polars_arrow::array::ListArray;

use super::{axes, coord_type};

// Because we don't consider CRS,
// A geometry crossing the antimeridian is hulled the long way round.
// We could add a flag that checks if the input crosses it (>180°),
// and then reproject them back

/// The convex hull of every row, forwarded to GEOS.
/// GEOS keeps z, but drops m.
/// An empty linestring's hull is empty.
/// A straight linestring (or one of a single vertex) has a point or a line for a hull,
/// so that row is missing.
/// If `ignore_per_row_errors=true`, a row GEOS fails on is null,
/// otherwise its error is returned.
/// If `ccw_outer=true`, the exterior is made to run counter-clockwise
/// at a small performance cost. If it is not set, orientation is clockwise,
/// as GEOS gives it.
pub fn line_string_convex_hulls(
    array: &LineStringArray,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
) -> geos::GResult<ListArray<i64>> {
    hulls(
        &array.rows,
        &array.coords,
        ignore_per_row_errors,
        ccw_outer,
        |vertices, ranges| ranges.push(vertices),
    )
}

/// Like [`line_string_convex_hulls`], with one hull around all the points.
/// One or two points, or points all on one line, give a missing row.
pub fn multi_point_convex_hulls(
    array: &MultiPointArray,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
) -> geos::GResult<ListArray<i64>> {
    hulls(
        &array.rows,
        &array.coords,
        ignore_per_row_errors,
        ccw_outer,
        |vertices, ranges| ranges.push(vertices),
    )
}

/// Like [`line_string_convex_hulls`], with one hull of all the parts as a whole.
pub fn multi_line_string_convex_hulls(
    array: &MultiLineStringArray,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
) -> geos::GResult<ListArray<i64>> {
    hulls(
        &array.rows,
        &array.coords,
        ignore_per_row_errors,
        ccw_outer,
        |parts, ranges| ranges.extend(parts.map(|j| array.lines.range(j))),
    )
}

/// Like [`line_string_convex_hulls`], hulling only the exterior.
/// A polygon with zero area gives a missing row.
pub fn polygon_convex_hulls(
    array: &PolygonArray,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
) -> geos::GResult<ListArray<i64>> {
    hulls(
        &array.rows,
        &array.coords,
        ignore_per_row_errors,
        ccw_outer,
        |parts, ranges| {
            if !parts.is_empty() {
                ranges.push(array.rings.range(parts.start));
            }
        },
    )
}

/// Like [`line_string_convex_hulls`],
/// with one hull of the exteriors of all the parts as a whole.
/// A multipolygon with zero area gives a missing row.
pub fn multi_polygon_convex_hulls(
    array: &MultiPolygonArray,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
) -> geos::GResult<ListArray<i64>> {
    hulls(
        &array.rows,
        &array.coords,
        ignore_per_row_errors,
        ccw_outer,
        |parts, ranges| {
            ranges.extend(
                parts
                    .map(|p| array.polygons.range(p))
                    .filter(|rings| !rings.is_empty())
                    .map(|rings| array.rings.range(rings.start)),
            )
        },
    )
}

/// `coord_ranges` adds the ranges of `coords` to hull for the parts of a row.
fn hulls(
    rows: &List,
    coords: &ValidCoords,
    ignore_per_row_errors: bool,
    ccw_outer: bool,
    coord_ranges: impl Fn(Range<usize>, &mut Vec<Range<usize>>),
) -> geos::GResult<ListArray<i64>> {
    let hulls = with_const_flags!(
        hulls_with(rows, coords, ignore_per_row_errors, coord_ranges);
        coords.dim().has_z(), rows.is_all_valid(), ccw_outer
    )?;
    Ok(polygon_array(
        hulls.dim,
        hulls.axes,
        hulls.rings,
        hulls.rows,
        rows.validity(),
        &hulls.failed,
    ))
}

struct HullBuffers {
    dim: GeoDimension,
    axes: OwnedAxes,
    rings: Vec<i64>,
    rows: Vec<i64>,
    failed: Vec<usize>,
}

/// [`hulls`], with `Z` saying whether the coordinates (and so the hulls) have z.
/// if `ALL_VALID=true`, we don't first check for validity, and just send it to GEOS.
/// (If it does contain invalid ones, it's fine because we still check for invalid returned values,
/// but we're just wasting time because it could have been caught before)
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
///
/// GEOS gives the exterior as running clockwise.
/// You can reverse this by giving `CCW_OUTER=true` for a minimal performance penalty.
fn hulls_with<const Z: bool, const ALL_VALID: bool, const CCW_OUTER: bool>(
    rows: &List,
    coords: &ValidCoords,
    ignore_per_row_errors: bool,
    coord_ranges: impl Fn(Range<usize>, &mut Vec<Range<usize>>),
) -> geos::GResult<HullBuffers> {
    let dim = if Z {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let coord_type = coord_type(dim);
    let width = dim.field_names().len();
    let (in_x, in_y, in_z, _) = axes(coords);
    let in_z = in_z.unwrap_or_default();
    let mut x = Vec::with_capacity(coords.len());
    let mut y = Vec::with_capacity(coords.len());
    let mut z = if Z {
        Vec::with_capacity(coords.len())
    } else {
        // will not be allocated because we don't push anything in.
        Vec::new()
    };

    // The coordinates of one row, gathered from its ranges, and reused across rows.
    // GEOS drops m, so it isn't copied.
    let mut ranges = Vec::new();
    let mut row_x = Vec::new();
    let mut row_y = Vec::new();
    let mut row_z = Vec::new();

    // The hull's exterior as one interleaved buffer, rather than a `Vec` per axis.
    // `None` if it is not a polygon (degenerate case)
    // Nothing is pushed until all of it succeeded.
    let hull_ring = |row_x: &[f64], row_y: &[f64], row_z: &[f64]| {
        let seq = CoordSeq::new_from_arrays(row_x, row_y, Z.then_some(row_z), None)?;
        let hull = Geometry::create_line_string(seq)?.convex_hull()?;
        debug_assert!(!hull.is_empty()?, "Check before offset should catch this");
        if hull.geometry_type()? != GeometryTypes::Polygon {
            return Ok(None);
        }
        let ring = hull.get_exterior_ring()?;
        Ok(Some(ring.get_coord_seq()?.as_buffer(Some(coord_type))?))
    };

    let mut rings = vec![0];
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    let mut failed = Vec::new();
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not converted.
        if ALL_VALID || rows.is_valid(i) {
            ranges.clear();
            coord_ranges(rows.range(i), &mut ranges);
            row_x.clear();
            row_y.clear();
            row_z.clear();
            for range in &ranges {
                row_x.extend_from_slice(&in_x[range.clone()]);
                row_y.extend_from_slice(&in_y[range.clone()]);
                if Z {
                    row_z.extend_from_slice(&in_z[range.clone()]);
                }
            }
            match row_x.len() {
                // An empty row has an empty hull: no rings, and no trip to GEOS.
                0 => {}
                // A single coordinate encloses nothing, and GEOS can't build a linestring of it.
                1 => failed.push(i),
                _ => match hull_ring(&row_x, &row_y, &row_z) {
                    Ok(Some(buffer)) => {
                        let mut push = |coord: &[f64]| {
                            x.push(coord[0]);
                            y.push(coord[1]);
                            if Z {
                                z.push(coord[2]);
                            }
                        };
                        if CCW_OUTER {
                            buffer.chunks_exact(width).rev().for_each(&mut push);
                        } else {
                            buffer.chunks_exact(width).for_each(&mut push);
                        }
                        rings.push(x.len() as i64);
                    }
                    // Zero area: a point or a linestring.
                    Ok(None) => failed.push(i),
                    Err(e) if !ignore_per_row_errors => return Err(e),
                    Err(_) => failed.push(i),
                },
            }
        }
        offsets.push(rings.len() as i64 - 1);
    }
    Ok(HullBuffers {
        dim,
        axes: (x, y, Z.then_some(z), None),
        rings,
        rows: offsets,
        failed,
    })
}
