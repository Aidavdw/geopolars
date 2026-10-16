//! Convex hulls, forwarded to GEOS.
//! GEOS has clockwise winding order,
//! so the user gets to choose if we want to reverse it.

use std::ops::Range;

use geos::{CoordType, Geom, Geometry, GeometryTypes};

use super::linestrings::{line_string_to_geos, multi_line_string_to_geos};
use super::polygons::{multi_polygon_to_geos, polygon_to_geos};
use super::{axes, Axes};
use crate::{
    GeoArrowListShared, GeoArrowListView, GeoArrowPolygonOutput, GeoArrowValidCoordsView,
    GeoArrowValidListView, GeoDimension, LineStringArrayView, MultiLineStringArrayView,
    MultiPointArrayView, MultiPolygonArrayView, OwnedAxes, PolygonArrayView,
};

// Because we don't consider CRS,
// A geometry crossing the antimeridian is hulled the long way round.
// We could add a flag that checks if the input crosses it (>180°),
// and then reproject them back

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArrayView<R, C> {
    /// The convex hull of every row, forwarded to GEOS.
    /// GEOS keeps z, but drops m.
    /// An empty linestring's hull is empty.
    /// GEOS returns a point or a linestring for a straight linestring,
    /// so we return invalid.
    /// With `stop_on_error`, a GEOS error is returned; otherwise that row is missing.
    /// With `ccw_outer`, the exterior runs counter-clockwise; otherwise as GEOS gives it (clockwise).
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
        ccw_outer: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            ccw_outer,
            |vertices| vertices.is_empty(),
            |axes, vertices| line_string_to_geos(axes, vertices).map(Some),
        )
    }
}

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> MultiPointArrayView<R, C> {
    /// The convex hull of every row, forwarded to GEOS.
    /// The points go to GEOS as one linestring through them, which has the same hull,
    /// rather than as a GEOS multipoint built point by point.
    /// GEOS keeps z, but drops m.
    /// An empty multipoint's hull is empty.
    /// A single point encloses nothing, so that row is invalid without going to GEOS.
    /// GEOS returns a linestring when all the points lie on one line, so we return invalid.
    /// With `stop_on_error`, a GEOS error is returned; otherwise that row is missing.
    /// With `ccw_outer`, the exterior runs counter-clockwise; otherwise as GEOS gives it (clockwise).
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
        ccw_outer: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            ccw_outer,
            |vertices| vertices.is_empty(),
            // The outer hull of a collection of points and that of a linestring are
            // identical.
            // It is cheaper to allocate a linestring though, so we use that to save
            // some operations.
            // GEOS refuses a linestring of one vertex.
            |axes, vertices| match vertices.len() {
                1 => Ok(None),
                _ => line_string_to_geos(axes, vertices).map(Some),
            },
        )
    }
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiLineStringArrayView<R, L, C>
{
    /// The convex hull of every row: the GEOS multilinestring of all its parts, hulled as a whole.
    /// GEOS keeps z, but drops m.
    /// An empty multilinestring's hull is empty.
    /// GEOS returns a point or a linestring when all the vertices lie on one line,
    /// so we return invalid.
    /// With `stop_on_error`, a GEOS error is returned; otherwise that row is missing.
    /// With `ccw_outer`, the exterior runs counter-clockwise; otherwise as GEOS gives it (clockwise).
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
        ccw_outer: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            ccw_outer,
            |parts| spans_nothing(&self.lines, parts),
            |axes, parts| multi_line_string_to_geos(axes, &self.lines, parts).map(Some),
        )
    }
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArrayView<R, L, C>
{
    /// The convex hull of every row, forwarded to GEOS.
    /// GEOS keeps z, but drops m.
    /// An empty polygon's hull is empty.
    /// GEOS returns a point or a linestring for a polygon with zero area
    /// (all the points intersecting), so we return invalid.
    /// With `stop_on_error`, a GEOS error is returned; otherwise that row is missing.
    /// With `ccw_outer`, the exterior runs counter-clockwise; otherwise as GEOS gives it (clockwise).
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
        ccw_outer: bool,
    ) -> geos::GResult<O> {
        // A ring has at least 4 vertices, so a polygon is only empty without rings.
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            ccw_outer,
            |parts| parts.is_empty(),
            |axes, parts| polygon_to_geos(axes, &self.rings, parts).map(Some),
        )
    }
}

impl<
        R: GeoArrowListView,
        P: GeoArrowValidListView,
        L: GeoArrowValidListView,
        C: GeoArrowValidCoordsView,
    > MultiPolygonArrayView<R, P, L, C>
{
    /// The convex hull of every row: the GEOS multipolygon of all its parts, hulled as a whole.
    /// GEOS keeps z, but drops m.
    /// An empty multipolygon's hull is empty.
    /// GEOS returns a point or a linestring for a multipolygon with zero area
    /// (all the points intersecting), so we return invalid.
    /// With `stop_on_error=true`, a GEOS error is returned; otherwise that row is missing.
    ///
    /// GEOS gives the exterior as running clockwise.
    /// You can reverse this by giving `ccw_outer=true` for a minimal performance penalty.
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
        ccw_outer: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            ccw_outer,
            |parts| spans_nothing(&self.polygons, parts),
            |axes, parts| multi_polygon_to_geos(axes, &self.polygons, &self.rings, parts).map(Some),
        )
    }
}

/// Whether the `parts` of `list` hold no items at all: none, or only empty ones.
#[inline(always)]
fn spans_nothing(list: &impl GeoArrowListShared, parts: Range<usize>) -> bool {
    parts.is_empty() || list.range(parts.start).start == list.range(parts.end - 1).end
}

/// The convex hull of every one of `rows`, each turned into GEOS by `to_geos` from its parts.
/// Where `to_geos` gives `None`, the row has no hull and is invalid.
/// A row that `is_empty` says is empty gets an empty hull without going to GEOS.
fn hulls<R: GeoArrowListView, O: GeoArrowPolygonOutput<Validity = R::Validity>>(
    rows: &R,
    coords: &impl GeoArrowValidCoordsView,
    stop_on_error: bool,
    ccw_outer: bool,
    is_empty: impl Fn(Range<usize>) -> bool,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Option<Geometry>>,
) -> geos::GResult<O> {
    let hulls = with_const_flags!(
        hulls_with(rows, coords, stop_on_error, is_empty, to_geos);
        coords.dim().has_z(), rows.is_all_valid(), ccw_outer
    )?;
    Ok(O::from_buffers(
        hulls.dim,
        hulls.axes,
        hulls.rings,
        hulls.rows,
        rows.validity(),
        &hulls.failed,
    ))
}

/// What [`hulls_with`] builds, for [`GeoArrowPolygonOutput::from_buffers`].
/// The output's type is left to [`hulls`], so `hulls_with` needs no generics but its flags.
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
///
/// GEOS gives the exterior as running clockwise.
/// You can reverse this by giving `ccw_outer=true` for a minimal performance penalty.
fn hulls_with<const Z: bool, const ALL_VALID: bool, const CCW_OUTER: bool>(
    rows: &impl GeoArrowListView,
    coords: &impl GeoArrowValidCoordsView,
    stop_on_error: bool,
    is_empty: impl Fn(Range<usize>) -> bool,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Option<Geometry>>,
) -> geos::GResult<HullBuffers> {
    let dim = if Z {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let coord_type = CoordType::from(dim);
    let width = dim.field_names().len();
    let axes = axes(coords);
    let mut x = Vec::with_capacity(coords.len());
    let mut y = Vec::with_capacity(coords.len());
    let mut z = if Z {
        Vec::with_capacity(coords.len())
    } else {
        // will not be allocated because we don't push anything in.
        Vec::new()
    };

    // The hull's exterior as one interleaved buffer, rather than a `Vec` per axis.
    // `None` if it is not a polygon. Nothing is pushed until all of it succeeded.
    let hull_ring = |parts| -> geos::GResult<Option<Vec<f64>>> {
        let Some(geometry) = to_geos(&axes, parts)? else {
            return Ok(None);
        };
        let hull = geometry.convex_hull()?;
        // Empty input never gets here, so an empty hull is a failure too.
        if hull.is_empty()? || hull.geometry_type()? != GeometryTypes::Polygon {
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
        // An empty one has an empty hull: no rings, and no trip to GEOS.
        if (ALL_VALID || rows.is_valid(i)) && !is_empty(rows.range(i)) {
            match hull_ring(rows.range(i)) {
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
                // Zero area (a point or a linestring), or a failure we were asked to pass over.
                Ok(None) => failed.push(i),
                Err(e) if stop_on_error => return Err(e),
                Err(_) => failed.push(i),
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
