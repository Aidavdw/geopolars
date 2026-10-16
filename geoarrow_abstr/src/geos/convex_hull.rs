//! Convex hulls, forwarded to GEOS.

use std::ops::Range;

use geos::{CoordType, Geom, Geometry, GeometryTypes};

use super::linestrings::{line_string_to_geos, multi_line_string_to_geos};
use super::polygons::{multi_polygon_to_geos, polygon_to_geos};
use super::{axes, Axes};
use crate::{
    GeoArrowListShared, GeoArrowListView, GeoArrowPolygonOutput, GeoArrowValidCoordsView,
    GeoArrowValidListView, GeoDimension, LineStringArrayView, MultiLineStringArrayView,
    MultiPolygonArrayView, PolygonArrayView,
};

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArrayView<R, C> {
    /// The convex hull of every row, forwarded to GEOS.
    /// GEOS keeps z, but drops m.
    /// An empty linestring's hull is empty.
    /// GEOS returns a point or a linestring for a straight linestring,
    /// so we return invalid.
    /// With `stop_on_error`, a GEOS error is returned; otherwise that row is missing.
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            |vertices| vertices.is_empty(),
            line_string_to_geos,
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
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            |parts| spans_nothing(&self.lines, parts),
            |axes, parts| multi_line_string_to_geos(axes, &self.lines, parts),
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
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
    ) -> geos::GResult<O> {
        // A ring has at least 4 vertices, so a polygon is only empty without rings.
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            |parts| parts.is_empty(),
            |axes, parts| polygon_to_geos(axes, &self.rings, parts),
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
    /// With `stop_on_error`, a GEOS error is returned; otherwise that row is missing.
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        stop_on_error: bool,
    ) -> geos::GResult<O> {
        hulls(
            &self.rows,
            &self.coords,
            stop_on_error,
            |parts| spans_nothing(&self.polygons, parts),
            |axes, parts| multi_polygon_to_geos(axes, &self.polygons, &self.rings, parts),
        )
    }
}

/// Whether the `parts` of `list` hold no items at all: none, or only empty ones.
#[inline(always)]
fn spans_nothing(list: &impl GeoArrowListShared, parts: Range<usize>) -> bool {
    parts.is_empty() || list.range(parts.start).start == list.range(parts.end - 1).end
}

/// The convex hull of every one of `rows`, each turned into GEOS by `to_geos` from its parts.
/// A row that `is_empty` says is empty gets an empty hull without going to GEOS.
fn hulls<R: GeoArrowListView, O: GeoArrowPolygonOutput<Validity = R::Validity>>(
    rows: &R,
    coords: &impl GeoArrowValidCoordsView,
    stop_on_error: bool,
    is_empty: impl Fn(Range<usize>) -> bool,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Geometry>,
) -> geos::GResult<O> {
    match (coords.dim().has_z(), rows.is_all_valid()) {
        (true, true) => {
            hulls_with::<true, true, R, O>(rows, coords, stop_on_error, is_empty, to_geos)
        }
        (true, false) => {
            hulls_with::<true, false, R, O>(rows, coords, stop_on_error, is_empty, to_geos)
        }
        (false, true) => {
            hulls_with::<false, true, R, O>(rows, coords, stop_on_error, is_empty, to_geos)
        }
        (false, false) => {
            hulls_with::<false, false, R, O>(rows, coords, stop_on_error, is_empty, to_geos)
        }
    }
}

/// [`hulls`], with `Z` saying whether the coordinates (and so the hulls) have z.
/// if `ALL_VALID=true`, we don't first check for validity, and just send it to GEOS.
/// (If it does contain invalid ones, it's fine because we still check for invalid returned values,
/// but we're just wasting time because it could have been caught before)
fn hulls_with<
    const Z: bool,
    const ALL_VALID: bool,
    R: GeoArrowListView,
    O: GeoArrowPolygonOutput<Validity = R::Validity>,
>(
    rows: &R,
    coords: &impl GeoArrowValidCoordsView,
    stop_on_error: bool,
    is_empty: impl Fn(Range<usize>) -> bool,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Geometry>,
) -> geos::GResult<O> {
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
        let hull = to_geos(&axes, parts)?.convex_hull()?;
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
                    for coord in buffer.chunks_exact(width) {
                        x.push(coord[0]);
                        y.push(coord[1]);
                        if Z {
                            z.push(coord[2]);
                        }
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
    Ok(O::from_buffers(
        dim,
        (x, y, Z.then_some(z), None),
        rings,
        offsets,
        rows.validity(),
        &failed,
    ))
}
