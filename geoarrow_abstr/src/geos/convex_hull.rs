//! Convex hulls, forwarded to GEOS.

use std::ops::Range;

use geos::{Geom, Geometry, GeometryTypes};

use super::polygons::{multi_polygon_to_geos, polygon_to_geos};
use super::{axes, Axes};
use crate::{
    GeoArrowListView, GeoArrowPolygonOutput, GeoArrowValidCoordsView, GeoArrowValidListView,
    GeoDimension, MultiPolygonArrayView, PolygonArrayView,
};

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArrayView<R, L, C>
{
    /// The convex hull of every row, forwarded to GEOS.
    /// GEOS keeps z, but drops m.
    /// An empty polygon's hull is empty.
    /// GEOS returns a point or a linestring for a polygon with zero area
    /// (all the points intersecting), so we return invalid.
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
    ) -> geos::GResult<O> {
        hulls(&self.rows, &self.coords, |axes, parts| {
            polygon_to_geos(axes, &self.rings, parts)
        })
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
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
    ) -> geos::GResult<O> {
        hulls(&self.rows, &self.coords, |axes, parts| {
            multi_polygon_to_geos(axes, &self.polygons, &self.rings, parts)
        })
    }
}

fn hulls<R: GeoArrowListView, O: GeoArrowPolygonOutput<Validity = R::Validity>>(
    rows: &R,
    coords: &impl GeoArrowValidCoordsView,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Geometry>,
) -> geos::GResult<O> {
    if coords.dim().has_z() {
        hulls_with::<true, R, O>(rows, coords, to_geos)
    } else {
        hulls_with::<false, R, O>(rows, coords, to_geos)
    }
}

/// [`hulls`], with `Z` saying whether the coordinates (and so the hulls) have z.
fn hulls_with<
    const Z: bool,
    R: GeoArrowListView,
    O: GeoArrowPolygonOutput<Validity = R::Validity>,
>(
    rows: &R,
    coords: &impl GeoArrowValidCoordsView,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Geometry>,
) -> geos::GResult<O> {
    let dim = if Z {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let axes = axes(coords);
    let mut x = Vec::with_capacity(coords.len());
    let mut y = Vec::with_capacity(coords.len());
    let mut z = if Z {
        Vec::with_capacity(coords.len())
    } else {
        // will not be allocated because we don't push anything in.
        Vec::new()
    };

    let mut rings = vec![0];
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    let mut failed = Vec::new();
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not converted.
        if rows.is_valid(i) {
            let hull = to_geos(&axes, rows.range(i))?.convex_hull()?;
            match hull.geometry_type()? {
                // An empty hull gets no rings.
                _ if hull.is_empty()? => {}
                GeometryTypes::Polygon => {
                    let (hull_x, hull_y, hull_z, _) =
                        hull.get_exterior_ring()?.get_coord_seq()?.as_arrays()?;
                    x.extend(hull_x);
                    y.extend(hull_y);
                    if Z {
                        z.extend(hull_z.expect("GEOS keeps the z it is given"));
                    }
                    rings.push(x.len() as i64);
                }
                // Zero area: a point or a linestring.
                _ => failed.push(i),
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
