//! Convex hulls, forwarded to GEOS.

use geos::{Geom, GeometryTypes};

use super::axes;
use super::polygons::polygon_to_geos;
use crate::{
    GeoArrowListView, GeoArrowPolygonOutput, GeoArrowValidCoordsView, GeoArrowValidListView,
    PolygonArrayView,
};

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArrayView<R, L, C>
{
    /// The convex hull of every row, forwarded to GEOS.
    /// GEOS keeps z, but drops m, so a hull's m is NaN.
    /// An empty polygon's hull is empty.
    /// GEOS returns a point or a linestring for a polygon with zero area
    /// (all the points intersecting), so we return invalid.
    pub fn convex_hulls<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
    ) -> geos::GResult<O> {
        // TODO: Have this set the 'm' values to none so it is clear we shuold reconstruct this
        // as a PolygonXY[Z]
        let dim = self.coords.dim();
        let axes = axes(&self.coords);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        let mut z = dim.has_z().then(Vec::new);
        let mut m = dim.has_m().then(Vec::new);
        let mut rings = vec![0];
        let mut rows = Vec::with_capacity(self.rows.len() + 1);
        rows.push(0);
        let mut failed = Vec::new();
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            if self.rows.is_valid(i) {
                let polygon = polygon_to_geos(&axes, &self.rings, self.rows.range(i))?;
                let hull = polygon.convex_hull()?;
                // An empty hull gets no rings.
                if !hull.is_empty()? {
                    if hull.geometry_type()? != GeometryTypes::Polygon {
                        failed.push(i);
                    } else {
                        let (hull_x, hull_y, hull_z, hull_m) =
                            hull.get_exterior_ring()?.get_coord_seq()?.as_arrays()?;
                        let len = hull_x.len();
                        x.extend(hull_x);
                        y.extend(hull_y);
                        extend_axis(&mut z, hull_z, len);
                        extend_axis(&mut m, hull_m, len);
                        rings.push(x.len() as i64);
                    }
                }
            }
            rows.push(rings.len() as i64 - 1);
        }
        Ok(O::from_buffers(
            dim,
            (x, y, z, m),
            rings,
            rows,
            self.rows.validity(),
            &failed,
        ))
    }
}

/// Where GEOS dropped the axis from its result, it appends `len` NaNs instead.
#[inline(always)]
fn extend_axis(axis: &mut Option<Vec<f64>>, values: Option<Vec<f64>>, len: usize) {
    if let Some(axis) = axis {
        match values {
            Some(values) => axis.extend(values),
            None => axis.resize(axis.len() + len, f64::NAN),
        }
    }
}
