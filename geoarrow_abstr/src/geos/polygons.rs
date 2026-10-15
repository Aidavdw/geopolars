//! Polygons and multipolygons to GEOS.

use std::ops::Range;

use geos::Geometry;

use super::{axes, coord_seq, Axes};
use crate::{
    GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, MultiPolygon,
    MultiPolygonArray, Polygon, PolygonArray,
};

impl<L: GeoArrowValidListView, C: GeoArrowValidCoordsView> Polygon<'_, L, C> {
    pub fn to_geos(&self) -> geos::GResult<Geometry> {
        polygon_to_geos(&axes(self.coords), self.rings, self.parts.clone())
    }
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArray<R, L, C>
{
    /// One geometry per row; `None` where the row is missing.
    pub fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        let axes = axes(&self.coords);
        let mut geometries = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            geometries.push(if self.rows.is_valid(i) {
                Some(polygon_to_geos(&axes, &self.rings, self.rows.range(i))?)
            } else {
                None
            });
        }
        Ok(geometries)
    }
}

impl<P: GeoArrowValidListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiPolygon<'_, P, L, C>
{
    pub fn to_geos(&self) -> geos::GResult<Geometry> {
        multi_polygon_to_geos(
            &axes(self.coords),
            self.polygons,
            self.rings,
            self.parts.clone(),
        )
    }
}

impl<
        R: GeoArrowListView,
        P: GeoArrowValidListView,
        L: GeoArrowValidListView,
        C: GeoArrowValidCoordsView,
    > MultiPolygonArray<R, P, L, C>
{
    /// One geometry per row; `None` where the row is missing.
    pub fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        let axes = axes(&self.coords);
        let mut geometries = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            geometries.push(if self.rows.is_valid(i) {
                Some(multi_polygon_to_geos(
                    &axes,
                    &self.polygons,
                    &self.rings,
                    self.rows.range(i),
                )?)
            } else {
                None
            });
        }
        Ok(geometries)
    }
}

/// The polygon of `rings` `parts`, the first of which is the exterior.
/// No parts gives an empty polygon.
#[inline(always)]
fn polygon_to_geos(
    axes: &Axes,
    rings: &impl GeoArrowValidListView,
    parts: Range<usize>,
) -> geos::GResult<Geometry> {
    let mut parts = parts.map(|j| Geometry::create_linear_ring(coord_seq(axes, rings.range(j))?));
    // GEOS can't build a polygon without an exterior, so an empty one has its own constructor.
    let Some(exterior) = parts.next() else {
        return Geometry::create_empty_polygon();
    };
    Geometry::create_polygon(exterior?, parts.collect::<geos::GResult<_>>()?)
}

/// The multipolygon of `polygons` `parts`.
/// No parts gives an empty multipolygon.
#[inline(always)]
fn multi_polygon_to_geos(
    axes: &Axes,
    polygons: &impl GeoArrowValidListView,
    rings: &impl GeoArrowValidListView,
    parts: Range<usize>,
) -> geos::GResult<Geometry> {
    // GEOS takes ownership of the parts, so they can't share a buffer across rows.
    let mut geometries = Vec::with_capacity(parts.len());
    for j in parts {
        geometries.push(polygon_to_geos(axes, rings, polygons.range(j))?);
    }
    Geometry::create_multipolygon(geometries)
}
