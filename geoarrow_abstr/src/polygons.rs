//! Geoarrow-implementation-agnostic Polygons

use std::ops::Range;

use crate::{GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, LineString};

/// One polygon: a run of rings, the first of which is the exterior.
pub struct Polygon<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> {
    rings: &'a L,
    coords: &'a C,
    parts: Range<usize>,
}

impl<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> Polygon<'a, L, C> {
    /// Its rings in order, the exterior first, each as the closed linestring it is.
    pub fn rings(&self) -> impl Iterator<Item = LineString<'a, C>> + 'a {
        let (rings, coords) = (self.rings, self.coords);
        self.parts.clone().map(move |j| LineString {
            coords,
            vertices: rings.range(j),
        })
    }
}

/// A column of polygons: rows of rings of coordinates.
pub struct PolygonArray<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> {
    pub rows: R,
    pub rings: L,
    pub coords: C,
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArray<R, L, C>
{
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> Polygon<'_, L, C> {
        Polygon {
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
