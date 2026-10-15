//! Geoarrow-implementation-agnostic Polygons

use std::ops::Range;

use crate::{GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, LineStringView};

/// One polygon: a run of rings, the first of which is the exterior.
pub struct PolygonView<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> {
    pub rings: &'a L,
    pub coords: &'a C,
    pub parts: Range<usize>,
}

impl<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> PolygonView<'a, L, C> {
    /// Its rings in order, the exterior first, each as the closed linestring it is.
    pub fn rings(&self) -> impl Iterator<Item = LineStringView<'a, C>> + 'a {
        let (rings, coords) = (self.rings, self.coords);
        self.parts.clone().map(move |j| LineStringView {
            coords,
            vertices: rings.range(j),
        })
    }
}

/// A column of polygons: rows of rings of coordinates.
pub struct PolygonArrayView<
    R: GeoArrowListView,
    L: GeoArrowValidListView,
    C: GeoArrowValidCoordsView,
> {
    pub rows: R,
    pub rings: L,
    pub coords: C,
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArrayView<R, L, C>
{
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> PolygonView<'_, L, C> {
        PolygonView {
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}

/// One multipolygon: a run of polygons.
pub struct MultiPolygonView<
    'a,
    P: GeoArrowValidListView,
    L: GeoArrowValidListView,
    C: GeoArrowValidCoordsView,
> {
    pub polygons: &'a P,
    pub rings: &'a L,
    pub coords: &'a C,
    pub parts: Range<usize>,
}

impl<'a, P: GeoArrowValidListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiPolygonView<'a, P, L, C>
{
    /// Its polygons, in order.
    pub fn polygons(&self) -> impl Iterator<Item = PolygonView<'a, L, C>> + 'a {
        let (polygons, rings, coords) = (self.polygons, self.rings, self.coords);
        self.parts.clone().map(move |j| PolygonView {
            rings,
            coords,
            parts: polygons.range(j),
        })
    }
}

/// A column of multipolygons: rows of polygons of rings of coordinates.
pub struct MultiPolygonArrayView<
    R: GeoArrowListView,
    P: GeoArrowValidListView,
    L: GeoArrowValidListView,
    C: GeoArrowValidCoordsView,
> {
    pub rows: R,
    pub polygons: P,
    pub rings: L,
    pub coords: C,
}

impl<
        R: GeoArrowListView,
        P: GeoArrowValidListView,
        L: GeoArrowValidListView,
        C: GeoArrowValidCoordsView,
    > MultiPolygonArrayView<R, P, L, C>
{
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> MultiPolygonView<'_, P, L, C> {
        MultiPolygonView {
            polygons: &self.polygons,
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
