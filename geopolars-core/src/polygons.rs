//! Polygons and multipolygons:
//! the list levels and coordinates that make one up, held together.

use std::ops::Range;

use polars_arrow::array::ListArray;

use crate::output::{coords_struct, list_array};
use crate::{LineString, List, ValidCoords, ValidList};

/// One polygon: a run of rings, the first of which is the exterior.
pub struct Polygon<'a> {
    pub rings: &'a ValidList,
    pub coords: &'a ValidCoords,
    pub parts: Range<usize>,
}

impl<'a> Polygon<'a> {
    /// Its rings in order, the exterior first, each as the closed linestring it is.
    pub fn rings(&self) -> impl Iterator<Item = LineString<'a>> + 'a {
        let (rings, coords) = (self.rings, self.coords);
        self.parts.clone().map(move |j| LineString {
            coords,
            vertices: rings.range(j),
        })
    }
}

/// A column of polygons: rows of rings of coordinates.
pub struct PolygonArray {
    pub rows: List,
    pub rings: ValidList,
    pub coords: ValidCoords,
}

impl PolygonArray {
    /// See [`LineStringArray::into_list_array`](crate::LineStringArray::into_list_array).
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes());
        let rings = list_array(coords.boxed(), self.rings.into_offsets(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(rings.boxed(), rows, validity)
    }

    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> Polygon<'_> {
        Polygon {
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}

/// One multipolygon: a run of polygons.
pub struct MultiPolygon<'a> {
    pub polygons: &'a ValidList,
    pub rings: &'a ValidList,
    pub coords: &'a ValidCoords,
    pub parts: Range<usize>,
}

impl<'a> MultiPolygon<'a> {
    /// Its polygons, in order.
    pub fn polygons(&self) -> impl Iterator<Item = Polygon<'a>> + 'a {
        let (polygons, rings, coords) = (self.polygons, self.rings, self.coords);
        self.parts.clone().map(move |j| Polygon {
            rings,
            coords,
            parts: polygons.range(j),
        })
    }
}

/// A column of multipolygons: rows of polygons of rings of coordinates.
pub struct MultiPolygonArray {
    pub rows: List,
    pub polygons: ValidList,
    pub rings: ValidList,
    pub coords: ValidCoords,
}

impl MultiPolygonArray {
    /// See [`LineStringArray::into_list_array`](crate::LineStringArray::into_list_array).
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes());
        let rings = list_array(coords.boxed(), self.rings.into_offsets(), None);
        let polygons = list_array(rings.boxed(), self.polygons.into_offsets(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(polygons.boxed(), rows, validity)
    }

    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> MultiPolygon<'_> {
        MultiPolygon {
            polygons: &self.polygons,
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
