//! Polygons and multipolygons:
//! the list levels and coordinates that make one up, held together.

use std::ops::Range;

use polars_arrow::array::ListArray;
use polars_core::prelude::PolarsResult;

use crate::lists::UNTRIMMED;
use crate::output::{coords_struct, list_array};
use crate::{downcast, List, ScalarLineString, ValidCoords, ValidList};

/// One polygon: a run of rings, the first of which is the exterior.
pub struct ScalarPolygon<'a> {
    pub rings: &'a ValidList,
    pub coords: &'a ValidCoords,
    pub parts: Range<usize>,
}

impl<'a> ScalarPolygon<'a> {
    /// Its rings in order, the exterior first, each as the closed linestring it is.
    pub fn rings(&self) -> impl Iterator<Item = ScalarLineString<'a>> + 'a {
        let (rings, coords) = (self.rings, self.coords);
        self.parts.clone().map(move |j| ScalarLineString {
            coords,
            vertices: rings.range(j),
        })
    }

    /// Whether it has no exterior, or an empty one.
    /// Either makes it the empty polygon, whose holes GeoArrow also requires to be empty.
    #[inline(always)]
    pub fn has_empty_exterior(&self) -> bool {
        self.rings()
            .next()
            .is_none_or(|exterior| exterior.vertices.is_empty())
    }
}

/// A column of polygons: rows of rings of coordinates.
pub struct PolygonArray {
    pub rows: List,
    pub rings: ValidList,
    pub coords: ValidCoords,
}

impl PolygonArray {
    pub fn new(rows: &ListArray<i64>) -> PolarsResult<Self> {
        let rings: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of rings")?;
        Ok(Self {
            rows: List::new(rows),
            rings: ValidList::new(rings),
            coords: ValidCoords::new(rings.values().as_ref())?,
        })
    }

    /// See [`LineStringArray::is_trimmed`](crate::LineStringArray::is_trimmed).
    pub fn is_trimmed(&self) -> bool {
        self.rows.ignoring_validity().spans_all_of(self.rings.len())
            && self.rings.spans_all_of(self.coords.len())
    }

    /// See [`LineStringArray::debug_assert_trimmed`](crate::LineStringArray::debug_assert_trimmed).
    #[track_caller]
    #[inline]
    pub fn debug_assert_trimmed(&self) {
        debug_assert!(self.is_trimmed(), "{UNTRIMMED}");
    }

    /// See [`LineStringArray::into_list_array`](crate::LineStringArray::into_list_array).
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes(), None);
        let rings = list_array(coords.boxed(), self.rings.into_offsets(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(rings.boxed(), rows, validity)
    }

    /// Row `i`. A null row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarPolygon<'_> {
        ScalarPolygon {
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}

/// One multipolygon: a run of polygons.
pub struct ScalarMultiPolygon<'a> {
    pub polygons: &'a ValidList,
    pub rings: &'a ValidList,
    pub coords: &'a ValidCoords,
    pub parts: Range<usize>,
}

impl<'a> ScalarMultiPolygon<'a> {
    /// Its polygons, in order.
    pub fn polygons(&self) -> impl Iterator<Item = ScalarPolygon<'a>> + 'a {
        let (polygons, rings, coords) = (self.polygons, self.rings, self.coords);
        self.parts.clone().map(move |j| ScalarPolygon {
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
    pub fn new(rows: &ListArray<i64>) -> PolarsResult<Self> {
        let polygons: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of polygons")?;
        let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
        Ok(Self {
            rows: List::new(rows),
            polygons: ValidList::new(polygons),
            rings: ValidList::new(rings),
            coords: ValidCoords::new(rings.values().as_ref())?,
        })
    }

    /// See [`LineStringArray::is_trimmed`](crate::LineStringArray::is_trimmed).
    pub fn is_trimmed(&self) -> bool {
        self.rows
            .ignoring_validity()
            .spans_all_of(self.polygons.len())
            && self.polygons.spans_all_of(self.rings.len())
            && self.rings.spans_all_of(self.coords.len())
    }

    /// See [`LineStringArray::debug_assert_trimmed`](crate::LineStringArray::debug_assert_trimmed).
    #[track_caller]
    #[inline]
    pub fn debug_assert_trimmed(&self) {
        debug_assert!(self.is_trimmed(), "{UNTRIMMED}");
    }

    /// See [`LineStringArray::into_list_array`](crate::LineStringArray::into_list_array).
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes(), None);
        let rings = list_array(coords.boxed(), self.rings.into_offsets(), None);
        let polygons = list_array(rings.boxed(), self.polygons.into_offsets(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(polygons.boxed(), rows, validity)
    }

    /// Row `i`. A null row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarMultiPolygon<'_> {
        ScalarMultiPolygon {
            polygons: &self.polygons,
            rings: &self.rings,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
