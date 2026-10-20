//! Linestrings and multilinestrings:
//! the list levels and coordinates that make one up, held together.

use std::ops::Range;

use polars_arrow::array::ListArray;

use crate::output::{coords_struct, list_array};
use crate::{List, ValidCoords, ValidList};

/// One linestring: a run of coordinates.
pub struct LineString<'a> {
    pub coords: &'a ValidCoords,
    pub vertices: Range<usize>,
}

impl<'a> LineString<'a> {
    /// The position of every vertex in order: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub fn positions<const Z: bool>(&self) -> impl Iterator<Item = [f64; 3]> + 'a {
        let coords = self.coords;
        self.vertices.clone().map(move |i| coords.position::<Z>(i))
    }
}

/// One multilinestring: a run of linestrings.
pub struct MultiLineString<'a> {
    pub lines: &'a ValidList,
    pub coords: &'a ValidCoords,
    pub parts: Range<usize>,
}

impl<'a> MultiLineString<'a> {
    /// Its linestrings, in order.
    pub fn line_strings(&self) -> impl Iterator<Item = LineString<'a>> + 'a {
        let (lines, coords) = (self.lines, self.coords);
        self.parts.clone().map(move |j| LineString {
            coords,
            vertices: lines.range(j),
        })
    }
}

/// A column of linestrings: rows of coordinates.
pub struct LineStringArray {
    pub rows: List,
    pub coords: ValidCoords,
}

impl LineStringArray {
    /// The polars storage of the column, taking over its buffers.
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes());
        let (rows, validity) = self.rows.into_parts();
        list_array(coords.boxed(), rows, validity)
    }

    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> LineString<'_> {
        LineString {
            coords: &self.coords,
            vertices: self.rows.range(i),
        }
    }
}

/// A column of multilinestrings: rows of linestrings of coordinates.
pub struct MultiLineStringArray {
    pub rows: List,
    pub lines: ValidList,
    pub coords: ValidCoords,
}

impl MultiLineStringArray {
    /// See [`LineStringArray::into_list_array`].
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes());
        let lines = list_array(coords.boxed(), self.lines.into_offsets(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(lines.boxed(), rows, validity)
    }

    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> MultiLineString<'_> {
        MultiLineString {
            lines: &self.lines,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
