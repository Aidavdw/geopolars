//! Linestrings and multilinestrings, whatever storage backs them:
//! the list levels and coordinates that make one up, held together.

use std::ops::Range;

use crate::{GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView};

/// One linestring: a run of coordinates.
pub struct LineString<'a, C: GeoArrowValidCoordsView> {
    coords: &'a C,
    vertices: Range<usize>,
}

impl<'a, C: GeoArrowValidCoordsView> LineString<'a, C> {
    /// The position of every vertex in order: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub fn positions<const Z: bool>(&self) -> impl Iterator<Item = [f64; 3]> + 'a {
        let coords = self.coords;
        self.vertices.clone().map(move |i| coords.position::<Z>(i))
    }
}

/// One multilinestring: a run of linestrings.
pub struct MultiLineString<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> {
    lines: &'a L,
    coords: &'a C,
    parts: Range<usize>,
}

impl<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> MultiLineString<'a, L, C> {
    /// Its linestrings, in order.
    pub fn line_strings(&self) -> impl Iterator<Item = LineString<'a, C>> + 'a {
        let (lines, coords) = (self.lines, self.coords);
        self.parts.clone().map(move |j| LineString {
            coords,
            vertices: lines.range(j),
        })
    }
}

/// A column of linestrings: rows of coordinates.
pub struct LineStringArray<R: GeoArrowListView, C: GeoArrowValidCoordsView> {
    pub rows: R,
    pub coords: C,
}

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArray<R, C> {
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> LineString<'_, C> {
        LineString {
            coords: &self.coords,
            vertices: self.rows.range(i),
        }
    }
}

/// A column of multilinestrings: rows of linestrings of coordinates.
pub struct MultiLineStringArray<
    R: GeoArrowListView,
    L: GeoArrowValidListView,
    C: GeoArrowValidCoordsView,
> {
    pub rows: R,
    pub lines: L,
    pub coords: C,
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiLineStringArray<R, L, C>
{
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> MultiLineString<'_, L, C> {
        MultiLineString {
            lines: &self.lines,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
