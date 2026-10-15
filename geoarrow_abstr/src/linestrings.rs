//! Linestrings and multilinestrings, whatever storage backs them:
//! the list levels and coordinates that make one up, held together.

use std::ops::Range;

use crate::{GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView};

/// One linestring: a run of coordinates.
pub struct LineStringView<'a, C: GeoArrowValidCoordsView> {
    pub(crate) coords: &'a C,
    pub(crate) vertices: Range<usize>,
}

impl<'a, C: GeoArrowValidCoordsView> LineStringView<'a, C> {
    /// The position of every vertex in order: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub fn positions<const Z: bool>(&self) -> impl Iterator<Item = [f64; 3]> + 'a {
        let coords = self.coords;
        self.vertices.clone().map(move |i| coords.position::<Z>(i))
    }
}

/// One multilinestring: a run of linestrings.
pub struct MultiLineStringView<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> {
    pub(crate) lines: &'a L,
    pub(crate) coords: &'a C,
    pub(crate) parts: Range<usize>,
}

impl<'a, L: GeoArrowValidListView, C: GeoArrowValidCoordsView> MultiLineStringView<'a, L, C> {
    /// Its linestrings, in order.
    pub fn line_strings(&self) -> impl Iterator<Item = LineStringView<'a, C>> + 'a {
        let (lines, coords) = (self.lines, self.coords);
        self.parts.clone().map(move |j| LineStringView {
            coords,
            vertices: lines.range(j),
        })
    }
}

/// A column of linestrings: rows of coordinates.
pub struct LineStringArrayView<R: GeoArrowListView, C: GeoArrowValidCoordsView> {
    pub rows: R,
    pub coords: C,
}

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArrayView<R, C> {
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> LineStringView<'_, C> {
        LineStringView {
            coords: &self.coords,
            vertices: self.rows.range(i),
        }
    }
}

/// A column of multilinestrings: rows of linestrings of coordinates.
pub struct MultiLineStringArrayView<
    R: GeoArrowListView,
    L: GeoArrowValidListView,
    C: GeoArrowValidCoordsView,
> {
    pub rows: R,
    pub lines: L,
    pub coords: C,
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiLineStringArrayView<R, L, C>
{
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> MultiLineStringView<'_, L, C> {
        MultiLineStringView {
            lines: &self.lines,
            coords: &self.coords,
            parts: self.rows.range(i),
        }
    }
}
