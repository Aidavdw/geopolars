//! Views that may hold lists and coordinates outside their rows, and trimming them.
//! See the README on trimming.

use crate::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
};

// FIXME: the fields of the views are `pub`.
// So a struct literal outside this crate can build a plain view from untrimmed parts
// (e.g. those of `rows_only()`), which then claims to be trimmed.
// Close this with `#[non_exhaustive]` on the views,
// and a `from_parts` constructor that debug-asserts `is_trimmed`.
// The few field assignments outside this crate should go through it too.

/// A view that may hold lists and coordinates outside its rows:
/// a chunk sliced from a larger one, before trimming.
/// Only for kernels that go row by row, so they can skip the cost of trimming.
pub struct Untrimmed<T>(pub(crate) T);

impl<T> Untrimmed<T> {
    /// The view, to be read row by row only.
    #[inline]
    pub fn rows_only(&self) -> &T {
        &self.0
    }
}

impl Untrimmed<LineStringArray> {
    /// The view, with every level sliced to what its rows use.
    /// Free when it already is.
    pub fn trimmed(self) -> LineStringArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, coords) = array.rows.sliced(0..array.rows.len());
        LineStringArray {
            rows,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<MultiPointArray> {
    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> MultiPointArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, coords) = array.rows.sliced(0..array.rows.len());
        MultiPointArray {
            rows,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<PolygonArray> {
    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> PolygonArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, rings) = array.rows.sliced(0..array.rows.len());
        let (rings, coords) = array.rings.sliced(rings);
        PolygonArray {
            rows,
            rings,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<MultiLineStringArray> {
    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> MultiLineStringArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, lines) = array.rows.sliced(0..array.rows.len());
        let (lines, coords) = array.lines.sliced(lines);
        MultiLineStringArray {
            rows,
            lines,
            coords: array.coords.sliced(coords),
        }
    }
}

impl Untrimmed<MultiPolygonArray> {
    /// See [`Untrimmed::<LineStringArray>::trimmed`].
    pub fn trimmed(self) -> MultiPolygonArray {
        let array = self.0;
        if array.is_trimmed() {
            return array;
        }
        let (rows, polygons) = array.rows.sliced(0..array.rows.len());
        let (polygons, rings) = array.polygons.sliced(polygons);
        let (rings, coords) = array.rings.sliced(rings);
        MultiPolygonArray {
            rows,
            polygons,
            rings,
            coords: array.coords.sliced(coords),
        }
    }
}
