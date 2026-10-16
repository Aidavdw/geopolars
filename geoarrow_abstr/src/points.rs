//! Points, whatever storage backs them: one coordinate per row.

use crate::{GeoArrowCoordsView, GeoArrowListView, GeoArrowValidCoordsView};

/// One point: a single coordinate, which may be missing.
pub struct PointView<'a, C: GeoArrowCoordsView> {
    pub(crate) coords: &'a C,
    pub(crate) index: usize,
}

impl<C: GeoArrowCoordsView> PointView<'_, C> {
    /// Its position: x, y, and with `Z` its z (`0.0` without).
    /// With `NULLS`, `None` if it is missing.
    /// Without, it assumes it is there.
    #[inline(always)]
    pub fn position<const NULLS: bool, const Z: bool>(&self) -> Option<[f64; 3]> {
        self.coords.position::<NULLS, Z>(self.index)
    }
}

/// A column of points: one coordinate per row,
/// with missing rows recorded in the coordinates' own validity.
pub struct PointArrayView<C: GeoArrowCoordsView> {
    pub coords: C,
}

impl<C: GeoArrowCoordsView> PointArrayView<C> {
    /// Row `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> PointView<'_, C> {
        PointView {
            coords: &self.coords,
            index: i,
        }
    }

    #[inline(always)]
    pub fn new(coords: C) -> Self {
        Self { coords }
    }
}

/// A column of multipoints: rows of coordinates.
/// The points of a multipoint are never missing, only a whole row is.
pub struct MultiPointArrayView<R: GeoArrowListView, C: GeoArrowValidCoordsView> {
    pub rows: R,
    pub coords: C,
}
