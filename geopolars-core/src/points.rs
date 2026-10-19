//! Points and multipoints: one coordinate, or a run of them, per row.

use std::ops::Range;

use crate::{Coords, List, ValidCoords};

/// One point: a single coordinate, which may be missing.
pub struct Point<'a> {
    pub coords: &'a Coords,
    pub index: usize,
}

impl Point<'_> {
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
pub struct PointArray {
    pub coords: Coords,
}

impl PointArray {
    /// Row `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> Point<'_> {
        Point {
            coords: &self.coords,
            index: i,
        }
    }

    #[inline(always)]
    pub fn new(coords: Coords) -> Self {
        Self { coords }
    }
}

/// One multipoint: a run of coordinates.
pub struct MultiPoint<'a> {
    pub coords: &'a ValidCoords,
    pub vertices: Range<usize>,
}

/// A column of multipoints: rows of coordinates.
/// The points of a multipoint are never missing, only a whole row is.
pub struct MultiPointArray {
    pub rows: List,
    pub coords: ValidCoords,
}

impl MultiPointArray {
    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> MultiPoint<'_> {
        MultiPoint {
            coords: &self.coords,
            vertices: self.rows.range(i),
        }
    }
}
