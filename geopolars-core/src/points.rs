//! Points and multipoints: one coordinate, or a run of them, per row.

use std::ops::Range;

use polars_arrow::array::{ListArray, StructArray};

use crate::output::{coords_struct, list_array};
use crate::{Coords, List, ValidCoords};

/// One point: a single coordinate, which may be missing.
pub struct ScalarPoint<'a> {
    pub coords: &'a Coords,
    pub index: usize,
}

impl ScalarPoint<'_> {
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
    pub fn get(&self, i: usize) -> ScalarPoint<'_> {
        ScalarPoint {
            coords: &self.coords,
            index: i,
        }
    }

    #[inline(always)]
    pub fn new(coords: Coords) -> Self {
        Self { coords }
    }

    /// The polars storage of the column, taking over its buffers.
    pub fn into_struct_array(self) -> StructArray {
        let dim = self.coords.dim();
        let (axes, validity) = self.coords.into_parts();
        coords_struct(dim, axes, validity)
    }
}

/// One multipoint: a run of coordinates.
pub struct ScalarMultiPoint<'a> {
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
    /// See [`LineStringArray::into_list_array`](crate::LineStringArray::into_list_array).
    pub fn into_list_array(self) -> ListArray<i64> {
        let dim = self.coords.dim();
        let coords = coords_struct(dim, self.coords.into_axes(), None);
        let (rows, validity) = self.rows.into_parts();
        list_array(coords.boxed(), rows, validity)
    }

    /// Row `i`. A missing row's offsets point anywhere: check `rows.is_valid(i)` first.
    #[inline(always)]
    pub fn get(&self, i: usize) -> ScalarMultiPoint<'_> {
        ScalarMultiPoint {
            coords: &self.coords,
            vertices: self.rows.range(i),
        }
    }
}
