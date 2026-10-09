//! Reading a geometry column's Arrow storage from inside a kernel.
//!
//! Kernels walk the storage in place: the list offsets of rings and parts,
//! and the coordinates underneath them, without copying a row out first.

use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray, StructArray};

use std::ops::Range;

use super::coord::is_close;
use super::GeoDimension;

/// Downcast one level of the storage to the array it has to be.
/// [super::describe] already checked the dtype, so this failing is a bug.
pub(crate) fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> PolarsResult<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(
        || polars_err!(ComputeError: "expected {what} in geometry storage, got: {:?}", array.dtype()),
    )
}

/// The fewest vertices a ring can have: a triangle, plus the vertex closing it.
pub(crate) const MIN_RING_VERTICES: usize = 4;

/// View (zero-allocation) of one chunk's coordinates in the Arrow array.
pub(crate) struct CoordsView<'a> {
    coords: &'a StructArray,
    x: &'a PrimitiveArray<f64>,
    y: &'a PrimitiveArray<f64>,
    z: Option<&'a PrimitiveArray<f64>>,
    m: Option<&'a PrimitiveArray<f64>>,
}

impl<'a> CoordsView<'a> {
    pub(crate) fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        // `coord::dimension_of` fixes the field order, so x and y always come first.
        let x = downcast(coords.values()[0].as_ref(), "f64 `x` coordinates")?;
        let y = downcast(coords.values()[1].as_ref(), "f64 `y` coordinates")?;
        let field = |name: &str| {
            coords
                .fields()
                .iter()
                .position(|field| field.name.as_str() == name)
                .map(|i| downcast(coords.values()[i].as_ref(), "f64 coordinates"))
                .transpose()
        };
        let z = field("z")?;
        let m = field("m")?;
        Ok(Self { coords, x, y, z, m })
    }

    pub(crate) fn xy(&self, i: usize) -> Option<(f64, f64)> {
        let present = self.coords.is_valid(i) && self.x.is_valid(i) && self.y.is_valid(i);
        present.then(|| (self.x.value(i), self.y.value(i)))
    }

    /// Whether coordinates `i` and `j` lie at almost the same place:
    /// [`is_close`] in x, y and z. m is a measure rather than a position, so it is not compared.
    /// A missing coordinate is close to nothing.
    pub(crate) fn same_place(&self, i: usize, j: usize) -> bool {
        let (Some((ax, ay)), Some((bx, by))) = (self.xy(i), self.xy(j)) else {
            return false;
        };
        is_close(ax, bx)
            && is_close(ay, by)
            && self
                .z
                .is_none_or(|z| z.is_valid(i) && z.is_valid(j) && is_close(z.value(i), z.value(j)))
    }

    pub(crate) fn is_ring(&self, range: Range<usize>) -> bool {
        range.len() >= MIN_RING_VERTICES && self.same_place(range.start, range.end - 1)
    }

    /// The dimension these coordinates carry.
    pub(crate) fn dim(&self) -> GeoDimension {
        match (self.z.is_some(), self.m.is_some()) {
            (false, false) => GeoDimension::XY,
            (true, false) => GeoDimension::XYZ,
            (false, true) => GeoDimension::XYM,
            (true, true) => GeoDimension::XYZM,
        }
    }

    pub(crate) fn nth(&self, i: usize, n: usize) -> f64 {
        let values = match (n, self.z) {
            (0, _) => self.x,
            (1, _) => self.y,
            (2, Some(z)) => z,
            (2 | 3, _) => self.m.expect("coordinate index out of range"),
            _ => panic!("coordinate index {n} out of range"),
        };
        values.value(i)
    }
}
