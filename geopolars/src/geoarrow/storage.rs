//! Reading a geometry column's Arrow storage from inside a kernel.
//!
//! Kernels walk the storage in place: the list offsets of rings and parts,
//! and the coordinates underneath them, without copying a row out first.

use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray, StructArray};

/// Downcast one level of the storage to the array it has to be.
/// [super::describe] already checked the dtype, so this failing is a bug.
pub(crate) fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> PolarsResult<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(
        || polars_err!(ComputeError: "expected {what} in geometry storage, got: {:?}", array.dtype()),
    )
}

/// View (zero-allocation) of the  `x` and `y` of one chunk's coordinates in the Arrow array
pub(crate) struct CoordsView<'a> {
    coords: &'a StructArray,
    x: &'a PrimitiveArray<f64>,
    y: &'a PrimitiveArray<f64>,
}

impl<'a> CoordsView<'a> {
    pub(crate) fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        // `coord::dimension_of` fixes the field order, so x and y always come first.
        let x = downcast(coords.values()[0].as_ref(), "f64 `x` coordinates")?;
        let y = downcast(coords.values()[1].as_ref(), "f64 `y` coordinates")?;
        Ok(Self { coords, x, y })
    }

    pub(crate) fn xy(&self, i: usize) -> Option<(f64, f64)> {
        let present = self.coords.is_valid(i) && self.x.is_valid(i) && self.y.is_valid(i);
        present.then(|| (self.x.value(i), self.y.value(i)))
    }
}
