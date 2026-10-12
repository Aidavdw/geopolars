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

/// Whether two [positions](CoordsView::position) are [`is_close`] on every axis.
#[inline(always)]
pub(crate) fn same_position(a: [f64; 3], b: [f64; 3]) -> bool {
    is_close(a[0], b[0]) && is_close(a[1], b[1]) && is_close(a[2], b[2])
}

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

    /// Whether coordinate `i` is there, with every one of its axes.
    pub(crate) fn present(&self, i: usize) -> bool {
        self.coords.is_valid(i) && self.axes().all(|axis| axis.is_valid(i))
    }

    /// Every axis of coordinate `i`, in x, y, z, m order.
    /// Only meaningful where it is [`present`](Self::present).
    pub(crate) fn values(&self, i: usize) -> impl Iterator<Item = f64> + '_ {
        self.axes().map(move |axis| axis.value(i))
    }

    fn axes(&self) -> impl Iterator<Item = &'a PrimitiveArray<f64>> {
        [Some(self.x), Some(self.y), self.z, self.m]
            .into_iter()
            .flatten()
    }

    /// Whether any coordinate, or any axis of one, is missing.
    /// Without any, the `NULLS = false` readers below can skip every validity lookup.
    pub(crate) fn has_missing_values(&self) -> bool {
        self.coords.null_count() > 0 || self.axes().any(|axis| axis.null_count() > 0)
    }

    pub(crate) fn xy(&self, i: usize) -> Option<(f64, f64)> {
        self.xy_checking_nulls::<true>(i)
    }

    /// [`xy`](Self::xy), checking whether it is missing only with `NULLS`.
    /// Without, it assumes there are [no missing values](Self::has_missing_values).
    #[inline(always)]
    pub(crate) fn xy_checking_nulls<const NULLS: bool>(&self, i: usize) -> Option<(f64, f64)> {
        let present =
            !NULLS || (self.coords.is_valid(i) && self.x.is_valid(i) && self.y.is_valid(i));
        present.then(|| (self.x.value(i), self.y.value(i)))
    }

    /// The position of coordinate `i`: x, y, and z (0 without a z axis).
    /// With `NULLS`, `None` if one of those is missing.
    /// Without, it assumes there are [no missing values](Self::has_missing_values).
    #[inline(always)]
    pub(crate) fn position<const NULLS: bool>(&self, i: usize) -> Option<[f64; 3]> {
        let (x, y) = self.xy_checking_nulls::<NULLS>(i)?;
        let z = match self.z {
            Some(z) if NULLS && !z.is_valid(i) => return None,
            Some(z) => z.value(i),
            None => 0.0,
        };
        Some([x, y, z])
    }

    /// Whether coordinates `i` and `j` lie at almost the same place:
    /// [`is_close`] in x, y and z. m is a measure rather than a position, so it is not compared.
    /// A missing coordinate is close to nothing.
    pub(crate) fn same_place(&self, i: usize, j: usize) -> bool {
        matches!(
            (self.position::<true>(i), self.position::<true>(j)),
            (Some(a), Some(b)) if same_position(a, b)
        )
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
