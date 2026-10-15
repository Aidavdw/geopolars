//! See the expression implementation guide on Array abstractions for GeoArrow.

use geoarrow_abstr::{
    GeoArrowCoordsShared, GeoArrowCoordsView, GeoArrowF64Array, GeoArrowListShared,
    GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, GeoDimension,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::combine_validities_and3;
use polars_arrow::offset::OffsetsBuffer;

use std::ops::Range;

use super::coord::is_close;

/// Downcast one level of the storage to the array it has to be.
/// [super::describe] already checked the dtype, so this failing is a bug.
pub(crate) fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> PolarsResult<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(
        || polars_err!(ComputeError: "expected {what} in geometry storage, got: {:?}", array.dtype()),
    )
}

/// The fewest vertices a ring can have: a triangle, plus the vertex closing it.
pub(crate) const MIN_RING_VERTICES: usize = 4;

/// Whether two [positions](GeoArrowCoordsView::position) are [`is_close`] on every axis.
#[inline(always)]
pub(crate) fn same_position(a: [f64; 3], b: [f64; 3]) -> bool {
    is_close(a[0], b[0]) && is_close(a[1], b[1]) && is_close(a[2], b[2])
}

/// Abstraction over Polars storage for coordinates that are all assumed to be valid.
/// Use for coordinates that **cannot** include invalid coordinates (columns of Points).
/// Otherwise, use [CoordsView].
///
/// See the expression implementation guide for more details.
pub(crate) struct ValidCoordsView<'a> {
    x: &'a [f64],
    y: &'a [f64],
    /// `None` for a dimension without a `z`.
    z: Option<&'a [f64]>,
    /// `None` for a dimension without an `m`.
    m: Option<&'a [f64]>,
}

impl<'a> ValidCoordsView<'a> {
    pub(crate) fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let arrays = AxisArrays::new(coords)?;
        Ok(Self {
            x: arrays.x.values().as_slice(),
            y: arrays.y.values().as_slice(),
            z: arrays.z.map(|z| z.values().as_slice()),
            m: arrays.m.map(|m| m.values().as_slice()),
        })
    }

    /// [`same_place`](GeoArrowCoordsShared::same_place), comparing z only with `Z`.
    fn same_place_in<const Z: bool>(&self, i: usize, j: usize) -> bool {
        same_position(self.position::<Z>(i), self.position::<Z>(j))
    }
}

impl GeoArrowValidCoordsView for ValidCoordsView<'_> {
    #[inline(always)]
    fn position<const Z: bool>(&self, i: usize) -> [f64; 3] {
        [
            self.x[i],
            self.y[i],
            if Z { self.z.unwrap()[i] } else { 0.0 },
        ]
    }

    #[inline(always)]
    fn m(&self, i: usize) -> f64 {
        self.m.unwrap()[i]
    }
}

impl GeoArrowCoordsShared for ValidCoordsView<'_> {
    fn len(&self) -> usize {
        self.x.len()
    }

    fn nth(&self, i: usize, n: usize) -> f64 {
        let values = match n {
            0 => Some(self.x),
            1 => Some(self.y),
            2 => self.z.or(self.m),
            3 => self.m,
            _ => None,
        };
        values.unwrap_or_else(|| panic!("coordinate index {n} out of range"))[i]
    }

    fn same_place(&self, i: usize, j: usize) -> bool {
        if self.z.is_none() {
            self.same_place_in::<false>(i, j)
        } else {
            self.same_place_in::<true>(i, j)
        }
    }

    fn is_ring(&self, range: Range<usize>) -> bool {
        range.len() >= MIN_RING_VERTICES && self.same_place(range.start, range.end - 1)
    }

    fn dim(&self) -> GeoDimension {
        match (self.z.is_some(), self.m.is_some()) {
            (false, false) => GeoDimension::XY,
            (true, false) => GeoDimension::XYZ,
            (false, true) => GeoDimension::XYM,
            (true, true) => GeoDimension::XYZM,
        }
    }
}

/// Polars-backed coordinate data that might be potentially invalid.
/// Considers validity of individual components of the coordinates
/// ('x' may be valid, but 'y' may not be).
/// Use this in validation only.
///
/// See the expression implementation guide for more details.
pub(crate) struct AxisArrays<'a> {
    pub(crate) x: &'a PrimitiveArray<f64>,
    pub(crate) y: &'a PrimitiveArray<f64>,
    /// `None` for a dimension without a `z`.
    pub(crate) z: Option<&'a PrimitiveArray<f64>>,
    /// `None` for a dimension without an `m`.
    pub(crate) m: Option<&'a PrimitiveArray<f64>>,
}

impl<'a> AxisArrays<'a> {
    pub(crate) fn new(coords: &'a StructArray) -> PolarsResult<Self> {
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
        Ok(Self {
            x,
            y,
            z: field("z")?,
            m: field("m")?,
        })
    }
}

/// Abstraction over Polars storage for coordinates where individual coordinates can be invalid.
/// Use for coordinates that **can** include invalid coordinates (columns of Points).
/// Otherwise, use [ValidCoordsView].
///
/// See the expression implementation guide for more details.
pub(crate) struct CoordsView<'a> {
    /// Which coordinates are there.
    validity: Option<&'a Bitmap>,
    axes: ValidCoordsView<'a>,
}

impl<'a> CoordsView<'a> {
    pub(crate) fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        Ok(Self {
            validity: coords.validity(),
            axes: ValidCoordsView::new(coords)?,
        })
    }

    #[inline(always)]
    fn is_valid(&self, i: usize) -> bool {
        self.validity.is_none_or(|validity| validity.get_bit(i))
    }
}

impl GeoArrowCoordsView for CoordsView<'_> {
    type Validity = Bitmap;

    #[inline]
    fn validity(&self) -> Option<&Bitmap> {
        self.validity
    }

    fn xy(&self, i: usize) -> Option<(f64, f64)> {
        self.xy_checking_nulls::<true>(i)
    }

    #[inline(always)]
    fn xy_checking_nulls<const NULLS: bool>(&self, i: usize) -> Option<(f64, f64)> {
        (!NULLS || self.is_valid(i)).then(|| (self.axes.x[i], self.axes.y[i]))
    }

    #[inline(always)]
    fn position<const NULLS: bool, const Z: bool>(&self, i: usize) -> Option<[f64; 3]> {
        (!NULLS || self.is_valid(i)).then(|| self.axes.position::<Z>(i))
    }
}

impl GeoArrowCoordsShared for CoordsView<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.axes.len()
    }

    #[inline]
    fn nth(&self, i: usize, n: usize) -> f64 {
        self.axes.nth(i, n)
    }

    #[inline]
    fn same_place(&self, i: usize, j: usize) -> bool {
        self.axes.same_place(i, j)
    }

    #[inline]
    fn is_ring(&self, range: Range<usize>) -> bool {
        self.axes.is_ring(range)
    }

    #[inline]
    fn dim(&self) -> GeoDimension {
        self.axes.dim()
    }
}

/// Abstraction over Polars storage for lists that are all assumed to be valid:
/// the inner levels of a nested geometry (the linestrings of a multilinestring).
/// Otherwise, use [ListView].
///
/// See the expression implementation guide for more details.
pub(crate) struct ValidListView<'a> {
    offsets: &'a OffsetsBuffer<i64>,
}

impl<'a> ValidListView<'a> {
    pub(crate) fn new(list: &'a ListArray<i64>) -> Self {
        Self {
            offsets: list.offsets(),
        }
    }
}

impl GeoArrowListShared for ValidListView<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    #[inline(always)]
    fn range(&self, i: usize) -> Range<usize> {
        let (start, end) = self.offsets.start_end(i);
        start..end
    }
}

impl GeoArrowValidListView for ValidListView<'_> {}

/// Abstraction over Polars storage for lists where individual lists can be missing:
/// the outer level of a geometry column.
/// Otherwise, use [ValidListView].
///
/// See the expression implementation guide for more details.
pub(crate) struct ListView<'a> {
    /// Which lists are there.
    validity: Option<&'a Bitmap>,
    lists: ValidListView<'a>,
}

impl<'a> ListView<'a> {
    pub(crate) fn new(list: &'a ListArray<i64>) -> Self {
        Self {
            validity: list.validity(),
            lists: ValidListView::new(list),
        }
    }
}

impl GeoArrowListShared for ListView<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.lists.len()
    }

    #[inline(always)]
    fn range(&self, i: usize) -> Range<usize> {
        self.lists.range(i)
    }
}

impl GeoArrowListView for ListView<'_> {
    type Validity = Bitmap;

    #[inline(always)]
    fn is_valid(&self, i: usize) -> bool {
        self.validity.is_none_or(|validity| validity.get_bit(i))
    }

    #[inline]
    fn validity(&self) -> Option<&Bitmap> {
        self.validity
    }
}

/// [GeoArrowF64Array] for Polars.
/// A newtype, because the orphan rule forbids implementing the trait for `PrimitiveArray<f64>` here.
pub(crate) struct F64Array(pub(crate) PrimitiveArray<f64>);

impl GeoArrowF64Array for F64Array {
    type Validity = Bitmap;

    fn from_values(values: Vec<f64>, validity: Option<&Bitmap>) -> Self {
        Self(PrimitiveArray::from_vec(values).with_validity(validity.cloned()))
    }

    fn from_values_and(
        values: Vec<f64>,
        a: Option<&Bitmap>,
        b: Option<&Bitmap>,
        failed: &[usize],
    ) -> Self {
        let measured = (!failed.is_empty()).then(|| {
            let mut measured = MutableBitmap::from_len_set(values.len());
            for &i in failed {
                measured.set(i, false);
            }
            Bitmap::from(measured)
        });
        let validity = combine_validities_and3(a, b, measured.as_ref());
        Self(PrimitiveArray::from_vec(values).with_validity(validity))
    }
}

// TODO: if we already know that we do not have any nulls, we can take a faster path.
// We should add a function to the GeoArrowCoordsView trait that allows us to check
// if it has any nulls at all. For the polars arrow impl, we can simply check if validity
// is 'some'
