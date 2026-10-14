//! See the expression implementation guide on Array abstractions for GeoArrow.

use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;

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

/// Abstraction over Polars storage for coordinates that are all assumed to be valid.
/// Use for coordinates that **cannot** include invalid coordinates (columns of Points).
/// Otherwise, use [CoordsView].
///
/// See the expression implementation guide for more details.
pub(crate) struct ValidCoordsView<'a> {
    x: &'a [f64],
    y: &'a [f64],
    /// Empty for a dimension without a `z`.
    z: &'a [f64],
    /// Empty for a dimension without an `m`.
    m: &'a [f64],
}

impl<'a> ValidCoordsView<'a> {
    /// The axes, and the dimension they carry.
    /// (An empty chunk has empty axes either way, so the dimension can't be read off them.)
    pub(crate) fn new(coords: &'a dyn Array) -> PolarsResult<(Self, GeoDimension)> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let arrays = AxisArrays::new(coords)?;
        let axis = |values: Option<&'a PrimitiveArray<f64>>| {
            values.map_or(&[][..], |values| values.values().as_slice())
        };
        let axes = Self {
            x: axis(Some(arrays.x)),
            y: axis(Some(arrays.y)),
            z: axis(arrays.z),
            m: axis(arrays.m),
        };
        Ok((axes, arrays.dim()))
    }

    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub(crate) fn position<const Z: bool>(&self, i: usize) -> [f64; 3] {
        [self.x[i], self.y[i], if Z { self.z[i] } else { 0.0 }]
    }

    /// The m of coordinate `i`. Only for a dimension with an `m`.
    #[inline(always)]
    pub(crate) fn m(&self, i: usize) -> f64 {
        self.m[i]
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

    /// The dimension these axes carry.
    pub(crate) fn dim(&self) -> GeoDimension {
        match (self.z.is_some(), self.m.is_some()) {
            (false, false) => GeoDimension::XY,
            (true, false) => GeoDimension::XYZ,
            (false, true) => GeoDimension::XYM,
            (true, true) => GeoDimension::XYZM,
        }
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
    dim: GeoDimension,
}

impl<'a> CoordsView<'a> {
    pub(crate) fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let (axes, dim) = ValidCoordsView::new(coords)?;
        Ok(Self {
            validity: coords.validity(),
            axes,
            dim,
        })
    }

    /// How many coordinates there are.
    pub(crate) fn len(&self) -> usize {
        self.axes.x.len()
    }

    #[inline(always)]
    fn is_valid(&self, i: usize) -> bool {
        self.validity.is_none_or(|validity| validity.get_bit(i))
    }

    pub(crate) fn xy(&self, i: usize) -> Option<(f64, f64)> {
        self.xy_checking_nulls::<true>(i)
    }

    /// [`xy`](Self::xy), checking whether it is missing only with `NULLS`.
    /// Without, it assumes every coordinate is there.
    #[inline(always)]
    pub(crate) fn xy_checking_nulls<const NULLS: bool>(&self, i: usize) -> Option<(f64, f64)> {
        (!NULLS || self.is_valid(i)).then(|| (self.axes.x[i], self.axes.y[i]))
    }

    /// The position of coordinate `i`: x, y, and z (0 without a z axis).
    /// With `NULLS`, `None` if it is missing.
    /// Without, it assumes every coordinate is there.
    #[inline(always)]
    pub(crate) fn position<const NULLS: bool>(&self, i: usize) -> Option<[f64; 3]> {
        let (x, y) = self.xy_checking_nulls::<NULLS>(i)?;
        let z = if self.axes.z.is_empty() {
            0.0
        } else {
            self.axes.z[i]
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
        self.dim
    }

    pub(crate) fn nth(&self, i: usize, n: usize) -> f64 {
        let values = match n {
            0 => self.axes.x,
            1 => self.axes.y,
            2 if !self.axes.z.is_empty() => self.axes.z,
            // Empty without an m, so indexing it panics.
            2 | 3 => self.axes.m,
            _ => panic!("coordinate index {n} out of range"),
        };
        values[i]
    }
}
