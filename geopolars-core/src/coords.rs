//! Views over the coordinate struct of a geometry column.

use std::ops::Range;

use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::prelude::PolarsResult;

use crate::{downcast, GeoDimension};

/// The fewest vertices a ring can have: a triangle, plus the vertex closing it.
pub const MIN_RING_VERTICES: usize = 4;

/// The relative tolerance of [`is_close`], as `pl.Expr.is_close`'s default.
pub const REL_TOL: f64 = 1e-9;

/// The absolute tolerance of [`is_close`], as `pl.Expr.is_close`'s default.
pub const ABS_TOL: f64 = 0.0;

/// Whether two coordinate values are almost the same, as `pl.Expr.is_close` with its defaults:
/// `|a - b| <= max(REL_TOL * max(|a|, |b|), ABS_TOL)` (PEP 485).
/// Infinities are only close to themselves, and NaN is close to nothing.
pub fn is_close(a: f64, b: f64) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= (REL_TOL * a.abs().max(b.abs())).max(ABS_TOL)
}

/// Whether two [positions](ValidCoords::position) are [`is_close`] on every axis.
#[inline(always)]
pub fn same_position(a: [f64; 3], b: [f64; 3]) -> bool {
    is_close(a[0], b[0]) && is_close(a[1], b[1]) && is_close(a[2], b[2])
}

/// Coordinates that are all assumed to be valid.
/// Use for coordinates that **cannot** include invalid coordinates (nested geometries).
/// Otherwise, use [Coords].
///
/// See the README on views.
pub struct ValidCoords {
    x: Buffer<f64>,
    y: Buffer<f64>,
    /// `None` for a dimension without a `z`.
    z: Option<Buffer<f64>>,
    /// `None` for a dimension without an `m`.
    m: Option<Buffer<f64>>,
}

impl ValidCoords {
    pub fn new(coords: &dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let arrays = AxisArrays::new(coords)?;
        Ok(Self {
            x: arrays.x.values().clone(),
            y: arrays.y.values().clone(),
            z: arrays.z.map(|z| z.values().clone()),
            m: arrays.m.map(|m| m.values().clone()),
        })
    }

    /// How many coordinates there are.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    #[inline(always)]
    pub fn position<const Z: bool>(&self, i: usize) -> [f64; 3] {
        [
            self.x[i],
            self.y[i],
            if Z { self.z.as_ref().unwrap()[i] } else { 0.0 },
        ]
    }

    /// The m of coordinate `i`. Only for a dimension with an `m`.
    #[inline(always)]
    pub fn m(&self, i: usize) -> f64 {
        self.m.as_ref().unwrap()[i]
    }

    /// Every value of axis `n`, in storage order: x, y, then z and/or m.
    pub fn axis(&self, n: usize) -> &[f64] {
        let values = match n {
            0 => Some(&self.x),
            1 => Some(&self.y),
            2 => self.z.as_ref().or(self.m.as_ref()),
            3 => self.m.as_ref(),
            _ => None,
        };
        values.unwrap_or_else(|| panic!("coordinate index {n} out of range"))
    }

    /// Value `n` of coordinate `i`, in the order of [`axis`](Self::axis).
    #[inline]
    pub fn nth(&self, i: usize, n: usize) -> f64 {
        self.axis(n)[i]
    }

    /// Whether coordinates `i` and `j` lie at almost the same place:
    /// close in x, y and z. m is a measure rather than a position, so it is not compared.
    pub fn same_place(&self, i: usize, j: usize) -> bool {
        if self.z.is_none() {
            self.same_place_in::<false>(i, j)
        } else {
            self.same_place_in::<true>(i, j)
        }
    }

    /// [`same_place`](Self::same_place), comparing z only with `Z`.
    fn same_place_in<const Z: bool>(&self, i: usize, j: usize) -> bool {
        same_position(self.position::<Z>(i), self.position::<Z>(j))
    }

    pub fn is_ring(&self, range: Range<usize>) -> bool {
        range.len() >= MIN_RING_VERTICES && self.same_place(range.start, range.end - 1)
    }

    /// The dimension these coordinates carry.
    pub fn dim(&self) -> GeoDimension {
        match (self.z.is_some(), self.m.is_some()) {
            (false, false) => GeoDimension::XY,
            (true, false) => GeoDimension::XYZ,
            (false, true) => GeoDimension::XYM,
            (true, true) => GeoDimension::XYZM,
        }
    }
}

/// Coordinate data that might be potentially invalid.
/// Considers validity of individual components of the coordinates
/// ('x' may be valid, but 'y' may not be).
/// Use this in validation only.
pub struct AxisArrays<'a> {
    pub x: &'a PrimitiveArray<f64>,
    pub y: &'a PrimitiveArray<f64>,
    /// `None` for a dimension without a `z`.
    pub z: Option<&'a PrimitiveArray<f64>>,
    /// `None` for a dimension without an `m`.
    pub m: Option<&'a PrimitiveArray<f64>>,
}

impl<'a> AxisArrays<'a> {
    pub fn new(coords: &'a StructArray) -> PolarsResult<Self> {
        // The dtype fixes the field order, so x and y always come first.
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

/// Coordinates where individual coordinates can be invalid.
/// Use for coordinates that **can** include invalid coordinates (columns of Points).
/// Otherwise, use [ValidCoords].
///
/// See the README on views.
pub struct Coords {
    /// Which coordinates are there.
    validity: Option<Bitmap>,
    axes: ValidCoords,
}

impl Coords {
    pub fn new(coords: &dyn Array) -> PolarsResult<Self> {
        Ok(Self {
            validity: coords.validity().cloned(),
            axes: ValidCoords::new(coords)?,
        })
    }

    #[inline(always)]
    fn is_valid(&self, i: usize) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.get_bit(i))
    }

    /// The coordinates, for once it is known which are there.
    #[inline]
    pub fn ignoring_validity(&self) -> &ValidCoords {
        &self.axes
    }

    /// `None` when every coordinate is there.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Whether every coordinate is there.
    #[inline]
    pub fn is_all_valid(&self) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.unset_bits() == 0)
    }

    pub fn xy(&self, i: usize) -> Option<(f64, f64)> {
        self.xy_checking_nulls::<true>(i)
    }

    /// [`xy`](Self::xy), checking whether it is missing only with `NULLS`.
    /// Without, it assumes every coordinate is there.
    #[inline(always)]
    pub fn xy_checking_nulls<const NULLS: bool>(&self, i: usize) -> Option<(f64, f64)> {
        (!NULLS || self.is_valid(i)).then(|| (self.axes.x[i], self.axes.y[i]))
    }

    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    /// With `NULLS`, `None` if it is missing.
    /// Without, it assumes every coordinate is there.
    #[inline(always)]
    pub fn position<const NULLS: bool, const Z: bool>(&self, i: usize) -> Option<[f64; 3]> {
        (!NULLS || self.is_valid(i)).then(|| self.axes.position::<Z>(i))
    }

    /// See [`ValidCoords::len`].
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.axes.len()
    }

    /// See [`ValidCoords::nth`].
    #[inline]
    pub fn nth(&self, i: usize, n: usize) -> f64 {
        self.axes.nth(i, n)
    }

    /// See [`ValidCoords::same_place`].
    #[inline]
    pub fn same_place(&self, i: usize, j: usize) -> bool {
        self.axes.same_place(i, j)
    }

    /// See [`ValidCoords::is_ring`].
    #[inline]
    pub fn is_ring(&self, range: Range<usize>) -> bool {
        self.axes.is_ring(range)
    }

    /// See [`ValidCoords::dim`].
    #[inline]
    pub fn dim(&self) -> GeoDimension {
        self.axes.dim()
    }
}
