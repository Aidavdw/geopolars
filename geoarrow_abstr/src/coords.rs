//! What a view over GeoArrow coordinates offers, whatever storage backs it.

use std::ops::Range;

use crate::GeoDimension;

/// What both [GeoArrowValidCoordsView] and [GeoArrowCoordsView] offer.
#[allow(clippy::len_without_is_empty)]
pub trait GeoArrowCoordsShared {
    /// How many coordinates there are.
    fn len(&self) -> usize;

    fn nth(&self, i: usize, n: usize) -> f64;

    /// Whether coordinates `i` and `j` lie at almost the same place:
    /// close in x, y and z. m is a measure rather than a position, so it is not compared.
    fn same_place(&self, i: usize, j: usize) -> bool;

    fn is_ring(&self, range: Range<usize>) -> bool;

    /// The dimension these coordinates carry.
    fn dim(&self) -> GeoDimension;
}

/// Coordinates that are all assumed to be valid.
/// Use for coordinates that **cannot** include invalid coordinates (columns of Points).
/// Otherwise, use [GeoArrowCoordsView].
pub trait GeoArrowValidCoordsView: GeoArrowCoordsShared {
    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    fn position<const Z: bool>(&self, i: usize) -> [f64; 3];

    /// The m of coordinate `i`. Only for a dimension with an `m`.
    fn m(&self, i: usize) -> f64;

    /// Every value of axis `n`, in storage order: x, y, then z and/or m
    /// (as [`nth`](GeoArrowCoordsShared::nth)).
    fn axis(&self, n: usize) -> &[f64];
}

/// Coordinates where individual coordinates can be invalid.
/// Use for coordinates that **can** include invalid coordinates (columns of Points).
/// Otherwise, use [GeoArrowValidCoordsView].
pub trait GeoArrowCoordsView: GeoArrowCoordsShared {
    /// How the storage records which coordinates are there.
    type Validity;

    /// `None` when every coordinate is there.
    fn validity(&self) -> Option<&Self::Validity>;

    fn xy(&self, i: usize) -> Option<(f64, f64)>;

    /// [`xy`](Self::xy), checking whether it is missing only with `NULLS`.
    /// Without, it assumes every coordinate is there.
    fn xy_checking_nulls<const NULLS: bool>(&self, i: usize) -> Option<(f64, f64)>;

    /// The position of coordinate `i`: x, y, and with `Z` its z (`0.0` without).
    /// With `NULLS`, `None` if it is missing.
    /// Without, it assumes every coordinate is there.
    fn position<const NULLS: bool, const Z: bool>(&self, i: usize) -> Option<[f64; 3]>;
}
