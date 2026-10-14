//! What a view over GeoArrow list offsets offers, whatever storage backs it.

use std::ops::Range;

/// What both [GeoArrowValidListView] and [GeoArrowListView] offer.
#[allow(clippy::len_without_is_empty)]
pub trait GeoArrowListShared {
    /// How many lists there are.
    fn len(&self) -> usize;

    /// Where list `i`'s items sit in the child array.
    fn range(&self, i: usize) -> Range<usize>;
}

/// Lists that are all assumed to be valid: the inner levels of a nested geometry.
/// Otherwise, use [GeoArrowListView].
pub trait GeoArrowValidListView: GeoArrowListShared {}

/// Lists where individual lists can be missing: the outer level of a geometry column.
/// Otherwise, use [GeoArrowValidListView].
pub trait GeoArrowListView: GeoArrowListShared {
    /// How the storage records which lists are there.
    type Validity;

    fn is_valid(&self, i: usize) -> bool;

    /// `None` when every list is there.
    fn validity(&self) -> Option<&Self::Validity>;
}
