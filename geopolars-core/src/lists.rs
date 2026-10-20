//! Views over the list offsets of a geometry column.

use std::ops::Range;

use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::OffsetsBuffer;

/// Lists that are all assumed to be valid:
/// the inner levels of a nested geometry (the linestrings of a multilinestring).
/// Otherwise, use [List].
///
/// See the README on views.
pub struct ValidList {
    offsets: OffsetsBuffer<i64>,
}

impl ValidList {
    pub fn new(list: &ListArray<i64>) -> Self {
        Self {
            offsets: list.offsets().clone(),
        }
    }

    pub fn from_offsets(offsets: OffsetsBuffer<i64>) -> Self {
        Self { offsets }
    }

    pub(crate) fn into_offsets(self) -> OffsetsBuffer<i64> {
        self.offsets
    }

    /// How many lists there are.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    /// Where list `i`'s items sit in the child array.
    #[inline(always)]
    pub fn range(&self, i: usize) -> Range<usize> {
        let (start, end) = self.offsets.start_end(i);
        start..end
    }
}

/// Lists where individual lists can be missing: the outer level of a geometry column.
/// Otherwise, use [ValidList].
///
/// See the README on views.
pub struct List {
    /// Which lists are there.
    validity: Option<Bitmap>,
    lists: ValidList,
}

impl List {
    pub fn new(list: &ListArray<i64>) -> Self {
        Self {
            validity: list.validity().cloned(),
            lists: ValidList::new(list),
        }
    }

    /// `validity` is `None` when every list is there.
    pub fn from_offsets(offsets: OffsetsBuffer<i64>, validity: Option<Bitmap>) -> Self {
        Self {
            validity,
            lists: ValidList::from_offsets(offsets),
        }
    }

    pub(crate) fn into_parts(self) -> (OffsetsBuffer<i64>, Option<Bitmap>) {
        (self.lists.into_offsets(), self.validity)
    }

    /// See [`ValidList::len`].
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.lists.len()
    }

    /// See [`ValidList::range`].
    #[inline(always)]
    pub fn range(&self, i: usize) -> Range<usize> {
        self.lists.range(i)
    }

    #[inline(always)]
    pub fn is_valid(&self, i: usize) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.get_bit(i))
    }

    /// `None` when every list is there.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// Whether every list is there.
    #[inline]
    pub fn is_all_valid(&self) -> bool {
        self.validity
            .as_ref()
            .is_none_or(|validity| validity.unset_bits() == 0)
    }
}
