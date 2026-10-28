//! Linestrings and rings that an operation left too short,
//! and the rows they make null.

use geopolars_core::output::validity_without;
use geopolars_core::{List, ValidList};

/// See `too_short` of `remove_repeats_in_succession` in Python.
#[derive(Debug, Clone, Copy)]
pub enum TooShort {
    Null,
    Keep,
    Empty,
}

/// The list of `lists` that holds item `item`, `None` if none does.
pub(super) fn list_holding(lists: &ValidList, item: usize) -> Option<usize> {
    let offsets = lists.offsets().buffer();
    let i = offsets
        .partition_point(|&offset| offset as usize <= item)
        .checked_sub(1)?;
    (i < lists.len() && item < offsets[i + 1] as usize).then_some(i)
}

/// `rows`, with the rows in `failed` null too.
pub(super) fn without_rows(rows: &List, failed: &[usize]) -> List {
    let validity = validity_without(rows.validity(), rows.len(), failed);
    rows.clone().with_validity(validity)
}
