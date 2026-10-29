//! A true (`pass`) or false check for geometries with levels.
//!
//! A check writes a mask for the level it looks at, such as the coordinates, the rings or the rows.
//! A set bit passes, as in a validity bitmap.
//! `None` means every item at that level passes.
//! [rows_passing] folds the masks up to the rows.
//! A list passes when it passes itself, and every item in it does.

#[cfg(feature = "native")]
use geopolars_core::List;
use geopolars_core::ValidList;
#[cfg(feature = "native")]
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
#[cfg(feature = "native")]
use polars_arrow::datatypes::ArrowDataType;

fn unless_all_set(passing: Option<Bitmap>) -> Option<Bitmap> {
    passing.filter(|passing| passing.unset_bits() > 0)
}

/// Which of `lists` hold only items set in `items`.
fn lists_passing(lists: &ValidList, items: &Bitmap) -> Bitmap {
    Bitmap::from_iter(
        lists
            .ranges()
            .map(|range| items.null_count_range(range.start, range.len()) == 0),
    )
}

/// The rows passing every mask in `passing`.
/// `lists` are the list levels, outermost first.
/// `passing` holds a mask per list level, in the same order.
/// Its last mask is for the coordinates.
pub(crate) fn rows_passing(lists: &[&ValidList], passing: Vec<Option<Bitmap>>) -> Option<Bitmap> {
    assert_eq!(passing.len(), lists.len() + 1, "a mask per level");
    let mut levels = passing.into_iter().rev();
    let coords = unless_all_set(levels.next().flatten());
    lists
        .iter()
        .rev()
        .zip(levels)
        .fold(coords, |items, (lists, own)| {
            let parts = items.map(|items| lists_passing(lists, &items));
            unless_all_set(combine_validities_and(own.as_ref(), parts.as_ref()))
        })
}

/// The valid rows of `rows` whose coordinates all pass `coords` (see [rows_passing]).
/// `inner_lists` are the list levels below the rows, outermost first.
#[cfg(feature = "native")]
pub(crate) fn valid_rows_passing(
    rows: &List,
    inner_lists: &[&ValidList],
    coords: Option<Bitmap>,
) -> Option<Bitmap> {
    let lists: Vec<&ValidList> = [rows.ignoring_validity()]
        .into_iter()
        .chain(inner_lists.iter().copied())
        .collect();
    let passing = [rows.validity().cloned()]
        .into_iter()
        .chain(inner_lists.iter().map(|_| None))
        .chain([coords])
        .collect();
    rows_passing(&lists, passing)
}

/// A `bool` per row, `true` where the row is set in `passing` (see [rows_passing]).
/// `validity` is the validity of the rows.
/// A row that is null in `validity` has to be unset in `passing`.
#[cfg(feature = "native")]
pub(crate) fn passing_bool_array(
    len: usize,
    validity: Option<&Bitmap>,
    passing: Option<Bitmap>,
) -> BooleanArray {
    let values = passing.unwrap_or_else(|| Bitmap::new_with_value(true, len));
    BooleanArray::new(ArrowDataType::Boolean, values, validity.cloned())
}
