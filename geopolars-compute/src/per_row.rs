//! Running a kernel over every row of a chunk.

use geopolars_core::output::{bool_array, f64_array};
use geopolars_core::List;
use polars_arrow::array::{BooleanArray, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;

/// `measure` of every row of a chunk, with the rows' own validity.
#[inline(always)]
pub(crate) fn per_row<E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<f64, E>,
) -> Result<PrimitiveArray<f64>, E> {
    Ok(f64_array(per_row_values(rows, measure)?, rows.validity()))
}

/// [per_row], for a `bool` per row.
#[inline(always)]
pub(crate) fn try_per_row_bool<E>(
    rows: &List,
    test: impl Fn(usize) -> Result<bool, E>,
) -> Result<BooleanArray, E> {
    Ok(bool_array(per_row_values(rows, test)?, rows.validity()))
}

#[inline(always)]
pub(crate) fn per_row_bool(rows: &List, test: impl Fn(usize) -> bool) -> BooleanArray {
    let Ok(tested) = try_per_row_bool(rows, |i| Ok::<_, std::convert::Infallible>(test(i)));
    tested
}

/// A `bool` per item, missing where it is `None`.
pub(crate) fn optional_bool_array(tested: &[Option<bool>]) -> BooleanArray {
    let values = tested
        .iter()
        .map(|tested| tested.unwrap_or(false))
        .collect();
    let validity = Bitmap::from_iter(tested.iter().map(Option::is_some));
    bool_array(values, Some(&validity))
}

/// `test` of every row of a chunk, missing where the row is or where `test` gives `None`.
#[inline(always)]
pub(crate) fn per_row_optional_bool<E>(
    rows: &List,
    test: impl Fn(usize) -> Result<Option<bool>, E>,
) -> Result<BooleanArray, E> {
    Ok(optional_bool_array(&per_row_values(rows, test)?))
}

/// `measure` of every row of a chunk, with a default value for a missing row.
#[inline(always)]
pub(crate) fn per_row_values<T: Default, E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<T, E>,
) -> Result<Vec<T>, E> {
    if rows.is_all_valid() {
        per_row_values_with::<true, T, E>(rows, measure)
    } else {
        per_row_values_with::<false, T, E>(rows, measure)
    }
}

/// If `ALL_VALID=true`, we don't have to individually check
/// each row for validity.
#[inline(always)]
fn per_row_values_with<const ALL_VALID: bool, T: Default, E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<T, E>,
) -> Result<Vec<T>, E> {
    (0..rows.len())
        // A missing row's offsets point anywhere, so it is not measured.
        .map(|i| {
            if ALL_VALID || rows.is_valid(i) {
                measure(i)
            } else {
                Ok(T::default())
            }
        })
        .collect()
}
