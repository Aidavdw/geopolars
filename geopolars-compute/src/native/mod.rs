//! Kernels of our own, measuring straight off the GeoArrow buffers.

mod area;
pub mod distance;
mod is_ccw;
mod length;

pub use area::{multi_polygon_areas, polygon_areas, AreaMetric};
pub use distance::{against_one, pairwise, DistanceMetric};
pub use is_ccw::{
    line_strings_are_ccw, multi_line_strings_are_ccw, multi_polygons_are_ccw, polygons_are_ccw,
};
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};

use geopolars_core::output::{bool_array, f64_array};
use geopolars_core::List;
use polars_arrow::array::{BooleanArray, PrimitiveArray};

/// `measure` of every row of a chunk, with the rows' own validity.
#[inline(always)]
pub(super) fn per_row<E>(
    rows: &List,
    measure: impl Fn(usize) -> Result<f64, E>,
) -> Result<PrimitiveArray<f64>, E> {
    Ok(f64_array(per_row_values(rows, measure)?, rows.validity()))
}

/// [per_row], for a `bool` per row.
#[inline(always)]
pub(super) fn per_row_bool(rows: &List, test: impl Fn(usize) -> bool) -> BooleanArray {
    let Ok(values) = per_row_values(rows, |i| Ok::<_, std::convert::Infallible>(test(i)));
    bool_array(values, rows.validity())
}

/// `measure` of every row of a chunk, with a default value for a missing row.
#[inline(always)]
fn per_row_values<T: Default, E>(
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
