//! Whether a point or multipoint is simple.
//! See `is_simple` in Python for the rules.

use std::ops::Range;

use geopolars_core::{MultiPointArray, PointArray, Tolerance, ValidCoords};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use super::has_repeats_in_succession::coincide;
use crate::per_row::per_row_bool;

// TODO: turn this into its own expression (whether all points of a geometry are unique).
/// Whether any two of `points` [coincide].
#[inline(always)]
fn has_coinciding_points<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    coords: &ValidCoords,
    points: Range<usize>,
    tolerance: Tolerance,
) -> bool {
    points.clone().any(|i| {
        (i + 1..points.end)
            .any(|j| coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(coords, i, j, tolerance))
    })
}

/// Every point is simple, so this is `true` for every point that is there.
pub fn points_are_simple(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::new_with_value(true, coords.ignoring_validity().len());
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn multi_points_are_simple(array: &MultiPointArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        !has_coinciding_points::<false, false, false>(
            &array.coords,
            array.rows.range(i),
            Tolerance::default(),
        )
    })
}
