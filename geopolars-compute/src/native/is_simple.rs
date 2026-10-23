//! Whether a point or multipoint is simple.
//! See `is_simple` in Python for the rules.

use geopolars_core::{MultiPointArray, PointArray, Tolerance};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use super::has_repeated_points::has_repeated_points;
use crate::per_row::per_row_bool;

/// Every point is simple, so this is `true` for every point that is there.
pub fn points_are_simple(array: &PointArray) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::new_with_value(true, coords.ignoring_validity().len());
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn multi_points_are_simple(array: &MultiPointArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        !has_repeated_points::<false, false>(
            &array.coords,
            array.rows.range(i),
            Tolerance::default(),
        )
    })
}
