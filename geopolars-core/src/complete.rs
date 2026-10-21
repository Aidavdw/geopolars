//! Finding the rows of unchecked storage that hold to GeoArrow's rule on missing values:
//! nulls only at the outermost level.
//! This is the one place that reads validity below the rows.

use polars_arrow::array::{Array, ListArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_core::prelude::PolarsResult;

use crate::{downcast, AxisArrays};

/// The validity of the rows of `chunk`, also missing where a part or coordinate
/// (or one of its axes) below the row is missing.
/// `nesting` is the number of list levels above the coordinates.
/// `None` when every row is there.
pub fn complete_rows(chunk: &dyn Array, nesting: u8) -> PolarsResult<Option<Bitmap>> {
    if nesting == 0 {
        let coords: &StructArray = downcast(chunk, "a coordinate struct")?;
        let axes = AxisArrays::new(coords)?;
        // Polars also nulls every axis under a missing coordinate, which changes nothing here.
        let validity = [Some(axes.x), Some(axes.y), axes.z, axes.m]
            .into_iter()
            .flatten()
            .fold(coords.validity().cloned(), |validity, axis| {
                combine_validities_and(validity.as_ref(), axis.validity())
            });
        return Ok(validity);
    }
    let lists: &ListArray<i64> = downcast(chunk, "a list of parts")?;
    let Some(parts) = complete_rows(lists.values().as_ref(), nesting - 1)? else {
        return Ok(lists.validity().cloned());
    };
    let offsets = lists.offsets();
    let validity = Bitmap::from_iter((0..lists.len()).map(|i| {
        let (start, end) = offsets.start_end(i);
        lists.is_valid(i) && parts.null_count_range(start, end - start) == 0
    }));
    Ok(Some(validity))
}
