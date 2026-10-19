//! The arrays kernels build their output in.
//! A row is missing where the given validity says, and at the rows in `failed`.

use polars_arrow::array::{Array, BooleanArray, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::compute::utils::{combine_validities_and, combine_validities_and3};
use polars_arrow::offset::{Offsets, OffsetsBuffer};
use polars_core::prelude::CompatLevel;

use crate::GeoDimension;

/// Every value of the x, y, z and m axes, in coordinate order.
/// z and m are `None` for a dimension without them.
pub type OwnedAxes = (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>);

/// `values` has one entry per row. `validity` is `None` when every row is there.
pub fn f64_array(values: Vec<f64>, validity: Option<&Bitmap>) -> PrimitiveArray<f64> {
    PrimitiveArray::from_vec(values).with_validity(validity.cloned())
}

/// [f64_array], with a row missing where `a` or `b` is, and at the rows in `failed`.
pub fn f64_array_masked(
    values: Vec<f64>,
    a: Option<&Bitmap>,
    b: Option<&Bitmap>,
    failed: &[usize],
) -> PrimitiveArray<f64> {
    let measured = succeeded(values.len(), failed);
    let validity = combine_validities_and3(a, b, measured.as_ref());
    PrimitiveArray::from_vec(values).with_validity(validity)
}

/// [f64_array], for a `bool` per row.
pub fn bool_array(values: Vec<bool>, validity: Option<&Bitmap>) -> BooleanArray {
    BooleanArray::from_slice(values).with_validity(validity.cloned())
}

/// `axes` holds every coordinate of `dim`.
/// `rows` are offsets, with one more entry than there are rows.
pub fn line_string_array(
    dim: GeoDimension,
    axes: OwnedAxes,
    rows: Vec<i64>,
    validity: Option<&Bitmap>,
    failed: &[usize],
) -> ListArray<i64> {
    wrap_in_list(coords_array(dim, axes).boxed(), rows, validity, failed)
}

/// See [line_string_array].
/// `lines` are offsets too, with one more entry than there are linestrings.
pub fn multi_line_string_array(
    dim: GeoDimension,
    axes: OwnedAxes,
    lines: Vec<i64>,
    rows: Vec<i64>,
    validity: Option<&Bitmap>,
    failed: &[usize],
) -> ListArray<i64> {
    let lines = line_string_array(dim, axes, lines, None, &[]);
    wrap_in_list(lines.boxed(), rows, validity, failed)
}

/// See [line_string_array].
/// `rings` are offsets too, with one more entry than there are rings.
pub fn polygon_array(
    dim: GeoDimension,
    axes: OwnedAxes,
    rings: Vec<i64>,
    rows: Vec<i64>,
    validity: Option<&Bitmap>,
    failed: &[usize],
) -> ListArray<i64> {
    // A ring is stored like a linestring.
    let rings = line_string_array(dim, axes, rings, None, &[]);
    wrap_in_list(rings.boxed(), rows, validity, failed)
}

/// See [polygon_array].
/// `polygons` are offsets too, with one more entry than there are polygons.
pub fn multi_polygon_array(
    dim: GeoDimension,
    axes: OwnedAxes,
    rings: Vec<i64>,
    polygons: Vec<i64>,
    rows: Vec<i64>,
    validity: Option<&Bitmap>,
    failed: &[usize],
) -> ListArray<i64> {
    let polygons = polygon_array(dim, axes, rings, polygons, None, &[]);
    wrap_in_list(polygons.boxed(), rows, validity, failed)
}

/// The coordinate struct of `dim`, taking over the `axes` buffers.
fn coords_array(dim: GeoDimension, (x, y, z, m): OwnedAxes) -> StructArray {
    let len = x.len();
    let axes = [Some(x), Some(y), z, m]
        .into_iter()
        .flatten()
        .map(|values| PrimitiveArray::from_vec(values).boxed())
        .collect();
    StructArray::new(
        dim.coordinates().to_arrow(CompatLevel::newest()),
        len,
        axes,
        None,
    )
}

/// `values` split into lists at `offsets`.
fn wrap_in_list(
    values: Box<dyn Array>,
    offsets: Vec<i64>,
    validity: Option<&Bitmap>,
    failed: &[usize],
) -> ListArray<i64> {
    // SAFETY: the kernel builds the offsets in order, starting at 0, ending at the child's length.
    let offsets: OffsetsBuffer<i64> = unsafe { Offsets::new_unchecked(offsets) }.into();
    let built = succeeded(offsets.len_proxy(), failed);
    ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(values.dtype().clone()),
        offsets,
        values,
        combine_validities_and(validity, built.as_ref()),
    )
}

/// Every one of `len` rows but those in `failed`.
/// `None` when nothing failed.
fn succeeded(len: usize, failed: &[usize]) -> Option<Bitmap> {
    (!failed.is_empty()).then(|| {
        let mut succeeded = MutableBitmap::from_len_set(len);
        for &i in failed {
            succeeded.set(i, false);
        }
        Bitmap::from(succeeded)
    })
}
