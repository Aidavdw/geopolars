//! What a kernel builds its output in, whatever storage backs it.

use crate::GeoDimension;

/// Every value of the x, y, z and m axes, in coordinate order.
/// z and m are `None` for a dimension without them.
pub type OwnedAxes = (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>);

/// A column of polygons: rows of rings of coordinates, built from their buffers.
pub trait GeoArrowPolygonOutput {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `axes` holds every coordinate of `dim`.
    /// `rings` and `rows` are offsets, with one more entry than there are rings or rows.
    /// A row is missing where `validity` says, and at the rows in `failed`.
    fn from_buffers(
        dim: GeoDimension,
        axes: OwnedAxes,
        rings: Vec<i64>,
        rows: Vec<i64>,
        validity: Option<&Self::Validity>,
        failed: &[usize],
    ) -> Self;
}

/// One `f64` per row, with missing rows recorded in the storage's own validity.
pub trait GeoArrowF64Array {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `values` has one entry per row. `validity` is `None` when every row is there.
    fn from_values(values: Vec<f64>, validity: Option<&Self::Validity>) -> Self;

    /// `values` has one entry per row.
    /// A row is missing where `a` or `b` is, and at the rows in `failed`.
    fn from_values_and(
        values: Vec<f64>,
        a: Option<&Self::Validity>,
        b: Option<&Self::Validity>,
        failed: &[usize],
    ) -> Self;
}

/// `measure` of every row of a chunk, with the rows' own validity.
#[cfg(feature = "native")]
#[inline(always)]
pub(crate) fn per_row<O: GeoArrowF64Array, E>(
    rows: &impl crate::GeoArrowListView<Validity = O::Validity>,
    measure: impl Fn(usize) -> Result<f64, E>,
) -> Result<O, E> {
    if rows.is_all_valid() {
        per_row_with::<true, O, E>(rows, measure)
    } else {
        per_row_with::<false, O, E>(rows, measure)
    }
}

/// If `ALL_VALID=true`, we don't have to individually check
/// each row for validity.
#[cfg(feature = "native")]
#[inline(always)]
fn per_row_with<const ALL_VALID: bool, O: GeoArrowF64Array, E>(
    rows: &impl crate::GeoArrowListView<Validity = O::Validity>,
    measure: impl Fn(usize) -> Result<f64, E>,
) -> Result<O, E> {
    let values = (0..rows.len())
        // A missing row's offsets point anywhere, so it is not measured.
        .map(|i| {
            if ALL_VALID || rows.is_valid(i) {
                measure(i)
            } else {
                Ok(0.0)
            }
        })
        .collect::<Result<Vec<f64>, E>>()?;
    Ok(O::from_values(values, rows.validity()))
}
