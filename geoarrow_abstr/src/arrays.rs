//! What a kernel builds its output in, whatever storage backs it.

use crate::GeoDimension;

/// Every value of the x, y, z and m axes, in coordinate order.
/// z and m are `None` for a dimension without them.
pub type OwnedAxes = (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>);

/// A column of linestrings
pub trait GeoArrowLineStringOutput {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `axes` holds every coordinate of `dim`.
    /// `rows` are offsets, with one more entry than there are rows.
    /// A row is missing where `validity` says, and at the rows in `failed`.
    fn from_buffers(
        dim: GeoDimension,
        axes: OwnedAxes,
        rows: Vec<i64>,
        validity: Option<&Self::Validity>,
        failed: &[usize],
    ) -> Self;
}

/// A column of multilinestrings
pub trait GeoArrowMultiLineStringOutput {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `axes` holds every coordinate of `dim`.
    /// `lines` and `rows` are offsets, with one more entry than there are linestrings or rows.
    /// A row is missing where `validity` says, and at the rows in `failed`.
    fn from_buffers(
        dim: GeoDimension,
        axes: OwnedAxes,
        lines: Vec<i64>,
        rows: Vec<i64>,
        validity: Option<&Self::Validity>,
        failed: &[usize],
    ) -> Self;
}

/// A column of polygons
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

/// A column of multipolygons
pub trait GeoArrowMultiPolygonOutput {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `axes` holds every coordinate of `dim`.
    /// `rings`, `polygons` and `rows` are offsets,
    /// with one more entry than there are rings, polygons or rows.
    /// A row is missing where `validity` says, and at the rows in `failed`.
    fn from_buffers(
        dim: GeoDimension,
        axes: OwnedAxes,
        rings: Vec<i64>,
        polygons: Vec<i64>,
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

/// One `bool` per row, with missing rows recorded in the storage's own validity.
pub trait GeoArrowBoolArray {
    /// How the storage records which rows are there.
    type Validity;

    /// `values` has one entry per row. `validity` is `None` when every row is there.
    fn from_values(values: Vec<bool>, validity: Option<&Self::Validity>) -> Self;
}

/// A list of `bool`s per row, with missing rows recorded in the storage's own validity.
pub trait GeoArrowBoolListArray {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `rows` are offsets into `values`, with one more entry than there are rows.
    /// A row is missing where `validity` says.
    fn from_buffers(values: Vec<bool>, rows: Vec<i64>, validity: Option<&Self::Validity>) -> Self;
}

/// `measure` of every row of a chunk, with the rows' own validity.
#[cfg(feature = "native")]
#[inline(always)]
pub(crate) fn per_row<O: GeoArrowF64Array, E>(
    rows: &impl crate::GeoArrowListView<Validity = O::Validity>,
    measure: impl Fn(usize) -> Result<f64, E>,
) -> Result<O, E> {
    Ok(O::from_values(
        per_row_values(rows, measure)?,
        rows.validity(),
    ))
}

/// [per_row], for a `bool` per row.
#[cfg(feature = "native")]
#[inline(always)]
pub(crate) fn per_row_bool<O: GeoArrowBoolArray>(
    rows: &impl crate::GeoArrowListView<Validity = O::Validity>,
    test: impl Fn(usize) -> bool,
) -> O {
    let Ok(values) = per_row_values(rows, |i| Ok::<_, std::convert::Infallible>(test(i)));
    O::from_values(values, rows.validity())
}

/// `measure` of every row of a chunk, with a default value for a missing row.
#[cfg(feature = "native")]
#[inline(always)]
fn per_row_values<T: Default, E>(
    rows: &impl crate::GeoArrowListView,
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
#[cfg(feature = "native")]
#[inline(always)]
fn per_row_values_with<const ALL_VALID: bool, T: Default, E>(
    rows: &impl crate::GeoArrowListView,
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
