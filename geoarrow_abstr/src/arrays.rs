//! What a kernel builds its output in, whatever storage backs it.

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
    let values = (0..rows.len())
        // A missing row's offsets point anywhere, so it is not measured.
        .map(|i| {
            if rows.is_valid(i) {
                measure(i)
            } else {
                Ok(0.0)
            }
        })
        .collect::<Result<Vec<f64>, E>>()?;
    Ok(O::from_values(values, rows.validity()))
}
