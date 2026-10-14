//! What a kernel builds its output in, whatever storage backs it.

/// One `f64` per row, with missing rows recorded in the storage's own validity.
pub trait GeoArrowF64Array {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `values` has one entry per row. `validity` is `None` when every row is there.
    fn from_values(values: Vec<f64>, validity: Option<&Self::Validity>) -> Self;

    /// Fill up a new one from a fallible function.
    /// On an error, with `STOP_ON_ERROR` it stops and returns it;
    /// without, it marks that row missing and carries on, so it never fails.
    fn try_from_options<const STOP_ON_ERROR: bool, E>(
        values: impl ExactSizeIterator<Item = Result<Option<f64>, E>>,
    ) -> Result<Self, E>
    where
        Self: Sized;
}
