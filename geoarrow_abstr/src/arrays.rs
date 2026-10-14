//! What a kernel builds its output in, whatever storage backs it.

/// One `f64` per row, with missing rows recorded in the storage's own validity.
pub trait GeoArrowF64Array {
    /// How the storage records which rows are there.
    /// The same as the rows' [`GeoArrowListView::Validity`](crate::GeoArrowListView::Validity).
    type Validity;

    /// `values` has one entry per row. `validity` is `None` when every row is there.
    fn from_values(values: Vec<f64>, validity: Option<&Self::Validity>) -> Self;
}
