//! Abstraction traits for accessing GeoArrow coordinates.
//! Extends geo-traits.
//! Also holds the kernels written against those traits, so they work whatever storage backs them.
//!
//! See the expression implementation guide on Array abstractions for GeoArrow.

mod arrays;
mod coords;
mod dimension;
pub mod distance;
mod length;
mod lists;
mod metric;

pub use arrays::GeoArrowF64Array;
pub use coords::{GeoArrowCoordsShared, GeoArrowCoordsView, GeoArrowValidCoordsView};
pub use dimension::GeoDimension;
pub use length::lengths;
pub use lists::{GeoArrowListShared, GeoArrowListView, GeoArrowValidListView};
pub use metric::{GeodesicMetric, Metric, Planar, ReprojectError, WithHeight};
