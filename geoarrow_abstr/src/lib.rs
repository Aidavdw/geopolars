//! Abstraction traits for accessing GeoArrow coordinates.
//! Extends geo-traits.
//!
//! See the expression implementation guide on Array abstractions for GeoArrow.

mod coords;
mod dimension;

pub use coords::{GeoArrowCoordsShared, GeoArrowCoordsView, GeoArrowValidCoordsView};
pub use dimension::GeoDimension;
