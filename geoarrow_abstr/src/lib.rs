//! Abstraction traits for accessing GeoArrow coordinates.
//! Extends geo-traits.
//! Also holds the kernels written against those traits, so they work whatever storage backs them:
//! [native] ones of our own, and [geos] ones forwarding to GEOS, each behind its feature.
//!
//! See the expression implementation guide on Array abstractions for GeoArrow.

mod arrays;
mod coords;
mod dimension;
#[cfg(feature = "geos")]
pub mod geos;
mod linestrings;
mod lists;
mod metric;
#[cfg(feature = "native")]
pub mod native;
mod points;
mod polygons;

pub use arrays::GeoArrowF64Array;
pub use coords::{GeoArrowCoordsShared, GeoArrowCoordsView, GeoArrowValidCoordsView};
pub use dimension::GeoDimension;
pub use linestrings::{LineString, LineStringArray, MultiLineString, MultiLineStringArray};
pub use lists::{GeoArrowListShared, GeoArrowListView, GeoArrowValidListView};
pub use metric::{BaseMetric, GeodesicMetric, Planar, ReprojectError, WithHeight};
pub use points::{Point, PointArray};
pub use polygons::{MultiPolygon, MultiPolygonArray, Polygon, PolygonArray};
