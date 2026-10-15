//! Abstraction traits for accessing GeoArrow coordinates.
//! Extends geo-traits.
//! Also holds the kernels written against those traits, so they work whatever storage backs them.
//!
//! See the expression implementation guide on Array abstractions for GeoArrow.

mod area;
mod arrays;
mod coords;
mod dimension;
mod distance;
mod geos;
mod length;
mod linestrings;
mod lists;
mod metric;
mod polygons;

pub use area::{multi_polygon_areas, polygon_areas, AreaMetric};
pub use arrays::GeoArrowF64Array;
pub use coords::{GeoArrowCoordsShared, GeoArrowCoordsView, GeoArrowValidCoordsView};
pub use dimension::GeoDimension;
pub use distance::{against_one, pairwise, DistanceMetric};
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};
pub use linestrings::{LineString, LineStringArray, MultiLineString, MultiLineStringArray};
pub use lists::{GeoArrowListShared, GeoArrowListView, GeoArrowValidListView};
pub use metric::{BaseMetric, GeodesicMetric, Planar, ReprojectError, WithHeight};
pub use polygons::{MultiPolygon, MultiPolygonArray, Polygon, PolygonArray};
