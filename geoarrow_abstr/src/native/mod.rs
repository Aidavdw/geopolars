//! Kernels of our own, measuring straight off the GeoArrow buffers.

mod area;
pub mod distance;
mod length;

pub use area::{multi_polygon_areas, polygon_areas, AreaMetric};
pub use distance::{against_one, pairwise, DistanceMetric};
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};
