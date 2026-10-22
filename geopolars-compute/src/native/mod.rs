//! Kernels of our own, working straight off the GeoArrow buffers.

mod affine;
mod area;
mod centroid;
pub mod distance;
mod is_ccw;
mod is_ring;
// Only for `validity_checking`.
#[cfg(feature = "rsgeo")]
mod is_valid;
mod length;
mod rings;

pub use affine::{
    affine_line_strings, affine_multi_line_strings, affine_multi_points, affine_multi_polygons,
    affine_points, affine_polygons, AffineMatrix, RowShift,
};
pub use area::{multi_polygon_areas, polygon_areas, AreaMetric};
pub use centroid::{
    line_string_centroids, multi_line_string_centroids, multi_polygon_centroids, polygon_centroids,
};
pub use distance::{against_one, pairwise, DistanceMetric};
pub use is_ccw::{
    line_strings_are_ccw, multi_line_strings_are_ccw, multi_polygons_are_ccw, polygons_are_ccw,
};
pub use is_ring::{line_strings_are_rings, multi_line_strings_are_rings};
#[cfg(feature = "rsgeo")]
pub use is_valid::Bounds;
#[cfg(feature = "rsgeo")]
pub(crate) use is_valid::{
    coordinates_are_valid, line_string_shape_is_valid, point_is_valid, polygon_shape_is_valid,
};
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};
pub use rings::{
    close_open_rings, multi_polygon_exteriors, multi_polygon_interiors, polygon_exteriors,
    polygon_interiors,
};
