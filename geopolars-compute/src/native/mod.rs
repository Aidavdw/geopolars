//! Kernels of our own, working straight off the GeoArrow buffers.

mod affine;
mod area;
mod centroid;
pub mod distance;
mod is_ccw;
mod is_close;
mod is_ring;
// Only for `validity` and `degeneracy`.
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
pub use is_close::{
    line_strings_are_close, multi_line_strings_are_close, multi_points_are_close,
    multi_polygons_are_close, points_are_close, polygons_are_close,
};
pub use is_ring::{line_strings_are_rings, multi_line_strings_are_rings};
pub use is_valid::Bounds;
pub(crate) use is_valid::{
    coordinate_is_valid, line_string_shape_is_valid, polygon_shape_is_valid,
};
// Only for `degeneracy`.
#[cfg(feature = "geos")]
pub(crate) use is_valid::is_empty_point;
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};
pub use rings::{
    close_open_rings, multi_polygon_exteriors, multi_polygon_interiors, polygon_exteriors,
    polygon_interiors,
};
