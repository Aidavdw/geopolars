//! Kernels of our own, working straight off the GeoArrow buffers.

mod affine;
mod area;
mod bounds;
mod centroid;
mod count_coordinates;
pub mod distance;
mod filter_coordinates;
mod fix_out_of_bounds;
mod has_repeated_points;
mod is_ccw;
mod is_close;
mod is_empty;
mod is_ring;
mod is_simple;
// For `geoarrow_validity`, `ogc_validity` and `fix_out_of_bounds`.
mod is_valid;
mod length;
mod remove_repeated_points;
mod reverse;
mod rings;
mod selected_coords;
mod too_short;

pub use affine::{
    affine_line_strings, affine_multi_line_strings, affine_multi_points, affine_multi_polygons,
    affine_points, affine_polygons, AffineMatrix, RowShift,
};
pub use area::{multi_polygon_areas, polygon_areas, AreaMetric};
pub use bounds::{
    box_array, line_string_bounds, line_string_column_extent, multi_line_string_bounds,
    multi_line_string_column_extent, multi_line_string_part_bounds, multi_point_bounds,
    multi_point_column_extent, multi_point_part_bounds, multi_polygon_bounds,
    multi_polygon_column_extent, multi_polygon_part_bounds, point_bounds, point_column_extent,
    polygon_bounds, polygon_column_extent, BoxOptions, Extent,
};
pub use centroid::{
    line_string_centroids, multi_line_string_centroids, multi_polygon_centroids, polygon_centroids,
};
pub use count_coordinates::{
    line_string_coordinate_counts, multi_line_string_coordinate_counts,
    multi_point_coordinate_counts, multi_polygon_coordinate_counts, point_coordinate_counts,
    polygon_coordinate_counts,
};
pub use distance::{against_one, pairwise, DistanceMetric};
pub use filter_coordinates::{
    line_strings_with_filtered_coordinates, multi_line_strings_with_filtered_coordinates,
    multi_points_with_filtered_coordinates, multi_polygons_with_filtered_coordinates,
    points_with_filtered_coordinates, polygons_with_filtered_coordinates, NonFinite,
};
pub use fix_out_of_bounds::{
    line_strings_with_fixed_out_of_bounds, multi_line_strings_with_fixed_out_of_bounds,
    multi_points_with_fixed_out_of_bounds, multi_polygons_with_fixed_out_of_bounds,
    points_with_fixed_out_of_bounds, polygons_with_fixed_out_of_bounds, OutOfBounds,
    OutOfBoundsAction,
};
pub use has_repeated_points::{
    line_strings_have_repeated_points, multi_line_strings_have_repeated_points,
    multi_points_have_repeated_points, multi_polygons_have_repeated_points,
    points_have_repeated_points, polygons_have_repeated_points,
};
pub use is_ccw::{
    line_strings_are_ccw, line_strings_to_ccw, multi_line_strings_are_ccw,
    multi_line_strings_to_ccw, multi_polygons_are_ccw, multi_polygons_to_ccw, polygons_are_ccw,
    polygons_to_ccw,
};
pub use is_close::{
    line_strings_are_close, multi_line_strings_are_close, multi_points_are_close,
    multi_polygons_are_close, points_are_close, polygons_are_close,
};
pub use is_empty::{
    line_strings_are_empty, multi_line_strings_are_empty, multi_points_are_empty,
    multi_polygons_are_empty, points_are_empty, polygons_are_empty,
};
pub use is_ring::{
    line_strings_are_rings, multi_line_strings_are_rings, multi_polygons_have_only_rings,
    polygons_have_only_rings,
};
pub use is_simple::{multi_points_are_simple, points_are_simple};
pub(crate) use is_valid::{all_coordinates_valid, coordinates_are_valid, polygon_rings_are_closed};
pub use is_valid::{Bounds, CoordinateChecks};
// Only for `ogc_validity`.
#[cfg(feature = "geos")]
pub(crate) use is_valid::{line_string_shape_is_valid, polygon_shape_is_valid};
pub use length::{line_string_lengths, multi_line_string_lengths, LengthMetric};
pub use remove_repeated_points::{
    line_strings_without_repeated_points, multi_line_strings_without_repeated_points,
    multi_points_without_repeated_points, multi_polygons_without_repeated_points,
    polygons_without_repeated_points, CollapsedLines, LineOptions,
};
pub use reverse::{
    reverse_line_strings, reverse_multi_line_strings, reverse_multi_points, reverse_multi_polygons,
    reverse_polygons,
};
pub use rings::{
    close_open_rings, multi_polygon_exteriors, multi_polygon_interiors, polygon_exteriors,
    polygon_interiors, polygons_from_exteriors_and_holes,
};
pub use selected_coords::SelectedAction;
pub use too_short::TooShort;
