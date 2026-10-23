//! Buffers, forwarded to GEOS.
//!
//! GEOS builds a buffer in 2D, so every output is XY.
//!
//! # Const generic flags
//!
//! ## `EARLY_FINITE_CHECK`
//!
//! The public functions take `early_finite_check`.
//! See [super] for `EARLY_FINITE_CHECK`.

// TODO: also take `distance` as a column, one per row (broadcast when it has one value).
// Its sign is then only known per row, not at plan time.
// So `allow_splitting` would have to give a multipolygon whatever the distance.

use geopolars_core::{MultiPolygonArray, PolygonArray};
use geos::{BufferParams, Geom};
pub use geos::{CapStyle, JoinStyle};

use super::{GeosGeometries, GeosRow, ToGeosRows};
use crate::GeoComputeResult;

/// GEOS's `BufferParams`
pub struct BufferStyle {
    /// The number of segments per quarter circle.
    pub quadrant_segments: i32,
    pub cap_style: CapStyle,
    pub join_style: JoinStyle,
    /// Only used with [`JoinStyle::Mitre`].
    pub mitre_limit: f64,
    /// If `single_sided=true`, only the left side of a line is buffered,
    /// or the right side for a negative distance.
    /// GEOS then ignores `cap_style`.
    pub single_sided: bool,
}

/// Every row buffered by `distance`, forwarded to GEOS, as a polygon.
/// This is for a point, linestring or polygon.
/// A distance of at most 0 gives an empty polygon for a point or linestring.
/// A negative distance shrinks a polygon.
/// A row that GEOS buffers into several polygons fails like a GEOS error.
/// For a single kind, that only happens with a negative distance.
/// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
/// otherwise its error is returned.
/// If `ccw_outer=true`, every ring is reversed, so the exterior runs counter-clockwise.
/// Otherwise it runs clockwise, as GEOS gives it.
pub fn buffer_into_polygons(
    array: &impl ToGeosRows,
    distance: f64,
    style: &BufferStyle,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<PolygonArray> {
    buffer_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        style,
        ignore_per_row_errors,
    )?
    .into_polygon_array(false, ignore_per_row_errors, ccw_outer)
}

/// Like [`buffer_into_polygons`], as a multipolygon, for any kind.
/// Parts whose buffers overlap are merged into one.
/// A row that GEOS buffers into several polygons gets them all as parts.
pub fn buffer_into_multi_polygons(
    array: &impl ToGeosRows,
    distance: f64,
    style: &BufferStyle,
    ccw_outer: bool,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
) -> GeoComputeResult<MultiPolygonArray> {
    buffer_via_geos(
        array.to_geos_rows(ignore_per_row_errors, early_finite_check)?,
        distance,
        style,
        ignore_per_row_errors,
    )?
    .into_multi_polygon_array(false, ignore_per_row_errors, ccw_outer)
}

fn buffer_params(style: &BufferStyle) -> GeoComputeResult<BufferParams> {
    Ok(BufferParams::builder()
        .quadrant_segments(style.quadrant_segments)
        .end_cap_style(style.cap_style)
        .join_style(style.join_style)
        .mitre_limit(style.mitre_limit)
        .single_sided(style.single_sided)
        .build()?)
}

/// Every geometry buffered by `distance`, forwarded to GEOS.
/// If `ignore_per_row_errors=true`, a geometry GEOS fails on becomes missing;
/// otherwise its error is returned.
/// The size hint stays the input's: it is only a starting capacity.
/// Errors are the exception, so `ignore_per_row_errors` is checked per row.
fn buffer_via_geos(
    mut geometries: GeosGeometries,
    distance: f64,
    style: &BufferStyle,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<GeosGeometries> {
    let params = buffer_params(style)?;
    for geometry in &mut geometries.geometries {
        let GeosRow::Geometry(input) = geometry else {
            continue;
        };
        *geometry = match input.buffer_with_params(distance, &params) {
            Ok(buffered) => GeosRow::Geometry(buffered),
            Err(e) if !ignore_per_row_errors => return Err(e.into()),
            Err(_) => GeosRow::Missing,
        };
    }
    Ok(geometries)
}
