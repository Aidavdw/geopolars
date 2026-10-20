//! Simplification, forwarded to GEOS.
//! Same thing as GeoPandas, where we switch algo based on `preserve_topology`:
//! - if `preserve_topology=false`: Douglas-Peucker
//! - if `preserve_topology=true`: GEOS's topology-preserving simplifier.

use geopolars_core::{LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray};
use geos::Geometry;

use super::{GeosGeometries, ToGeosRows};

/// Every row simplified to within `tolerance`, forwarded to GEOS.
/// The endpoints are always kept.
/// GEOS keeps z, but drops m.
/// If `preserve_topology=true`, the result never self-intersects.
/// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
/// otherwise its error is returned.
pub fn simplify_line_strings(
    array: &LineStringArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
) -> geos::GResult<LineStringArray> {
    simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?
    .into_line_string_array(array.coords.dim().has_z(), ignore_per_row_errors)
}

/// Every row simplified to within `tolerance`, forwarded to GEOS.
/// The endpoints of every part are always kept, but an empty part is dropped.
/// GEOS keeps z, but drops m.
/// If `preserve_topology=true`, the result never self-intersects.
/// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
/// otherwise its error is returned.
pub fn simplify_multi_line_strings(
    array: &MultiLineStringArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
) -> geos::GResult<MultiLineStringArray> {
    simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?
    .into_multi_line_string_array(array.coords.dim().has_z(), ignore_per_row_errors)
}

/// Every row simplified to within `tolerance`, forwarded to GEOS.
/// GEOS keeps z, but drops m.
///
/// If `preserve_topology=false`, a polygon (or hole) smaller than `tolerance` collapses.
/// Then the polygon becomes empty, and the hole is dropped.
///
/// GEOS repairs a result that crosses itself, which can split it into a multipolygon,
/// which a polygon column can't hold: that row fails.
/// If `preserve_topology=true`, every ring is kept and nothing crosses.
///
/// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
/// otherwise its error is returned.
pub fn simplify_polygons(
    array: &PolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
) -> geos::GResult<PolygonArray> {
    simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?
    .into_polygon_array(array.coords.dim().has_z(), ignore_per_row_errors)
}

/// Every row simplified to within `tolerance`, forwarded to GEOS.
/// GEOS keeps z, but drops m.
///
/// If `preserve_topology=false`, a part (or hole) smaller than `tolerance` is dropped,
/// and a part crossing itself is repaired, which can split it in more parts.
/// If `preserve_topology=true`, every part and ring is kept and nothing crosses.
///
/// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
/// otherwise its error is returned.
pub fn simplify_multi_polygons(
    array: &MultiPolygonArray,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
) -> geos::GResult<MultiPolygonArray> {
    simplify_via_geos(
        array.to_geos_rows(ignore_per_row_errors)?,
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    )?
    .into_multi_polygon_array(array.coords.dim().has_z(), ignore_per_row_errors)
}

/// Every geometry simplified to within `tolerance`, forwarded to GEOS.
/// A missing geometry stays missing.
/// If `ignore_per_row_errors=true`, a geometry GEOS fails on becomes missing;
/// otherwise its error is returned.
/// The size hint is kept as an upper bound: simplifying removes vertices.
/// Only where GEOS repairs a crossing can it add some.
fn simplify_via_geos(
    mut geometries: GeosGeometries,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
) -> geos::GResult<GeosGeometries> {
    with_const_flags!(
        simplify_via_geos_with(&mut geometries.geometries, tolerance, ignore_per_row_errors);
        preserve_topology
    )?;
    Ok(geometries)
}

/// See [`simplify_via_geos`].
/// If `PRESERVE_TOPOLOGY=true`, GEOS's topology-preserving simplifier is used,
/// otherwise Douglas-Peucker.
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
fn simplify_via_geos_with<const PRESERVE_TOPOLOGY: bool>(
    geometries: &mut [Option<Geometry>],
    tolerance: f64,
    ignore_per_row_errors: bool,
) -> geos::GResult<()> {
    for geometry in geometries {
        let Some(input) = geometry else {
            continue;
        };
        let simplified = if PRESERVE_TOPOLOGY {
            input.topology_preserve_simplify(tolerance)
        } else {
            input.simplify(tolerance)
        };
        *geometry = match simplified {
            Ok(simple) => Some(simple),
            Err(e) if !ignore_per_row_errors => return Err(e),
            Err(_) => None,
        };
    }
    Ok(())
}
