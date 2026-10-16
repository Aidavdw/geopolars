//! Simplification, forwarded to GEOS.
//! Same thing as GeoPandas, where we switch algo based on `preserve_topology`:
//! - if `preserve_topology=false`: Douglas-Peucker
//! - if `preserve_topology=true`: GEOS's topology-preserving simplifier.

use std::ops::Range;

use geos::{Geom, Geometry};

use super::linestrings::{
    line_string_from_geos, line_string_to_geos, multi_line_string_from_geos,
    multi_line_string_to_geos,
};
use super::polygons::{
    multi_polygon_from_geos, multi_polygon_to_geos, polygon_from_geos, polygon_to_geos,
};
use super::{axes, Axes, Buffers};
use crate::{
    GeoArrowLineStringOutput, GeoArrowListView, GeoArrowMultiLineStringOutput,
    GeoArrowMultiPolygonOutput, GeoArrowPolygonOutput, GeoArrowValidCoordsView,
    GeoArrowValidListView, GeoDimension, LineStringArrayView, MultiLineStringArrayView,
    MultiPolygonArrayView, OwnedAxes, PolygonArrayView,
};

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArrayView<R, C> {
    /// Every row simplified to within `tolerance`, forwarded to GEOS.
    /// The endpoints are always kept.
    /// GEOS keeps z, but drops m.
    /// If `preserve_topology=true`, the result never self-intersects.
    /// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
    /// otherwise its error is returned.
    pub fn simplify<O: GeoArrowLineStringOutput<Validity = R::Validity>>(
        &self,
        tolerance: f64,
        preserve_topology: bool,
        ignore_per_row_errors: bool,
    ) -> geos::GResult<O> {
        let Simplified {
            dim,
            axes,
            offsets: [rows],
            failed,
        } = simplify_rows_via_geos(
            &self.rows,
            &self.coords,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            line_string_to_geos,
            line_string_from_geos,
        )?;
        Ok(O::from_buffers(
            dim,
            axes,
            rows,
            self.rows.validity(),
            &failed,
        ))
    }
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiLineStringArrayView<R, L, C>
{
    /// Every row simplified to within `tolerance`, forwarded to GEOS.
    /// The endpoints of every part are always kept, but an empty part is dropped.
    /// GEOS keeps z, but drops m.
    /// If `preserve_topology=true`, the result never self-intersects.
    /// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
    /// otherwise its error is returned.
    pub fn simplify<O: GeoArrowMultiLineStringOutput<Validity = R::Validity>>(
        &self,
        tolerance: f64,
        preserve_topology: bool,
        ignore_per_row_errors: bool,
    ) -> geos::GResult<O> {
        let Simplified {
            dim,
            axes,
            offsets: [lines, rows],
            failed,
        } = simplify_rows_via_geos(
            &self.rows,
            &self.coords,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            |axes, parts| multi_line_string_to_geos(axes, &self.lines, parts),
            multi_line_string_from_geos,
        )?;
        Ok(O::from_buffers(
            dim,
            axes,
            lines,
            rows,
            self.rows.validity(),
            &failed,
        ))
    }
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    PolygonArrayView<R, L, C>
{
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
    pub fn simplify<O: GeoArrowPolygonOutput<Validity = R::Validity>>(
        &self,
        tolerance: f64,
        preserve_topology: bool,
        ignore_per_row_errors: bool,
    ) -> geos::GResult<O> {
        let Simplified {
            dim,
            axes,
            offsets: [rings, rows],
            failed,
        } = simplify_rows_via_geos(
            &self.rows,
            &self.coords,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            |axes, parts| polygon_to_geos(axes, &self.rings, parts),
            polygon_from_geos,
        )?;
        Ok(O::from_buffers(
            dim,
            axes,
            rings,
            rows,
            self.rows.validity(),
            &failed,
        ))
    }
}

impl<R, P, L, C> MultiPolygonArrayView<R, P, L, C>
where
    R: GeoArrowListView,
    P: GeoArrowValidListView,
    L: GeoArrowValidListView,
    C: GeoArrowValidCoordsView,
{
    /// Every row simplified to within `tolerance`, forwarded to GEOS.
    /// GEOS keeps z, but drops m.
    ///
    /// If `preserve_topology=false`, a part (or hole) smaller than `tolerance` is dropped,
    /// and a part crossing itself is repaired, which can split it in more parts.
    /// If `preserve_topology=true`, every part and ring is kept and nothing crosses.
    ///
    /// If `ignore_per_row_errors=true`, a row GEOS fails on is missing;
    /// otherwise its error is returned.
    pub fn simplify<O: GeoArrowMultiPolygonOutput<Validity = R::Validity>>(
        &self,
        tolerance: f64,
        preserve_topology: bool,
        ignore_per_row_errors: bool,
    ) -> geos::GResult<O> {
        let Simplified {
            dim,
            axes,
            offsets: [rings, polygons, rows],
            failed,
        } = simplify_rows_via_geos(
            &self.rows,
            &self.coords,
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
            |axes, parts| multi_polygon_to_geos(axes, &self.polygons, &self.rings, parts),
            multi_polygon_from_geos,
        )?;
        Ok(O::from_buffers(
            dim,
            axes,
            rings,
            polygons,
            rows,
            self.rows.validity(),
            &failed,
        ))
    }
}

/// What [`simplify_rows_via_geos`] builds for the `from_buffers` of the output.
/// Shared among all kinds.
/// The output's type is left to the caller, so [`simplify_rows_via_geos`] needs no generics for it.
struct Simplified<const N: usize> {
    dim: GeoDimension,
    axes: OwnedAxes,
    /// innermost first, so the rows last
    offsets: [Vec<i64>; N],
    failed: Vec<usize>,
}

/// Simplifies every one of `rows` through GEOS.
/// `to_geos` turns the parts of a row into a GEOS geometry.
/// `from_geos` reads the simplified geometry back into the `N` list levels.
/// It leaves the outermost level, the rows, to this function.
/// A row fails where GEOS does.
/// It also fails where `from_geos` gives `false`, because GEOS changed its kind.
fn simplify_rows_via_geos<const N: usize>(
    rows: &impl GeoArrowListView,
    coords: &impl GeoArrowValidCoordsView,
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Geometry>,
    from_geos: impl Fn(&mut Buffers, &Geometry) -> geos::GResult<bool>,
) -> geos::GResult<Simplified<N>> {
    let mut buffers = Buffers::new(coords.dim().has_z(), N, coords.len());
    let failed = with_const_flags!(
        simplify_rows_via_geos_with(
            rows,
            coords,
            tolerance,
            ignore_per_row_errors,
            &mut buffers,
            to_geos,
            from_geos
        );
        rows.is_all_valid(), preserve_topology
    )?;
    let (dim, axes, offsets) = buffers.finish();
    Ok(Simplified {
        dim,
        axes,
        offsets,
        failed,
    })
}

/// See [`simplify_rows_via_geos`].
/// This fills `buffers` and gives the rows that failed.
/// If `ALL_VALID=true`, no row is checked for being missing first.
/// If `PRESERVE_TOPOLOGY=true`, GEOS's topology-preserving simplifier is used,
/// otherwise Douglas-Peucker.
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
fn simplify_rows_via_geos_with<const ALL_VALID: bool, const PRESERVE_TOPOLOGY: bool>(
    rows: &impl GeoArrowListView,
    coords: &impl GeoArrowValidCoordsView,
    tolerance: f64,
    ignore_per_row_errors: bool,
    buffers: &mut Buffers,
    to_geos: impl Fn(&Axes, Range<usize>) -> geos::GResult<Geometry>,
    from_geos: impl Fn(&mut Buffers, &Geometry) -> geos::GResult<bool>,
) -> geos::GResult<Vec<usize>> {
    let axes = axes(coords);
    let mut failed = Vec::new();
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not converted.
        if ALL_VALID || rows.is_valid(i) {
            let filled = buffers.filled();
            let simplified = to_geos(&axes, rows.range(i)).and_then(|geometry| {
                let simple = if PRESERVE_TOPOLOGY {
                    geometry.topology_preserve_simplify(tolerance)?
                } else {
                    geometry.simplify(tolerance)?
                };
                if from_geos(buffers, &simple)? {
                    return Ok(());
                }
                Err(geos::Error::ConversionError(format!(
                    "simplified into a {:?}, which this column can't hold; \
                     preserve_topology=true never does that",
                    simple.geometry_type()?
                )))
            });
            match simplified {
                Ok(()) => {}
                Err(e) if !ignore_per_row_errors => return Err(e),
                // Nothing of the failed row may stay behind.
                Err(_) => {
                    buffers.truncate(&filled);
                    failed.push(i);
                }
            }
        }
        buffers.close_row();
    }
    Ok(failed)
}
