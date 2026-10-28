//! The coordinate and shape checks of `is_valid_geoarrow` and `is_valid_ogc`.
//! See them in Python for the rules.
//!
//! The coordinates are checked in one pass that does not know the geometry.
//! What that pass reads for every vertex is a const flag.
//!
//! # Const generic flags
//!
//! ## `CHECK_FINITE_COORDINATES`
//!
//! The coordinate checks are generic over `CHECK_FINITE_COORDINATES`.
//! If `CHECK_FINITE_COORDINATES=true`, every axis that is read has to be finite.
//!
//! ## `CHECK_CRS_BOUNDS`
//!
//! The coordinate checks are generic over `CHECK_CRS_BOUNDS`.
//! If `CHECK_CRS_BOUNDS=true`, x and y have to lie inside the [Bounds].
//!
//! ## `ALLOW_EMPTY_POINTS`
//!
//! The coordinate checks are generic over `ALLOW_EMPTY_POINTS`.
//! If `ALLOW_EMPTY_POINTS=true`, the empty point passes every check.
//! Otherwise it is held to them like any other coordinate.
//!
//! ## `Z`
//!
//! The coordinate checks and the ring closure are generic over `Z`.
//! If `Z=true`, the z values are read too.
//! A ring then has to close in z as well.
//!
//! ## `M`
//!
//! The coordinate checks are generic over `M`.
//! If `M=true`, the m values are read too.

use std::ops::Range;

#[cfg(feature = "geos")]
use geopolars_core::MIN_RING_VERTICES;
use geopolars_core::{same_place, ScalarPolygon, ValidCoords};
use polars_arrow::bitmap::Bitmap;

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
pub type Bounds = [f64; 4];

/// The coordinate checks to run. See `is_valid_geoarrow` in Python.
#[derive(Debug, Clone, Copy)]
pub struct CoordinateChecks {
    /// Where x and y have to lie, if anywhere.
    pub bounds: Option<Bounds>,
    pub check_finite_coordinates: bool,
    pub allow_empty_points: bool,
}

impl CoordinateChecks {
    pub fn check_crs_bounds(&self) -> bool {
        self.bounds.is_some()
    }
}

/// Whether coordinate `i` is the empty point: NaN in x and y.
#[inline(always)]
pub(crate) fn is_empty_point(coords: &ValidCoords, i: usize) -> bool {
    coords.x()[i].is_nan() && coords.y()[i].is_nan()
}

/// Whether `x` and `y` lie inside `bounds`.
#[inline(always)]
pub(crate) fn is_within(bounds: &Bounds, x: f64, y: f64) -> bool {
    let [xmin, ymin, xmax, ymax] = *bounds;
    (xmin..=xmax).contains(&x) && (ymin..=ymax).contains(&y)
}

#[inline(always)]
fn coordinate_is_valid<
    const CHECK_FINITE_COORDINATES: bool,
    const CHECK_CRS_BOUNDS: bool,
    const ALLOW_EMPTY_POINTS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    i: usize,
    bounds: &Bounds,
) -> bool {
    let (x, y) = (coords.x()[i], coords.y()[i]);
    let finite = !CHECK_FINITE_COORDINATES
        || (x.is_finite()
            && y.is_finite()
            && (!Z || coords.z().expect("a z, with `Z`")[i].is_finite())
            && (!M || coords.m().expect("an m, with `M`")[i].is_finite()));
    let within = !CHECK_CRS_BOUNDS || is_within(bounds, x, y);
    (finite && within) || (ALLOW_EMPTY_POINTS && is_empty_point(coords, i))
}

fn coordinates_are_valid_with<
    const CHECK_FINITE_COORDINATES: bool,
    const CHECK_CRS_BOUNDS: bool,
    const ALLOW_EMPTY_POINTS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    bounds: &Bounds,
) -> Bitmap {
    Bitmap::from_iter((0..coords.len()).map(|i| {
        coordinate_is_valid::<CHECK_FINITE_COORDINATES, CHECK_CRS_BOUNDS, ALLOW_EMPTY_POINTS, Z, M>(
            coords, i, bounds,
        )
    }))
}

/// Which coordinates of `coords` pass `checks`.
/// `None` when there is nothing to check.
pub(crate) fn coordinates_are_valid(
    coords: &ValidCoords,
    checks: &CoordinateChecks,
) -> Option<Bitmap> {
    if !checks.check_finite_coordinates && !checks.check_crs_bounds() {
        return None;
    }
    let dim = coords.dim();
    let bounds = checks.bounds.unwrap_or_default();
    Some(with_const_flags!(
        coordinates_are_valid_with(coords, &bounds);
        checks.check_finite_coordinates, checks.check_crs_bounds(), checks.allow_empty_points,
        dim.has_z(), dim.has_m()
    ))
}

/// Whether every coordinate in `span` passed, by [coordinates_are_valid].
#[inline(always)]
pub(crate) fn all_coordinates_valid(valid: Option<&Bitmap>, span: Range<usize>) -> bool {
    valid.is_none_or(|valid| valid.null_count_range(span.start, span.len()) == 0)
}

/// Whether every ring of `polygon` is empty or closed, as GeoArrow asks.
#[inline(always)]
pub(crate) fn polygon_rings_are_closed<const Z: bool>(polygon: &ScalarPolygon) -> bool {
    polygon.rings().all(|ring| {
        let vertices = ring.vertices;
        vertices.is_empty() || same_place::<Z>(ring.coords, vertices.start, vertices.end - 1)
    })
}

/// Whether `vertices` hold at least `min` vertices that lie elsewhere in x and y than the one before.
#[cfg(feature = "geos")]
#[inline(always)]
fn has_at_least_distinct_vertices(
    coords: &ValidCoords,
    vertices: Range<usize>,
    min: usize,
) -> bool {
    let start = vertices.start;
    vertices
        .filter(|&i| i == start || !same_place::<false>(coords, i - 1, i))
        .take(min)
        .count()
        == min
}

/// Whether `vertices` make a linestring OGC accepts: none, or at least two distinct ones.
#[cfg(feature = "geos")]
#[inline(always)]
pub(crate) fn line_string_shape_is_valid(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    vertices.is_empty() || has_at_least_distinct_vertices(coords, vertices, 2)
}

/// Whether `polygon` has a shape OGC accepts:
/// every ring empty, or every ring with at least 4 distinct vertices.
#[cfg(feature = "geos")]
#[inline(always)]
pub(crate) fn polygon_shape_is_valid(polygon: &ScalarPolygon) -> bool {
    // An empty exterior makes the empty polygon, so every other ring has to be empty too.
    if polygon
        .rings()
        .next()
        .is_some_and(|exterior| exterior.vertices.is_empty())
    {
        return polygon.rings().all(|ring| ring.vertices.is_empty());
    }
    polygon
        .rings()
        .all(|ring| has_at_least_distinct_vertices(ring.coords, ring.vertices, MIN_RING_VERTICES))
}
