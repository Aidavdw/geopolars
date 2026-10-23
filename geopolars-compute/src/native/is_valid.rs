//! The coordinate and shape checks of `is_valid`.
//! See `is_valid` in Python for the rules.
//!
//! What a check reads for every vertex is a const flag.
//!
//! # Const generic flags
//!
//! See [crate::validity] for `CHECK_FINITE_COORDINATES`, `CHECK_CRS_BOUNDS`, `Z` and `M`.

use std::ops::Range;

use geopolars_core::{is_ring, ScalarPolygon, ValidCoords};

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
pub type Bounds = [f64; 4];

/// Whether coordinate `i` is the empty point: NaN in x and y.
#[inline(always)]
pub(crate) fn is_empty_point(coords: &ValidCoords, i: usize) -> bool {
    coords.x()[i].is_nan() && coords.y()[i].is_nan()
}

/// Whether coordinate `i` passes the coordinate checks, or is the empty point.
#[inline(always)]
pub(crate) fn coordinate_is_valid<
    const CHECK_FINITE_COORDINATES: bool,
    const CHECK_CRS_BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    i: usize,
    bounds: &Bounds,
) -> bool {
    let (x, y) = (coords.x()[i], coords.y()[i]);
    let [xmin, ymin, xmax, ymax] = *bounds;
    let finite = !CHECK_FINITE_COORDINATES
        || (x.is_finite()
            && y.is_finite()
            && (!Z || coords.z().expect("a z, with `Z`")[i].is_finite())
            && (!M || coords.m().expect("an m, with `M`")[i].is_finite()));
    let within = !CHECK_CRS_BOUNDS || ((xmin..=xmax).contains(&x) && (ymin..=ymax).contains(&y));
    (finite && within) || is_empty_point(coords, i)
}

/// Whether `vertices` make a valid linestring: none, or at least two.
#[inline(always)]
pub(crate) fn line_string_shape_is_valid(vertices: Range<usize>) -> bool {
    vertices.len() != 1
}

/// Whether `polygon` has a valid shape: all rings empty, or all of them rings.
#[inline(always)]
pub(crate) fn polygon_shape_is_valid<const Z: bool>(polygon: &ScalarPolygon) -> bool {
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
        .all(|ring| is_ring::<Z>(ring.coords, ring.vertices))
}
