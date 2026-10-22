//! The coordinate and shape checks of `is_valid`
//! See `is_valid` in Python for the rules.
//!
//! What a check reads for every vertex is a const flag.

use std::ops::Range;

use geopolars_core::{is_ring, same_place, ScalarPolygon, ValidCoords};

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
pub type Bounds = [f64; 4];

// Does not stop at the first that fails, so that it vectorises.
#[inline(always)]
fn holds_for_all_without_short_circuit(values: &[f64], f: impl Fn(f64) -> bool) -> bool {
    values.iter().fold(true, |ok, &value| ok & f(value))
}

/// Whether the coordinates `range` of `coords` pass the coordinate checks.
/// If `FINITE=true`, every axis has to be finite, z only if `Z=true` and m only if `M=true`.
/// If `BOUNDS=true`, x and y have to lie inside `bounds`.
#[inline(always)]
pub(crate) fn coordinates_are_valid<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    range: Range<usize>,
    bounds: &Bounds,
) -> bool {
    let finite =
        |axis: &[f64]| holds_for_all_without_short_circuit(&axis[range.clone()], f64::is_finite);
    let within = |axis: &[f64], min: f64, max: f64| {
        holds_for_all_without_short_circuit(&axis[range.clone()], |value| {
            (min..=max).contains(&value)
        })
    };
    let [xmin, ymin, xmax, ymax] = *bounds;
    (!FINITE
        || (finite(coords.x())
            && finite(coords.y())
            && (!Z || finite(coords.z().expect("a z, with `Z`")))
            && (!M || finite(coords.m().expect("an m, with `M`")))))
        && (!BOUNDS || (within(coords.x(), xmin, xmax) && within(coords.y(), ymin, ymax)))
}

/// Whether coordinate `i` is NaN on every axis: z only if `Z=true`, m only if `M=true`.
#[inline(always)]
fn is_empty_point<const Z: bool, const M: bool>(coords: &ValidCoords, i: usize) -> bool {
    coords.x()[i].is_nan()
        && coords.y()[i].is_nan()
        && (!Z || coords.z().expect("a z, with `Z`")[i].is_nan())
        && (!M || coords.m().expect("an m, with `M`")[i].is_nan())
}

/// Whether coordinate `i` passes the coordinate checks, or is the empty point.
/// See [coordinates_are_valid] for the flags.
/// The empty point needs every axis, so `Z` and `M` have to follow the dimension here.
#[inline(always)]
pub(crate) fn point_is_valid<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    coords: &ValidCoords,
    i: usize,
    bounds: &Bounds,
) -> bool {
    coordinates_are_valid::<FINITE, BOUNDS, Z, M>(coords, i..i + 1, bounds)
        || is_empty_point::<Z, M>(coords, i)
}

/// Whether no two of `vertices` in a row lie at the [same place](same_place), if `ZERO=true`.
/// See [same_place] for `Z`.
#[inline(always)]
fn no_zero_length_sections<const ZERO: bool, const Z: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
) -> bool {
    !ZERO || (vertices.start + 1..vertices.end).all(|i| !same_place::<Z>(coords, i - 1, i))
}

/// Whether `vertices` make a valid linestring: none, or at least two.
/// See [no_zero_length_sections] for the flags.
#[inline(always)]
pub(crate) fn line_string_shape_is_valid<const ZERO: bool, const Z: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
) -> bool {
    vertices.len() != 1 && no_zero_length_sections::<ZERO, Z>(coords, vertices)
}

/// Whether `polygon` has a valid shape: all rings empty, or all of them rings.
/// See [no_zero_length_sections] for the flags.
#[inline(always)]
pub(crate) fn polygon_shape_is_valid<const ZERO: bool, const Z: bool>(
    polygon: &ScalarPolygon,
) -> bool {
    // An empty exterior makes the empty polygon, so every other ring has to be empty too.
    if polygon
        .rings()
        .next()
        .is_some_and(|exterior| exterior.vertices.is_empty())
    {
        return polygon.rings().all(|ring| ring.vertices.is_empty());
    }
    polygon.rings().all(|ring| {
        is_ring::<Z>(ring.coords, ring.vertices.clone())
            && no_zero_length_sections::<ZERO, Z>(ring.coords, ring.vertices)
    })
}
