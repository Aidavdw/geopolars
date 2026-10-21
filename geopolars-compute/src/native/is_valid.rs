//! The coordinate and shape checks of `is_valid`, one kernel per check and kind.
//! See `is_valid` in Python for the rules.
//!
//! Each gives a `bool` per row, `false` for a missing row.
//! What a check reads for every vertex is a const flag.

use std::ops::Range;

use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ScalarPolygon, ValidCoords,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use super::{is_ring, per_row_bool, same_place};

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
fn coordinates_are_valid<const FINITE: bool, const BOUNDS: bool, const Z: bool, const M: bool>(
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
fn point_is_valid<const FINITE: bool, const BOUNDS: bool, const Z: bool, const M: bool>(
    coords: &ValidCoords,
    i: usize,
    bounds: &Bounds,
) -> bool {
    coordinates_are_valid::<FINITE, BOUNDS, Z, M>(coords, i..i + 1, bounds)
        || is_empty_point::<Z, M>(coords, i)
}

/// See [point_is_valid] for the flags.
fn points_have_valid_coordinates_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    array: &PointArray,
    bounds: &Bounds,
) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::from_iter((0..coords.len()).map(|i| {
        coords.is_valid(i)
            && point_is_valid::<FINITE, BOUNDS, Z, M>(coords.ignoring_validity(), i, bounds)
    }));
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

/// Whether each point is valid, or the empty point.
/// If `check_finite=true`, every axis has to be finite.
/// If `bounds` is given, x and y have to lie inside it.
pub fn points_have_valid_coordinates(
    array: &PointArray,
    check_finite: bool,
    bounds: Option<&Bounds>,
) -> BooleanArray {
    let dim = array.coords.dim();
    let bounds_or_any = bounds.copied().unwrap_or_default();
    with_const_flags!(
        points_have_valid_coordinates_with(array, &bounds_or_any);
        check_finite, bounds.is_some(), dim.has_z(), dim.has_m()
    )
}

/// See [point_is_valid] for the flags.
fn multi_points_have_valid_coordinates_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    array: &MultiPointArray,
    bounds: &Bounds,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .rows
            .range(i)
            .all(|j| point_is_valid::<FINITE, BOUNDS, Z, M>(&array.coords, j, bounds))
    })
}

/// [points_have_valid_coordinates], for every point of a multipoint.
pub fn multi_points_have_valid_coordinates(
    array: &MultiPointArray,
    check_finite: bool,
    bounds: Option<&Bounds>,
) -> BooleanArray {
    let dim = array.coords.dim();
    let bounds_or_any = bounds.copied().unwrap_or_default();
    with_const_flags!(
        multi_points_have_valid_coordinates_with(array, &bounds_or_any);
        check_finite, bounds.is_some(), dim.has_z(), dim.has_m()
    )
}

/// See [coordinates_are_valid] for the flags.
fn have_valid_coordinates_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    rows: &List,
    coords: &ValidCoords,
    bounds: &Bounds,
    coordinates: impl Fn(usize) -> Range<usize>,
) -> BooleanArray {
    per_row_bool(rows, |i| {
        coordinates_are_valid::<FINITE, BOUNDS, Z, M>(coords, coordinates(i), bounds)
    })
}

/// Whether the coordinates of each row pass the coordinate checks.
/// `coordinates(i)` gives where the coordinates of row `i` sit in `coords`.
/// See [points_have_valid_coordinates] for `check_finite` and `bounds`.
fn have_valid_coordinates(
    rows: &List,
    coords: &ValidCoords,
    check_finite: bool,
    bounds: Option<&Bounds>,
    coordinates: impl Fn(usize) -> Range<usize>,
) -> BooleanArray {
    let dim = coords.dim();
    let bounds_or_any = bounds.copied().unwrap_or_default();
    // z and m are only read to check that they are finite.
    with_const_flags!(
        have_valid_coordinates_with(rows, coords, &bounds_or_any, coordinates);
        check_finite, bounds.is_some(), check_finite && dim.has_z(), check_finite && dim.has_m()
    )
}

/// See [points_have_valid_coordinates] for `check_finite` and `bounds`.
pub fn line_strings_have_valid_coordinates(
    array: &LineStringArray,
    check_finite: bool,
    bounds: Option<&Bounds>,
) -> BooleanArray {
    have_valid_coordinates(&array.rows, &array.coords, check_finite, bounds, |i| {
        array.rows.range(i)
    })
}

/// See [points_have_valid_coordinates] for `check_finite` and `bounds`.
pub fn multi_line_strings_have_valid_coordinates(
    array: &MultiLineStringArray,
    check_finite: bool,
    bounds: Option<&Bounds>,
) -> BooleanArray {
    have_valid_coordinates(&array.rows, &array.coords, check_finite, bounds, |i| {
        array.lines.span(array.rows.range(i))
    })
}

/// See [points_have_valid_coordinates] for `check_finite` and `bounds`.
pub fn polygons_have_valid_coordinates(
    array: &PolygonArray,
    check_finite: bool,
    bounds: Option<&Bounds>,
) -> BooleanArray {
    have_valid_coordinates(&array.rows, &array.coords, check_finite, bounds, |i| {
        array.rings.span(array.rows.range(i))
    })
}

/// See [points_have_valid_coordinates] for `check_finite` and `bounds`.
pub fn multi_polygons_have_valid_coordinates(
    array: &MultiPolygonArray,
    check_finite: bool,
    bounds: Option<&Bounds>,
) -> BooleanArray {
    have_valid_coordinates(&array.rows, &array.coords, check_finite, bounds, |i| {
        array.rings.span(array.polygons.span(array.rows.range(i)))
    })
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
fn line_string_shape_is_valid<const ZERO: bool, const Z: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
) -> bool {
    vertices.len() != 1 && no_zero_length_sections::<ZERO, Z>(coords, vertices)
}

/// Whether `polygon` has a valid shape: all rings empty, or all of them rings.
/// See [no_zero_length_sections] for the flags.
#[inline(always)]
fn polygon_shape_is_valid<const ZERO: bool, const Z: bool>(polygon: &ScalarPolygon) -> bool {
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

fn line_strings_have_valid_shape_with<const ZERO: bool, const Z: bool>(
    array: &LineStringArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        line_string_shape_is_valid::<ZERO, Z>(&array.coords, array.rows.range(i))
    })
}

/// Whether each linestring has no vertices, or at least two.
/// If `check_zero_length_sections=true`, no two vertices in a row may lie at the same place.
pub fn line_strings_have_valid_shape(
    array: &LineStringArray,
    check_zero_length_sections: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(line_strings_have_valid_shape_with(array); check_zero_length_sections, z)
}

fn multi_line_strings_have_valid_shape_with<const ZERO: bool, const Z: bool>(
    array: &MultiLineStringArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .get(i)
            .line_strings()
            .all(|line| line_string_shape_is_valid::<ZERO, Z>(line.coords, line.vertices))
    })
}

/// [line_strings_have_valid_shape], for every part.
pub fn multi_line_strings_have_valid_shape(
    array: &MultiLineStringArray,
    check_zero_length_sections: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_line_strings_have_valid_shape_with(array);
        check_zero_length_sections, z
    )
}

fn polygons_have_valid_shape_with<const ZERO: bool, const Z: bool>(
    array: &PolygonArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        polygon_shape_is_valid::<ZERO, Z>(&array.get(i))
    })
}

/// Whether each polygon has only empty rings, or only rings (see [is_ring]).
/// See [line_strings_have_valid_shape] for `check_zero_length_sections`.
pub fn polygons_have_valid_shape(
    array: &PolygonArray,
    check_zero_length_sections: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(polygons_have_valid_shape_with(array); check_zero_length_sections, z)
}

fn multi_polygons_have_valid_shape_with<const ZERO: bool, const Z: bool>(
    array: &MultiPolygonArray,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .get(i)
            .polygons()
            .all(|polygon| polygon_shape_is_valid::<ZERO, Z>(&polygon))
    })
}

/// [polygons_have_valid_shape], for every part.
pub fn multi_polygons_have_valid_shape(
    array: &MultiPolygonArray,
    check_zero_length_sections: bool,
) -> BooleanArray {
    let z = array.coords.dim().has_z();
    with_const_flags!(
        multi_polygons_have_valid_shape_with(array);
        check_zero_length_sections, z
    )
}
