//! Reversing the order of a geometry's vertices.
//! See `reverse` in Python for the rules.
//!
//! Every kernel copies the coordinates once, then reverses runs of them in place.
//! The list levels are shared with the input, unless the order of the parts changes.
//! A missing row is reversed like any other, as its value is masked anyway.
//!
//! # Const generic flags
//!
//! ## `REVERSE_HOLES`
//!
//! [reverse_polygons] and [reverse_multi_polygons] are generic over `REVERSE_HOLES`.
//! If `REVERSE_HOLES=false`, only the exterior of every polygon is reversed.
//! If `REVERSE_HOLES=true`, its holes are reversed too.
//!
//! ## `NESTED`
//!
//! [reverse_multi_line_strings] and [reverse_multi_polygons] are generic over `NESTED`.
//! If `NESTED=true`, the order of every row's parts is reversed as well.
//! The rings of a polygon keep their order, so its exterior stays first.

use std::ops::Range;

use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    ValidCoords, ValidList,
};
use polars_arrow::offset::Offsets;

use super::rings::offsets_of;

#[inline(always)]
fn ranges_of_lists(lists: &ValidList) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
    (0..lists.len()).map(move |i| lists.range(i))
}

/// `coords` copied, with each of `ranges` reversed.
fn with_ranges_reversed(
    coords: &ValidCoords,
    ranges: impl Iterator<Item = Range<usize>> + Clone,
) -> ValidCoords {
    let reversed = |axis: &[f64]| {
        let mut axis = axis.to_vec();
        for range in ranges.clone() {
            axis[range].reverse();
        }
        axis
    };
    ValidCoords::from_axes((
        reversed(coords.x()),
        reversed(coords.y()),
        coords.z().map(&reversed),
        coords.m().map(&reversed),
    ))
}

/// `coords` with the exterior of each of `polygons` reversed.
/// Each item of `polygons` is the range of that polygon's rings in `rings`.
fn reverse_rings<const REVERSE_HOLES: bool>(
    polygons: impl Iterator<Item = Range<usize>> + Clone,
    rings: &ValidList,
    coords: &ValidCoords,
) -> ValidCoords {
    if REVERSE_HOLES {
        with_ranges_reversed(coords, ranges_of_lists(rings))
    } else {
        let exteriors = polygons
            .filter(|polygon| !polygon.is_empty())
            .map(|polygon| rings.range(polygon.start));
        with_ranges_reversed(coords, exteriors)
    }
}

/// `parts` with the order of the parts of every row of `rows` reversed,
/// for when all items of each row are reversed as one.
fn parts_reversed_per_row(rows: &ValidList, parts: &ValidList) -> ValidList {
    let old: &[i64] = parts.offsets().buffer();
    let mut new = old.to_vec();
    for row in ranges_of_lists(rows) {
        // Reversing the items `start..end` moves the boundary at `o` to `start + end - o`.
        let (start, end) = (old[row.start], old[row.end]);
        for k in 0..=row.len() {
            new[row.start + k] = start + end - old[row.end - k];
        }
    }
    // SAFETY: within a row, the new offsets grow as the old ones do,
    // and every row still starts and ends where it did.
    ValidList::from_offsets(unsafe { Offsets::new_unchecked(new) }.into())
}

/// `array` with the polygons of every row in reverse order.
/// The rings of each polygon keep their order and coordinates.
fn polygons_in_reverse_order(array: &MultiPolygonArray) -> MultiPolygonArray {
    let (rows, polygons, rings) = (
        array.rows.ignoring_validity(),
        &array.polygons,
        &array.rings,
    );
    let reordered = ranges_of_lists(rows).flat_map(|row| row.rev());
    let ring_lengths = reordered
        .clone()
        .flat_map(|j| polygons.range(j).map(|ring| rings.range(ring).len()));
    MultiPolygonArray {
        rows: List::from_offsets(
            offsets_of(ranges_of_lists(rows).map(|row| row.len())),
            array.rows.validity().cloned(),
        ),
        polygons: ValidList::from_offsets(offsets_of(
            reordered.clone().map(|j| polygons.range(j).len()),
        )),
        rings: ValidList::from_offsets(offsets_of(ring_lengths)),
        coords: array
            .coords
            .copy_ranges(reordered.map(|j| rings.span(polygons.range(j)))),
    }
}

pub fn reverse_line_strings(array: &LineStringArray) -> LineStringArray {
    let rows = ranges_of_lists(array.rows.ignoring_validity());
    LineStringArray {
        rows: array.rows.clone(),
        coords: with_ranges_reversed(&array.coords, rows),
    }
}

pub fn reverse_multi_points(array: &MultiPointArray) -> MultiPointArray {
    let rows = ranges_of_lists(array.rows.ignoring_validity());
    MultiPointArray {
        rows: array.rows.clone(),
        coords: with_ranges_reversed(&array.coords, rows),
    }
}

pub fn reverse_polygons<const REVERSE_HOLES: bool>(array: &PolygonArray) -> PolygonArray {
    let rows = ranges_of_lists(array.rows.ignoring_validity());
    PolygonArray {
        rows: array.rows.clone(),
        rings: array.rings.clone(),
        coords: reverse_rings::<REVERSE_HOLES>(rows, &array.rings, &array.coords),
    }
}

pub fn reverse_multi_line_strings<const NESTED: bool>(
    array: &MultiLineStringArray,
) -> MultiLineStringArray {
    let (rows, lines) = (array.rows.ignoring_validity(), &array.lines);
    if NESTED {
        let spans = ranges_of_lists(rows).map(|parts| lines.span(parts));
        MultiLineStringArray {
            rows: array.rows.clone(),
            lines: parts_reversed_per_row(rows, lines),
            coords: with_ranges_reversed(&array.coords, spans),
        }
    } else {
        MultiLineStringArray {
            rows: array.rows.clone(),
            lines: lines.clone(),
            coords: with_ranges_reversed(&array.coords, ranges_of_lists(lines)),
        }
    }
}

/// With `NESTED=true`, the polygons are first reordered into a copy,
/// so unlike for multilinestrings, this is the more expensive option.
pub fn reverse_multi_polygons<const REVERSE_HOLES: bool, const NESTED: bool>(
    array: &MultiPolygonArray,
) -> MultiPolygonArray {
    if NESTED {
        return reverse_multi_polygons::<REVERSE_HOLES, false>(&polygons_in_reverse_order(array));
    }
    let polygons = ranges_of_lists(&array.polygons);
    MultiPolygonArray {
        rows: array.rows.clone(),
        polygons: array.polygons.clone(),
        rings: array.rings.clone(),
        coords: reverse_rings::<REVERSE_HOLES>(polygons, &array.rings, &array.coords),
    }
}
