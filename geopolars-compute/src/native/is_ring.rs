//! Whether a linestring is a ring: at least [MIN_RING_VERTICES] vertices,
//! with its last at exactly the same place as its first.
//! A multilinestring is a ring when its parts chain into one (see [multi_line_strings_are_rings]).
//! m is a measure rather than a position, so it is not compared.

use geopolars_core::{
    is_ring, same_place, LineStringArray, MultiLineStringArray, ScalarMultiLineString,
    MIN_RING_VERTICES,
};
use polars_arrow::array::BooleanArray;

use crate::per_row::per_row_bool;

/// See [same_place] for `Z`.
pub fn line_strings_are_rings<const Z: bool>(array: &LineStringArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        is_ring::<Z>(&array.coords, array.rows.range(i))
    })
}

/// Whether the parts of `multi` chain into one ring.
/// each part ends where the next starts, and the last ends where the first starts.
/// Other than that same requirements as single linestring.
/// See [same_place] for `Z`.
#[inline(always)]
fn parts_form_ring<const Z: bool>(multi: &ScalarMultiLineString) -> bool {
    let coords = multi.coords;
    let mut parts = multi.line_strings();
    let Some(first) = parts.next() else {
        return false;
    };
    if first.vertices.is_empty() {
        return false;
    }
    let mut joined_vertices = first.vertices.len();
    let mut end = first.vertices.end - 1;
    for part in parts {
        if part.vertices.is_empty() || !same_place::<Z>(coords, end, part.vertices.start) {
            return false;
        }
        joined_vertices += part.vertices.len() - 1;
        end = part.vertices.end - 1;
    }
    joined_vertices >= MIN_RING_VERTICES && same_place::<Z>(coords, first.vertices.start, end)
}

/// See [parts_form_ring] for what makes a ring, and `Z`.
pub fn multi_line_strings_are_rings<const Z: bool>(array: &MultiLineStringArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| parts_form_ring::<Z>(&array.get(i)))
}
