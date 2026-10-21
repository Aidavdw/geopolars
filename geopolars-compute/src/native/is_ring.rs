//! Whether a linestring is a ring: at least [MIN_RING_VERTICES] vertices,
//! with its last at the same place as its first.
//! m is a measure rather than a position, so it is not compared.

use std::ops::Range;

use geopolars_core::output::{bool_array, list_array};
use geopolars_core::{
    is_close, LineStringArray, MultiLineStringArray, ValidCoords, MIN_RING_VERTICES,
};
use polars_arrow::array::{BooleanArray, ListArray};
use polars_arrow::offset::Offsets;

use super::per_row_bool;

/// Whether coordinates `i` and `j` of `coords` lie at almost the same place:
/// [`is_close`] in x and y, and in z if `Z=true`.
#[inline(always)]
pub fn same_place<const Z: bool>(coords: &ValidCoords, i: usize, j: usize) -> bool {
    is_close(coords.x()[i], coords.x()[j])
        && is_close(coords.y()[i], coords.y()[j])
        && (!Z || {
            let z = coords.z().expect("a z, with `Z`");
            is_close(z[i], z[j])
        })
}

/// Whether the coordinates `vertices` of `coords` make a ring.
/// See [same_place] for `Z`.
#[inline(always)]
pub fn is_ring<const Z: bool>(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    vertices.len() >= MIN_RING_VERTICES && same_place::<Z>(coords, vertices.start, vertices.end - 1)
}

/// See [same_place] for `Z`.
pub fn line_strings_are_rings<const Z: bool>(array: &LineStringArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        is_ring::<Z>(&array.coords, array.rows.range(i))
    })
}

/// Whether each part of a multilinestring is a ring, as a list per row of a chunk.
/// See [same_place] for `Z`.
pub fn multi_line_strings_are_rings<const Z: bool>(array: &MultiLineStringArray) -> ListArray<i64> {
    // TODO: For multiline strings, we only need to check if they all combined form a ring:
    // Every end point needs to be approx overlapping to the start point of the last one,
    // looping over at the end (like a normal ring)
    let rows = &array.rows;
    let mut values = Vec::with_capacity(array.lines.len());
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it gets no parts.
        if rows.is_valid(i) {
            values.extend(
                array
                    .get(i)
                    .line_strings()
                    .map(|line| is_ring::<Z>(line.coords, line.vertices)),
            );
        }
        offsets.push(values.len() as i64);
    }
    // SAFETY: they start at 0, and every end is the number of values so far, which only grows.
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();
    list_array(
        bool_array(values, None).boxed(),
        offsets,
        rows.validity().cloned(),
    )
}
