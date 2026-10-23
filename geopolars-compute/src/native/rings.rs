//! The rings of a polygon:
//! - its exterior is its first ring, as a linestring,
//! - its interior is the rest (its holes), as a multilinestring.
//!
//! A multipolygon gets a list of them, one per polygon.
//!
//! # Const generic flags
//!
//! ## `ALL_VALID`
//!
//! [kept_rings_of_polygons] is generic over `ALL_VALID`.
//! If `ALL_VALID=true`, every row is there, so none is checked for being missing.
//!
//! ## `Z`
//!
//! [close_open_rings] is generic over `Z`.
//! See [same_place] for `Z`.

use std::ops::Range;

use geopolars_core::output::list_array;
use geopolars_core::{
    same_place, LineStringArray, List, MultiLineStringArray, MultiPolygonArray, PolygonArray,
    ValidCoords, ValidList,
};
use polars_arrow::array::ListArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::{Offsets, OffsetsBuffer};

/// Which of a polygon's rings `rings` to keep.
type Keep = fn(Range<usize>) -> Range<usize>;

#[inline(always)]
fn exterior_of(rings: Range<usize>) -> Range<usize> {
    rings.start..(rings.start + 1).min(rings.end)
}

#[inline(always)]
fn interior_of(rings: Range<usize>) -> Range<usize> {
    (rings.start + 1).min(rings.end)..rings.end
}

/// The rings every row of `array` keeps, none for a missing row.
#[inline(always)]
fn kept_rings_of_polygons<const ALL_VALID: bool>(
    array: &PolygonArray,
    keep: Keep,
) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
    let rows = &array.rows;
    // A missing row's offsets point anywhere, so it keeps nothing.
    (0..rows.len()).map(move |i| {
        if ALL_VALID || rows.is_valid(i) {
            keep(rows.range(i))
        } else {
            0..0
        }
    })
}

/// The rings every polygon of the rows of `array` keeps.
#[inline(always)]
fn kept_rings_of_multi_polygons(
    array: &MultiPolygonArray,
    keep: Keep,
) -> impl Iterator<Item = Range<usize>> + Clone + '_ {
    let polygons = &array.polygons;
    array
        .rows
        .span(0..array.rows.len())
        .map(move |j| keep(polygons.range(j)))
}

/// Offsets of lists of lengths `lengths`.
fn offsets_of(lengths: impl Iterator<Item = usize>) -> OffsetsBuffer<i64> {
    Offsets::try_from_lengths(lengths)
        .expect("no more items than the input")
        .into()
}

/// One linestring per item of `kept` (a single ring or none),
/// with its row missing where `validity` says.
#[inline(always)]
fn exteriors(
    rings: &ValidList,
    coords: &ValidCoords,
    kept: impl Iterator<Item = Range<usize>> + Clone,
    validity: Option<Bitmap>,
) -> LineStringArray {
    // The output cannot reuse the input's coordinate buffers.
    // In an Arrow list, each list ends where the next one starts,
    // meaning there is no way to leave out the dropped rings in between.
    // Instead, the coordinates of the kept rings are copied into new buffers.
    // The rings a polygon keeps (its exterior, or all of its holes) sit next to each other.
    // So for each polygon, its kept coordinates are copied as one slice per axis.
    // This is much faster than Polars' own `take` on the rings,
    // which copies ring by ring (~1.5 µs per ring).
    let spans = kept.map(|kept| rings.span(kept));
    LineStringArray {
        rows: List::from_offsets(offsets_of(spans.clone().map(|span| span.len())), validity),
        coords: coords.copy_ranges(spans),
    }
}

/// One multilinestring per item of `kept`,
/// with its row missing where `validity` says.
#[inline(always)]
fn interiors(
    rings: &ValidList,
    coords: &ValidCoords,
    kept: impl Iterator<Item = Range<usize>> + Clone,
    validity: Option<Bitmap>,
) -> MultiLineStringArray {
    let ring_lengths = kept
        .clone()
        .flat_map(|kept| kept.map(|ring| rings.range(ring).len()));
    MultiLineStringArray {
        rows: List::from_offsets(offsets_of(kept.clone().map(|kept| kept.len())), validity),
        lines: ValidList::from_offsets(offsets_of(ring_lengths)),
        // Copied as in `exteriors`.
        coords: coords.copy_ranges(kept.map(|kept| rings.span(kept))),
    }
}

/// `parts`, one list per row of `rows`.
fn per_multi_polygon(rows: &List, parts: ListArray<i64>) -> ListArray<i64> {
    let lengths = (0..rows.len()).map(|i| rows.range(i).len());
    list_array(parts.boxed(), offsets_of(lengths), rows.validity().cloned())
}

pub fn polygon_exteriors(array: &PolygonArray) -> LineStringArray {
    let validity = array.rows.validity().cloned();
    if array.rows.is_all_valid() {
        let kept = kept_rings_of_polygons::<true>(array, exterior_of);
        exteriors(&array.rings, &array.coords, kept, validity)
    } else {
        let kept = kept_rings_of_polygons::<false>(array, exterior_of);
        exteriors(&array.rings, &array.coords, kept, validity)
    }
}

pub fn polygon_interiors(array: &PolygonArray) -> MultiLineStringArray {
    let validity = array.rows.validity().cloned();
    if array.rows.is_all_valid() {
        let kept = kept_rings_of_polygons::<true>(array, interior_of);
        interiors(&array.rings, &array.coords, kept, validity)
    } else {
        let kept = kept_rings_of_polygons::<false>(array, interior_of);
        interiors(&array.rings, &array.coords, kept, validity)
    }
}

/// A list of linestrings per row, one per polygon.
pub fn multi_polygon_exteriors(array: &MultiPolygonArray) -> ListArray<i64> {
    let kept = kept_rings_of_multi_polygons(array, exterior_of);
    let parts = exteriors(&array.rings, &array.coords, kept, None);
    per_multi_polygon(&array.rows, parts.into_list_array())
}

/// A list of multilinestrings per row, one per polygon.
pub fn multi_polygon_interiors(array: &MultiPolygonArray) -> ListArray<i64> {
    let kept = kept_rings_of_multi_polygons(array, interior_of);
    let parts = interiors(&array.rings, &array.coords, kept, None);
    per_multi_polygon(&array.rows, parts.into_list_array())
}

/// Makes a series of coordinates a closed ring by repeating the last vertex.
/// `None` if they are all closed already.
/// An empty ring is left empty.
pub fn close_open_rings<const Z: bool>(
    rings: &ValidList,
    coords: &ValidCoords,
) -> Option<LineStringArray> {
    // Checked again in every pass over the rings, rather than stored.
    let is_open = |ring: usize| {
        let vertices = rings.range(ring);
        !vertices.is_empty() && !same_place::<Z>(coords, vertices.start, vertices.end - 1)
    };
    if !(0..rings.len()).any(is_open) {
        return None;
    }
    let lengths = (0..rings.len()).map(|ring| rings.range(ring).len() + usize::from(is_open(ring)));
    let ranges = (0..rings.len()).flat_map(|ring| {
        let vertices = rings.range(ring);
        let first = is_open(ring).then(|| vertices.start..vertices.start + 1);
        std::iter::once(vertices).chain(first)
    });
    Some(LineStringArray {
        rows: List::from_offsets(offsets_of(lengths), None),
        coords: coords.copy_ranges(ranges),
    })
}
