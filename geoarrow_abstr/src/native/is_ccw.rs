//! Whether a geometry runs counter-clockwise in x and y.
//! m and z don't really matter here.

use super::area::twice_signed_area;
use crate::arrays::per_row_bool;
use crate::{
    GeoArrowBoolArray, GeoArrowBoolListArray, GeoArrowListView, GeoArrowValidCoordsView,
    GeoArrowValidListView, LineStringArrayView, LineStringView, MultiLineStringArrayView,
    MultiPolygonArrayView, PolygonArrayView, PolygonView,
};

/// Whether `ring` runs counter-clockwise, or clockwise if `CW=true`.
/// An empty ring runs either way.
/// Any other ring that encloses nothing runs neither way.
#[inline(always)]
fn ring_runs<const CW: bool>(ring: &LineStringView<impl GeoArrowValidCoordsView>) -> bool {
    if ring.vertices.is_empty() {
        return true;
    }
    let winding = twice_signed_area(ring);
    if CW {
        winding < 0.0
    } else {
        winding > 0.0
    }
}

/// Whether `polygon` has its exterior counter-clockwise, and its holes clockwise.
/// If `HOLES_CW=false`, its holes have to run counter-clockwise instead.
/// No rings at all is the empty polygon, which runs either way.
#[inline(always)]
fn polygon_is_ccw<const HOLES_CW: bool>(
    polygon: &PolygonView<impl GeoArrowValidListView, impl GeoArrowValidCoordsView>,
) -> bool {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return true;
    };
    ring_runs::<false>(&exterior) && rings.all(|hole| ring_runs::<HOLES_CW>(&hole))
}

/// One list per row of a chunk, holding `parts_of(i)` for row `i`.
/// A missing row gets an empty list.
#[inline(always)]
fn per_row_parts<O: GeoArrowBoolListArray, P: Iterator<Item = bool>>(
    rows: &impl GeoArrowListView<Validity = O::Validity>,
    parts_of: impl Fn(usize) -> P,
) -> O {
    let mut values = Vec::new();
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not looked at.
        if rows.is_valid(i) {
            values.extend(parts_of(i));
        }
        offsets.push(values.len() as i64);
    }
    O::from_buffers(values, offsets, rows.validity())
}

pub fn line_strings_are_ccw<O: GeoArrowBoolArray>(
    array: &LineStringArrayView<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidCoordsView,
    >,
) -> O {
    per_row_bool(&array.rows, |i| ring_runs::<false>(&array.get(i)))
}

/// See [polygon_is_ccw] for `HOLES_CW`.
pub fn polygons_are_ccw<const HOLES_CW: bool, O: GeoArrowBoolArray>(
    array: &PolygonArrayView<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidListView,
        impl GeoArrowValidCoordsView,
    >,
) -> O {
    per_row_bool(&array.rows, |i| polygon_is_ccw::<HOLES_CW>(&array.get(i)))
}

/// One per linestring of every multilinestring of a chunk.
pub fn multi_line_strings_are_ccw<O: GeoArrowBoolListArray>(
    array: &MultiLineStringArrayView<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidListView,
        impl GeoArrowValidCoordsView,
    >,
) -> O {
    per_row_parts(&array.rows, |i| {
        array
            .get(i)
            .line_strings()
            .map(|line| ring_runs::<false>(&line))
    })
}

/// One per polygon of every multipolygon of a chunk.
/// See [polygon_is_ccw] for `HOLES_CW`.
pub fn multi_polygons_are_ccw<const HOLES_CW: bool, O: GeoArrowBoolListArray>(
    array: &MultiPolygonArrayView<
        impl GeoArrowListView<Validity = O::Validity>,
        impl GeoArrowValidListView,
        impl GeoArrowValidListView,
        impl GeoArrowValidCoordsView,
    >,
) -> O {
    per_row_parts(&array.rows, |i| {
        array
            .get(i)
            .polygons()
            .map(|polygon| polygon_is_ccw::<HOLES_CW>(&polygon))
    })
}
