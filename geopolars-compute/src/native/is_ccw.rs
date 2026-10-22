//! Whether a geometry runs counter-clockwise in x and y.
//! m and z don't really matter here.

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray, ScalarLineString,
    ScalarPolygon,
};
use polars_arrow::array::BooleanArray;

use super::area::twice_signed_area;
use crate::per_row::per_row_bool;

/// Whether `ring` runs counter-clockwise, or clockwise if `CW=true`.
/// An empty ring runs either way.
/// Any other ring that encloses nothing runs neither way.
#[inline(always)]
fn ring_runs<const CW: bool>(ring: &ScalarLineString) -> bool {
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
fn polygon_is_ccw<const HOLES_CW: bool>(polygon: &ScalarPolygon) -> bool {
    let mut rings = polygon.rings();
    let Some(exterior) = rings.next() else {
        return true;
    };
    ring_runs::<false>(&exterior) && rings.all(|hole| ring_runs::<HOLES_CW>(&hole))
}

pub fn line_strings_are_ccw(array: &LineStringArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| ring_runs::<false>(&array.get(i)))
}

/// See [polygon_is_ccw] for `HOLES_CW`.
pub fn polygons_are_ccw<const HOLES_CW: bool>(array: &PolygonArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| polygon_is_ccw::<HOLES_CW>(&array.get(i)))
}

/// Whether every linestring of a multilinestring runs counter-clockwise, for every row of a chunk.
/// No linestrings at all runs either way.
pub fn multi_line_strings_are_ccw(array: &MultiLineStringArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .get(i)
            .line_strings()
            .all(|line| ring_runs::<false>(&line))
    })
}

/// Whether every polygon of a multipolygon [is CCW](polygon_is_ccw), for every row of a chunk.
/// No polygons at all runs either way.
pub fn multi_polygons_are_ccw<const HOLES_CW: bool>(array: &MultiPolygonArray) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .get(i)
            .polygons()
            .all(|polygon| polygon_is_ccw::<HOLES_CW>(&polygon))
    })
}
