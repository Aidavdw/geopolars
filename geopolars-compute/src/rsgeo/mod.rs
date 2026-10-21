//! Interfacing with the `rsgeo` crate (the Rust `geo` crate).
//!
//! rsgeo has geometry types of its own, so what it relates is copied into them.
//! Its algorithms only look at x and y, so z and m are never read.

mod is_valid;
mod rings_overlap;
mod self_intersects;

pub use is_valid::{multi_polygons_fit, polygons_fit, FitChecks};
pub use rings_overlap::{rings_overlap_against_one, rings_overlap_pairwise};
pub use self_intersects::{
    line_strings_self_intersect, multi_line_strings_self_intersect, multi_polygons_self_intersect,
    polygons_self_intersect, vertices_self_intersect,
};

use ::rsgeo::{Coord, LineString, Polygon};
use geopolars_core::output::{bool_array, list_array};
use geopolars_core::{List, ScalarLineString, ValidList};
use polars_arrow::array::{BooleanArray, ListArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;

use crate::native::{is_ring, per_row_values};
use crate::{GeoComputeError, GeoComputeResult};

/// Whether every vertex of `line` is finite in x and y.
#[inline(always)]
fn xy_is_finite(line: &ScalarLineString) -> bool {
    let x = &line.coords.x()[line.vertices.clone()];
    let y = &line.coords.y()[line.vertices.clone()];
    x.iter().chain(y).all(|value| value.is_finite())
}

/// Whether `line` is finite in x and y.
/// Errors if it is, but is not a ring.
#[inline(always)]
fn ring_is_finite(line: &ScalarLineString) -> GeoComputeResult<bool> {
    if !xy_is_finite(line) {
        return Ok(false);
    }
    if !is_ring::<false>(line.coords, line.vertices.clone()) {
        return Err(GeoComputeError::NotARing {
            vertices: line.vertices.len(),
        });
    }
    Ok(true)
}

/// The polygon the ring `line` encloses, in x and y.
/// This is the only copy made of its vertices.
fn to_rsgeo_ring(line: &ScalarLineString) -> Polygon<f64> {
    let vertices = line
        .positions::<false>()
        .map(|[x, y, _]| Coord { x, y })
        .collect();
    Polygon::new(LineString::new(vertices), vec![])
}

/// A `bool` per item, missing where it is `None`.
fn optional_bool_array(tested: &[Option<bool>]) -> BooleanArray {
    let values = tested
        .iter()
        .map(|tested| tested.unwrap_or(false))
        .collect();
    let validity = Bitmap::from_iter(tested.iter().map(Option::is_some));
    bool_array(values, Some(&validity))
}

/// `test` of every row of a chunk, missing where the row is or where `test` gives `None`.
#[inline(always)]
fn per_row_optional_bool<E>(
    rows: &List,
    test: impl Fn(usize) -> Result<Option<bool>, E>,
) -> Result<BooleanArray, E> {
    Ok(optional_bool_array(&per_row_values(rows, test)?))
}

/// [per_row_optional_bool], as a list per row of a chunk, one per part.
/// `parts` holds where the parts of each row sit, and `test` takes the index of a part.
fn per_part_optional_bool<E>(
    rows: &List,
    parts: &ValidList,
    test: impl Fn(usize) -> Result<Option<bool>, E>,
) -> Result<ListArray<i64>, E> {
    let mut tested = Vec::with_capacity(parts.len());
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it gets no parts.
        if rows.is_valid(i) {
            for part in rows.range(i) {
                tested.push(test(part)?);
            }
        }
        offsets.push(tested.len() as i64);
    }
    // SAFETY: they start at 0, and every end is the number of values so far, which only grows.
    let offsets = unsafe { Offsets::new_unchecked(offsets) }.into();
    Ok(list_array(
        optional_bool_array(&tested).boxed(),
        offsets,
        rows.validity().cloned(),
    ))
}
