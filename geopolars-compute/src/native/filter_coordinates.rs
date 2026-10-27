//! Emptying or dropping the coordinates that are not finite, or empty.
//! See `filter_coordinates` in Python for the rules.
//!
//! Every kernel first marks the selected coordinates ([selected_coords]).
//! With [NonFiniteAction::SetEmpty], it then only rewrites the axes, and keeps every offset.
//! With [NonFiniteAction::Drop], it copies the kept coordinates once, a run of them at a time.
//! No part is ever removed, so only the innermost list level gets new offsets.
//! A missing row is processed like any other, as its value is masked anyway.
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! [without_selected] is generic over `Z`, to check whether a ring is still closed.
//! See [same_place] for `Z`.
//! Only the kernels of rings dispatch on it.

use std::ops::Range;

use geopolars_core::output::validity_without;
use geopolars_core::{
    same_place, Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ValidCoords, ValidList, MIN_RING_VERTICES,
};
use polars_arrow::offset::OffsetsBuffer;

use super::rings::offsets_of;
use super::too_short::{list_holding, without_rows, TooShort};

/// See `action` of `NonFinite` in Python.
#[derive(Debug, Clone, Copy)]
pub enum NonFiniteAction {
    SetEmpty,
    Drop,
}

/// See `NonFinite` in Python.
#[derive(Debug, Clone, Copy)]
pub struct NonFinite {
    pub select_non_finite: bool,
    pub select_empty: bool,
    pub ignore_z: bool,
    pub ignore_m: bool,
    pub action: NonFiniteAction,
    pub too_short: TooShort,
}

/// Whether `filter` selects each of `coords`.
fn selected_coords(coords: &ValidCoords, filter: &NonFinite) -> Vec<bool> {
    let (x, y) = (coords.x(), coords.y());
    let mut non_finite: Vec<bool> = x
        .iter()
        .zip(y)
        .map(|(x, y)| !(x.is_finite() & y.is_finite()))
        .collect();
    let mut empty: Vec<bool> = x
        .iter()
        .zip(y)
        .map(|(x, y)| x.is_nan() & y.is_nan())
        .collect();
    for (axis, ignored) in [(coords.z(), filter.ignore_z), (coords.m(), filter.ignore_m)] {
        let Some(axis) = axis else { continue };
        for ((non_finite, empty), value) in non_finite.iter_mut().zip(&mut empty).zip(axis) {
            *non_finite |= !ignored & !value.is_finite();
            *empty &= value.is_nan();
        }
    }
    let NonFinite {
        select_non_finite,
        select_empty,
        ..
    } = *filter;
    non_finite
        .iter()
        .zip(&empty)
        .map(|(&non_finite, &empty)| {
            (select_non_finite & non_finite & !empty) | (select_empty & empty)
        })
        .collect()
}

/// `coords`, with every axis NaN where `selected`.
fn emptied(coords: &ValidCoords, selected: &[bool]) -> ValidCoords {
    let emptied = |axis: &[f64]| -> Vec<f64> {
        axis.iter()
            .zip(selected)
            .map(|(&value, &selected)| if selected { f64::NAN } else { value })
            .collect()
    };
    ValidCoords::from_axes((
        emptied(coords.x()),
        emptied(coords.y()),
        coords.z().map(&emptied),
        coords.m().map(&emptied),
    ))
}

/// What the lists of vertices handed to [without_selected] are.
#[derive(Clone, Copy)]
enum Lists {
    /// The points of a multipoint, which are never too short.
    Points,
    Lines,
    Rings,
}

/// Pushes the runs of `vertices` that are not `selected` onto `runs`,
/// and returns how many vertices they hold.
#[inline(always)]
fn push_kept_runs(
    selected: &[bool],
    vertices: Range<usize>,
    runs: &mut Vec<Range<usize>>,
) -> usize {
    let (mut kept, mut run_start) = (0, vertices.start);
    for i in vertices.clone() {
        if selected[i] {
            if run_start < i {
                runs.push(run_start..i);
            }
            run_start = i + 1;
        } else {
            kept += 1;
        }
    }
    if run_start < vertices.end {
        runs.push(run_start..vertices.end);
    }
    kept
}

/// Whether the `kept` vertices in `runs` are too short for `lists`:
/// a linestring of a single vertex,
/// or a ring with vertices that is not closed or has fewer than [MIN_RING_VERTICES].
#[inline(always)]
fn is_too_short<const Z: bool>(
    coords: &ValidCoords,
    lists: Lists,
    runs: &[Range<usize>],
    kept: usize,
) -> bool {
    match lists {
        Lists::Points => false,
        Lists::Lines => kept == 1,
        Lists::Rings => match (runs.first(), runs.last()) {
            (Some(first), Some(last)) => {
                kept < MIN_RING_VERTICES || !same_place::<Z>(coords, first.start, last.end - 1)
            }
            _ => false,
        },
    }
}

/// The offsets of what each of `lists` keeps, and the coordinates it keeps.
/// A list keeps the vertices that are not `selected`, unless it is then [too short](is_too_short).
/// Last, the lists that are too short with [TooShort::Invalid], which keep none.
fn without_selected<const Z: bool>(
    lists: &ValidList,
    kind: Lists,
    coords: &ValidCoords,
    selected: &[bool],
    too_short: TooShort,
) -> (OffsetsBuffer<i64>, ValidCoords, Vec<usize>) {
    let (mut runs, mut invalid) = (Vec::new(), Vec::new());
    let lengths: Vec<usize> = lists
        .ranges()
        .enumerate()
        .map(|(j, list)| {
            let first_run = runs.len();
            let kept = push_kept_runs(selected, list, &mut runs);
            if !is_too_short::<Z>(coords, kind, &runs[first_run..], kept) {
                return kept;
            }
            match too_short {
                TooShort::Keep => kept,
                TooShort::Empty => {
                    runs.truncate(first_run);
                    0
                }
                TooShort::Invalid => {
                    runs.truncate(first_run);
                    invalid.push(j);
                    0
                }
            }
        })
        .collect();
    (
        offsets_of(lengths.into_iter()),
        coords.copy_ranges(runs.into_iter()),
        invalid,
    )
}

/// [without_selected] for the rings of `coords`.
fn without_selected_in_rings(
    rings: &ValidList,
    coords: &ValidCoords,
    selected: &[bool],
    too_short: TooShort,
) -> (OffsetsBuffer<i64>, ValidCoords, Vec<usize>) {
    let z = coords.dim().has_z();
    with_const_flags!(without_selected(rings, Lists::Rings, coords, selected, too_short); z)
}

/// A dropped point leaves the empty point, as a point cannot lose its coordinate.
pub fn points_with_filtered_coordinates(array: &PointArray, filter: &NonFinite) -> PointArray {
    let coords = array.coords.ignoring_validity();
    let selected = selected_coords(coords, filter);
    PointArray::new(Coords::from_valid(
        emptied(coords, &selected),
        array.coords.validity().cloned(),
    ))
}

pub fn line_strings_with_filtered_coordinates(
    array: &LineStringArray,
    filter: &NonFinite,
) -> LineStringArray {
    let selected = selected_coords(&array.coords, filter);
    match filter.action {
        NonFiniteAction::SetEmpty => LineStringArray {
            rows: array.rows.clone(),
            coords: emptied(&array.coords, &selected),
        },
        NonFiniteAction::Drop => {
            let (offsets, coords, invalid) = without_selected::<false>(
                array.rows.ignoring_validity(),
                Lists::Lines,
                &array.coords,
                &selected,
                filter.too_short,
            );
            let validity = validity_without(array.rows.validity(), array.rows.len(), &invalid);
            LineStringArray {
                rows: List::from_offsets(offsets, validity),
                coords,
            }
        }
    }
}

pub fn multi_points_with_filtered_coordinates(
    array: &MultiPointArray,
    filter: &NonFinite,
) -> MultiPointArray {
    let selected = selected_coords(&array.coords, filter);
    match filter.action {
        NonFiniteAction::SetEmpty => MultiPointArray {
            rows: array.rows.clone(),
            coords: emptied(&array.coords, &selected),
        },
        NonFiniteAction::Drop => {
            let (offsets, coords, _) = without_selected::<false>(
                array.rows.ignoring_validity(),
                Lists::Points,
                &array.coords,
                &selected,
                filter.too_short,
            );
            MultiPointArray {
                rows: List::from_offsets(offsets, array.rows.validity().cloned()),
                coords,
            }
        }
    }
}

pub fn polygons_with_filtered_coordinates(
    array: &PolygonArray,
    filter: &NonFinite,
) -> PolygonArray {
    let selected = selected_coords(&array.coords, filter);
    match filter.action {
        NonFiniteAction::SetEmpty => PolygonArray {
            rows: array.rows.clone(),
            rings: array.rings.clone(),
            coords: emptied(&array.coords, &selected),
        },
        NonFiniteAction::Drop => {
            let (rings, coords, invalid) =
                without_selected_in_rings(&array.rings, &array.coords, &selected, filter.too_short);
            let failed: Vec<usize> = invalid
                .into_iter()
                .filter_map(|ring| list_holding(array.rows.ignoring_validity(), ring))
                .collect();
            PolygonArray {
                rows: without_rows(&array.rows, &failed),
                rings: ValidList::from_offsets(rings),
                coords,
            }
        }
    }
}

pub fn multi_line_strings_with_filtered_coordinates(
    array: &MultiLineStringArray,
    filter: &NonFinite,
) -> MultiLineStringArray {
    let selected = selected_coords(&array.coords, filter);
    match filter.action {
        NonFiniteAction::SetEmpty => MultiLineStringArray {
            rows: array.rows.clone(),
            lines: array.lines.clone(),
            coords: emptied(&array.coords, &selected),
        },
        NonFiniteAction::Drop => {
            let (lines, coords, invalid) = without_selected::<false>(
                &array.lines,
                Lists::Lines,
                &array.coords,
                &selected,
                filter.too_short,
            );
            let failed: Vec<usize> = invalid
                .into_iter()
                .filter_map(|line| list_holding(array.rows.ignoring_validity(), line))
                .collect();
            MultiLineStringArray {
                rows: without_rows(&array.rows, &failed),
                lines: ValidList::from_offsets(lines),
                coords,
            }
        }
    }
}

pub fn multi_polygons_with_filtered_coordinates(
    array: &MultiPolygonArray,
    filter: &NonFinite,
) -> MultiPolygonArray {
    let selected = selected_coords(&array.coords, filter);
    match filter.action {
        NonFiniteAction::SetEmpty => MultiPolygonArray {
            rows: array.rows.clone(),
            polygons: array.polygons.clone(),
            rings: array.rings.clone(),
            coords: emptied(&array.coords, &selected),
        },
        NonFiniteAction::Drop => {
            let (rings, coords, invalid) =
                without_selected_in_rings(&array.rings, &array.coords, &selected, filter.too_short);
            let failed: Vec<usize> = invalid
                .into_iter()
                .filter_map(|ring| list_holding(&array.polygons, ring))
                .filter_map(|polygon| list_holding(array.rows.ignoring_validity(), polygon))
                .collect();
            MultiPolygonArray {
                rows: without_rows(&array.rows, &failed),
                polygons: array.polygons.clone(),
                rings: ValidList::from_offsets(rings),
                coords,
            }
        }
    }
}
