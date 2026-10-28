//! What to do with the selected coordinates of a geometry,
//! shared by `filter_coordinates` and `fix_out_of_bounds`.
//! See [SelectedAction] for the actions.
//!
//! [SelectedAction::Drop] copies the kept coordinates once, a run of them at a time.
//! No part is ever removed, so only the innermost list level gets new offsets.
//! A linestring or ring left [too short](is_too_short) is handled by [TooShort].
//! A point cannot lose its coordinate, so it becomes the empty point instead.
//! The other actions keep every offset.
//! A null row is processed like any other, as its value is masked anyway.
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

/// See `action` of `NonFinite` and `OutOfBounds` in Python.
#[derive(Debug, Clone, Copy)]
pub enum SelectedAction {
    /// Every axis of a selected coordinate becomes NaN.
    SetEmpty,
    /// A selected coordinate is removed.
    Drop,
    /// A row holding a selected coordinate becomes null.
    SetNull,
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
/// Last, the lists that are too short with [TooShort::Null], which keep none.
fn without_selected<const Z: bool>(
    lists: &ValidList,
    kind: Lists,
    coords: &ValidCoords,
    selected: &[bool],
    too_short: TooShort,
) -> (OffsetsBuffer<i64>, ValidCoords, Vec<usize>) {
    let (mut runs, mut nulled) = (Vec::new(), Vec::new());
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
                TooShort::Null => {
                    runs.truncate(first_run);
                    nulled.push(j);
                    0
                }
            }
        })
        .collect();
    (
        offsets_of(lengths.into_iter()),
        coords.copy_ranges(runs.into_iter()),
        nulled,
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

fn line_strings_without_selected(
    array: &LineStringArray,
    selected: &[bool],
    too_short: TooShort,
) -> LineStringArray {
    let (offsets, coords, nulled) = without_selected::<false>(
        array.rows.ignoring_validity(),
        Lists::Lines,
        &array.coords,
        selected,
        too_short,
    );
    let validity = validity_without(array.rows.validity(), array.rows.len(), &nulled);
    LineStringArray {
        rows: List::from_offsets(offsets, validity),
        coords,
    }
}

fn multi_points_without_selected(
    array: &MultiPointArray,
    selected: &[bool],
    too_short: TooShort,
) -> MultiPointArray {
    let (offsets, coords, _) = without_selected::<false>(
        array.rows.ignoring_validity(),
        Lists::Points,
        &array.coords,
        selected,
        too_short,
    );
    MultiPointArray {
        rows: List::from_offsets(offsets, array.rows.validity().cloned()),
        coords,
    }
}

fn polygons_without_selected(
    array: &PolygonArray,
    selected: &[bool],
    too_short: TooShort,
) -> PolygonArray {
    let (rings, coords, nulled) =
        without_selected_in_rings(&array.rings, &array.coords, selected, too_short);
    let failed: Vec<usize> = nulled
        .into_iter()
        .filter_map(|ring| list_holding(array.rows.ignoring_validity(), ring))
        .collect();
    PolygonArray {
        rows: without_rows(&array.rows, &failed),
        rings: ValidList::from_offsets(rings),
        coords,
    }
}

fn multi_line_strings_without_selected(
    array: &MultiLineStringArray,
    selected: &[bool],
    too_short: TooShort,
) -> MultiLineStringArray {
    let (lines, coords, nulled) = without_selected::<false>(
        &array.lines,
        Lists::Lines,
        &array.coords,
        selected,
        too_short,
    );
    let failed: Vec<usize> = nulled
        .into_iter()
        .filter_map(|line| list_holding(array.rows.ignoring_validity(), line))
        .collect();
    MultiLineStringArray {
        rows: without_rows(&array.rows, &failed),
        lines: ValidList::from_offsets(lines),
        coords,
    }
}

fn multi_polygons_without_selected(
    array: &MultiPolygonArray,
    selected: &[bool],
    too_short: TooShort,
) -> MultiPolygonArray {
    let (rings, coords, nulled) =
        without_selected_in_rings(&array.rings, &array.coords, selected, too_short);
    let failed: Vec<usize> = nulled
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

/// The rows of `len` rows that hold a selected coordinate.
/// `coords_of` gives the coordinates of a row.
fn rows_with_selected(
    len: usize,
    coords_of: impl Fn(usize) -> Range<usize>,
    selected: &[bool],
) -> Vec<usize> {
    (0..len)
        .filter(|&i| selected[coords_of(i)].contains(&true))
        .collect()
}

/// [SelectedAction::Drop] leaves the empty point, as a point cannot lose its coordinate.
pub(super) fn points_with_selected(
    array: &PointArray,
    selected: &[bool],
    action: SelectedAction,
) -> PointArray {
    let (coords, validity) = (array.coords.ignoring_validity(), array.coords.validity());
    PointArray::new(match action {
        SelectedAction::SetEmpty | SelectedAction::Drop => {
            Coords::from_valid(emptied(coords, selected), validity.cloned())
        }
        SelectedAction::SetNull => {
            let failed = rows_with_selected(coords.len(), |i| i..i + 1, selected);
            Coords::from_valid(
                coords.clone(),
                validity_without(validity, coords.len(), &failed),
            )
        }
    })
}

pub(super) fn line_strings_with_selected(
    array: &LineStringArray,
    selected: &[bool],
    action: SelectedAction,
    too_short: TooShort,
) -> LineStringArray {
    match action {
        SelectedAction::SetEmpty => LineStringArray {
            rows: array.rows.clone(),
            coords: emptied(&array.coords, selected),
        },
        SelectedAction::Drop => line_strings_without_selected(array, selected, too_short),
        SelectedAction::SetNull => {
            let failed = rows_with_selected(array.rows.len(), |i| array.rows.range(i), selected);
            LineStringArray {
                rows: without_rows(&array.rows, &failed),
                coords: array.coords.clone(),
            }
        }
    }
}

pub(super) fn multi_points_with_selected(
    array: &MultiPointArray,
    selected: &[bool],
    action: SelectedAction,
    too_short: TooShort,
) -> MultiPointArray {
    match action {
        SelectedAction::SetEmpty => MultiPointArray {
            rows: array.rows.clone(),
            coords: emptied(&array.coords, selected),
        },
        SelectedAction::Drop => multi_points_without_selected(array, selected, too_short),
        SelectedAction::SetNull => {
            let failed = rows_with_selected(array.rows.len(), |i| array.rows.range(i), selected);
            MultiPointArray {
                rows: without_rows(&array.rows, &failed),
                coords: array.coords.clone(),
            }
        }
    }
}

pub(super) fn polygons_with_selected(
    array: &PolygonArray,
    selected: &[bool],
    action: SelectedAction,
    too_short: TooShort,
) -> PolygonArray {
    match action {
        SelectedAction::SetEmpty => PolygonArray {
            rows: array.rows.clone(),
            rings: array.rings.clone(),
            coords: emptied(&array.coords, selected),
        },
        SelectedAction::Drop => polygons_without_selected(array, selected, too_short),
        SelectedAction::SetNull => {
            let coords_of = |i| array.rings.span(array.rows.range(i));
            let failed = rows_with_selected(array.rows.len(), coords_of, selected);
            PolygonArray {
                rows: without_rows(&array.rows, &failed),
                rings: array.rings.clone(),
                coords: array.coords.clone(),
            }
        }
    }
}

pub(super) fn multi_line_strings_with_selected(
    array: &MultiLineStringArray,
    selected: &[bool],
    action: SelectedAction,
    too_short: TooShort,
) -> MultiLineStringArray {
    match action {
        SelectedAction::SetEmpty => MultiLineStringArray {
            rows: array.rows.clone(),
            lines: array.lines.clone(),
            coords: emptied(&array.coords, selected),
        },
        SelectedAction::Drop => multi_line_strings_without_selected(array, selected, too_short),
        SelectedAction::SetNull => {
            let coords_of = |i| array.lines.span(array.rows.range(i));
            let failed = rows_with_selected(array.rows.len(), coords_of, selected);
            MultiLineStringArray {
                rows: without_rows(&array.rows, &failed),
                lines: array.lines.clone(),
                coords: array.coords.clone(),
            }
        }
    }
}

pub(super) fn multi_polygons_with_selected(
    array: &MultiPolygonArray,
    selected: &[bool],
    action: SelectedAction,
    too_short: TooShort,
) -> MultiPolygonArray {
    match action {
        SelectedAction::SetEmpty => MultiPolygonArray {
            rows: array.rows.clone(),
            polygons: array.polygons.clone(),
            rings: array.rings.clone(),
            coords: emptied(&array.coords, selected),
        },
        SelectedAction::Drop => multi_polygons_without_selected(array, selected, too_short),
        SelectedAction::SetNull => {
            let coords_of = |i| array.rings.span(array.polygons.span(array.rows.range(i)));
            let failed = rows_with_selected(array.rows.len(), coords_of, selected);
            MultiPolygonArray {
                rows: without_rows(&array.rows, &failed),
                polygons: array.polygons.clone(),
                rings: array.rings.clone(),
                coords: array.coords.clone(),
            }
        }
    }
}
