//! What to do with the selected coordinates of a geometry,
//! shared by `filter_coordinates` and `fix_out_of_bounds`.
//! Every kernel marks the coordinates its [CoordinateSelection] selects ([selected_coords]),
//! then acts on them. See [SelectedAction] for the actions.
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
//! ## `CHECK_CLOSED`
//!
//! [without_selected] is generic over `CHECK_CLOSED`.
//! If `CHECK_CLOSED=true`, a ring that is not closed after the drop is too short.
//! If `CHECK_CLOSED=false`, the caller guarantees each ring keeps both its ends,
//! so it is as closed as it came in.
//! Only the kernels of rings dispatch on it.
//!
//! ## `Z`
//!
//! [without_selected] is generic over `Z`, to check whether a ring is still closed.
//! See [same_place] for `Z`.
//! Only the kernels of rings with `CHECK_CLOSED=true` dispatch on it.

use std::ops::Range;

use geopolars_core::output::validity_without;
use geopolars_core::{
    same_place, Coords, LineStringArray, List, MultiLineStringArray, MultiPointArray,
    MultiPolygonArray, PointArray, PolygonArray, ValidCoords, ValidList, MIN_RING_VERTICES,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::offset::OffsetsBuffer;

use super::coordinate_selection::{selected_coords, which_are_not_selected, CoordinateSelection};
use super::rings::offsets_of;
use super::too_short::{list_holding, without_rows, TooShort};
use crate::passing::valid_rows_passing;

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

/// [selected_coords], with no bit set when `selection` selects nothing.
fn selected_coords_bitmap(coords: &ValidCoords, selection: &CoordinateSelection) -> Bitmap {
    selected_coords(coords, selection).unwrap_or_else(|| Bitmap::new_zeroed(coords.len()))
}

/// `coords`, with every axis NaN where `selected`.
fn emptied(coords: &ValidCoords, selected: &Bitmap) -> ValidCoords {
    let emptied = |axis: &[f64]| -> Vec<f64> {
        axis.iter()
            .zip(selected.iter())
            .map(|(&value, selected)| if selected { f64::NAN } else { value })
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
    selected: &Bitmap,
    vertices: Range<usize>,
    runs: &mut Vec<Range<usize>>,
) -> usize {
    // Selected vertices are rare, so most lists are kept whole.
    if selected.null_count_range(vertices.start, vertices.len()) == vertices.len() {
        if !vertices.is_empty() {
            runs.push(vertices.clone());
        }
        return vertices.len();
    }
    let (mut kept, mut run_start) = (0, vertices.start);
    for i in vertices.clone() {
        if selected.get_bit(i) {
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
/// or a ring with vertices that has fewer than [MIN_RING_VERTICES], or is not closed.
#[inline(always)]
fn is_too_short<const CHECK_CLOSED: bool, const Z: bool>(
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
                kept < MIN_RING_VERTICES
                    || (CHECK_CLOSED && !same_place::<Z>(coords, first.start, last.end - 1))
            }
            _ => false,
        },
    }
}

/// The offsets of what each of `lists` keeps, and the coordinates it keeps.
/// A list keeps the vertices that are not `selected`, unless it is then [too short](is_too_short).
/// Last, the lists that are too short with [TooShort::Null], which keep none.
fn without_selected<const CHECK_CLOSED: bool, const Z: bool>(
    lists: &ValidList,
    kind: Lists,
    coords: &ValidCoords,
    selected: &Bitmap,
    too_short: TooShort,
) -> (OffsetsBuffer<i64>, ValidCoords, Vec<usize>) {
    let (mut runs, mut nulled) = (Vec::new(), Vec::new());
    let lengths: Vec<usize> = lists
        .ranges()
        .enumerate()
        .map(|(j, list)| {
            let first_run = runs.len();
            let kept = push_kept_runs(selected, list, &mut runs);
            if !is_too_short::<CHECK_CLOSED, Z>(coords, kind, &runs[first_run..], kept) {
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
    selected: &Bitmap,
    too_short: TooShort,
    check_closed: bool,
) -> (OffsetsBuffer<i64>, ValidCoords, Vec<usize>) {
    let kind = Lists::Rings;
    // `Z` only matters for the closure check, so it is not dispatched on without it.
    if check_closed {
        with_const_flags!(
            Z = coords.dim().has_z();
            without_selected::<true, Z>(rings, kind, coords, selected, too_short)
        )
    } else {
        without_selected::<false, false>(rings, kind, coords, selected, too_short)
    }
}

pub(super) fn line_strings_without_selected(
    array: &LineStringArray,
    selected: &Bitmap,
    too_short: TooShort,
) -> LineStringArray {
    let (offsets, coords, nulled) = without_selected::<false, false>(
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

pub(super) fn multi_points_without_selected(
    array: &MultiPointArray,
    selected: &Bitmap,
    too_short: TooShort,
) -> MultiPointArray {
    let (offsets, coords, _) = without_selected::<false, false>(
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

pub(super) fn polygons_without_selected(
    array: &PolygonArray,
    selected: &Bitmap,
    too_short: TooShort,
    check_closed: bool,
) -> PolygonArray {
    let (rings, coords, nulled) = without_selected_in_rings(
        &array.rings,
        &array.coords,
        selected,
        too_short,
        check_closed,
    );
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

pub(super) fn multi_line_strings_without_selected(
    array: &MultiLineStringArray,
    selected: &Bitmap,
    too_short: TooShort,
) -> MultiLineStringArray {
    let (lines, coords, nulled) = without_selected::<false, false>(
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

pub(super) fn multi_polygons_without_selected(
    array: &MultiPolygonArray,
    selected: &Bitmap,
    too_short: TooShort,
    check_closed: bool,
) -> MultiPolygonArray {
    let (rings, coords, nulled) = without_selected_in_rings(
        &array.rings,
        &array.coords,
        selected,
        too_short,
        check_closed,
    );
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

/// `rows`, null too where a row holds a coordinate `selection` selects.
/// `inner_lists` are the list levels below the rows, outermost first.
fn rows_nulled_where_selected(
    rows: &List,
    inner_lists: &[&ValidList],
    coords: &ValidCoords,
    selection: &CoordinateSelection,
) -> List {
    let not_selected = which_are_not_selected(coords, selection);
    rows.clone()
        .with_validity(valid_rows_passing(rows, inner_lists, not_selected))
}

/// [SelectedAction::Drop] leaves the empty point, as a point cannot lose its coordinate.
pub fn points_with_selected(
    array: &PointArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
) -> PointArray {
    let (coords, validity) = (array.coords.ignoring_validity(), array.coords.validity());
    PointArray::new(match action {
        SelectedAction::SetEmpty | SelectedAction::Drop => {
            let selected = &selected_coords_bitmap(coords, selection);
            Coords::from_valid(emptied(coords, selected), validity.cloned())
        }
        SelectedAction::SetNull => {
            let not_selected = which_are_not_selected(coords, selection);
            let validity = combine_validities_and(validity, not_selected.as_ref());
            Coords::from_valid(coords.clone(), validity)
        }
    })
}

pub fn line_strings_with_selected(
    array: &LineStringArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> LineStringArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => LineStringArray {
            rows: array.rows.clone(),
            coords: emptied(&array.coords, &selected()),
        },
        SelectedAction::Drop => line_strings_without_selected(array, &selected(), too_short),
        SelectedAction::SetNull => LineStringArray {
            rows: rows_nulled_where_selected(&array.rows, &[], &array.coords, selection),
            coords: array.coords.clone(),
        },
    }
}

pub fn multi_points_with_selected(
    array: &MultiPointArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> MultiPointArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => MultiPointArray {
            rows: array.rows.clone(),
            coords: emptied(&array.coords, &selected()),
        },
        SelectedAction::Drop => multi_points_without_selected(array, &selected(), too_short),
        SelectedAction::SetNull => MultiPointArray {
            rows: rows_nulled_where_selected(&array.rows, &[], &array.coords, selection),
            coords: array.coords.clone(),
        },
    }
}

pub fn polygons_with_selected(
    array: &PolygonArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> PolygonArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => PolygonArray {
            rows: array.rows.clone(),
            rings: array.rings.clone(),
            coords: emptied(&array.coords, &selected()),
        },
        SelectedAction::Drop => polygons_without_selected(array, &selected(), too_short, true),
        SelectedAction::SetNull => PolygonArray {
            rows: rows_nulled_where_selected(
                &array.rows,
                &[&array.rings],
                &array.coords,
                selection,
            ),
            rings: array.rings.clone(),
            coords: array.coords.clone(),
        },
    }
}

pub fn multi_line_strings_with_selected(
    array: &MultiLineStringArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> MultiLineStringArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => MultiLineStringArray {
            rows: array.rows.clone(),
            lines: array.lines.clone(),
            coords: emptied(&array.coords, &selected()),
        },
        SelectedAction::Drop => multi_line_strings_without_selected(array, &selected(), too_short),
        SelectedAction::SetNull => MultiLineStringArray {
            rows: rows_nulled_where_selected(
                &array.rows,
                &[&array.lines],
                &array.coords,
                selection,
            ),
            lines: array.lines.clone(),
            coords: array.coords.clone(),
        },
    }
}

pub fn multi_polygons_with_selected(
    array: &MultiPolygonArray,
    selection: &CoordinateSelection,
    action: SelectedAction,
    too_short: TooShort,
) -> MultiPolygonArray {
    let selected = || selected_coords_bitmap(&array.coords, selection);
    match action {
        SelectedAction::SetEmpty => MultiPolygonArray {
            rows: array.rows.clone(),
            polygons: array.polygons.clone(),
            rings: array.rings.clone(),
            coords: emptied(&array.coords, &selected()),
        },
        SelectedAction::Drop => {
            multi_polygons_without_selected(array, &selected(), too_short, true)
        }
        SelectedAction::SetNull => MultiPolygonArray {
            rows: rows_nulled_where_selected(
                &array.rows,
                &[&array.polygons, &array.rings],
                &array.coords,
                selection,
            ),
            polygons: array.polygons.clone(),
            rings: array.rings.clone(),
            coords: array.coords.clone(),
        },
    }
}
