//! Removing the points that coincide with the one before them.
//! See `remove_repeated_points` in Python for the rules.
//!
//! Every kernel copies the kept coordinates once, a run of them at a time.
//! No part is ever removed, so only the innermost list level gets new offsets.
//! A null row is processed like any other, as its value is masked anyway.
//! A linestring or ring left with fewer than [min_vertices](Lists::min_vertices) vertices is too short.
//! [TooShort] says what happens to it.
//!
//! # Const generic flags
//!
//! Every kernel here is generic over `TOLERANCE_COINCIDENCE`, `UNIQUE_Z` and `UNIQUE_M`,
//! as in [has_repeated_points](super::has_repeated_points).

use std::ops::Range;

use geopolars_core::output::validity_without;
use geopolars_core::{
    LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PolygonArray,
    Tolerance, ValidCoords, ValidList, MIN_RING_VERTICES,
};
use polars_arrow::offset::OffsetsBuffer;

use super::has_repeated_points::{coincide, unique_axes};
use super::rings::offsets_of;
use super::too_short::{list_holding, without_rows, TooShort};

/// See `collapsed_lines` of `remove_repeated_points` in Python.
#[derive(Debug, Clone, Copy)]
pub enum CollapsedLines {
    KeepEnds,
    SingleVertex,
}

/// What happens to a linestring that collapses, or that is too short.
/// See `remove_repeated_points` in Python.
#[derive(Debug, Clone, Copy)]
pub struct LineOptions {
    pub collapsed_lines: CollapsedLines,
    pub too_short: TooShort,
}

/// What the lists of vertices handed to [push_kept_runs] are.
#[derive(Clone, Copy)]
enum Lists {
    /// The points of a multipoint. These have no ends to keep, and are never too short.
    Points,
    Lines(LineOptions),
    /// Rings always keep both ends, so they stay closed.
    Rings {
        too_short: TooShort,
    },
}

impl Lists {
    /// Whether a list that collapsed into its first vertex keeps its last vertex too.
    fn keeps_both_ends(self) -> bool {
        !matches!(
            self,
            Lists::Lines(LineOptions {
                collapsed_lines: CollapsedLines::SingleVertex,
                ..
            })
        )
    }

    /// A list with some vertices, but fewer than this, is too short.
    fn min_vertices(self) -> usize {
        match self {
            Lists::Points => 0,
            Lists::Lines(_) => 2,
            Lists::Rings { .. } => MIN_RING_VERTICES,
        }
    }

    fn too_short(self) -> TooShort {
        match self {
            Lists::Points => TooShort::Keep,
            Lists::Lines(options) => options.too_short,
            Lists::Rings { too_short } => too_short,
        }
    }
}

/// What [push_kept_runs] returns for a list that keeps `kept` vertices,
/// in the runs from `first_run` on.
fn kept_or_too_short(
    kept: usize,
    lists: Lists,
    runs: &mut Vec<Range<usize>>,
    first_run: usize,
) -> Option<usize> {
    if kept == 0 || kept >= lists.min_vertices() {
        return Some(kept);
    }
    match lists.too_short() {
        TooShort::Keep => Some(kept),
        TooShort::Empty => {
            runs.truncate(first_run);
            Some(0)
        }
        TooShort::Null => {
            runs.truncate(first_run);
            None
        }
    }
}

/// Pushes the runs of `vertices` to keep onto `runs`, and returns how many vertices they hold.
/// A vertex is dropped when it [coincides](coincide) with the one kept before it.
///
/// Unless `lists` is [Lists::Points], the last vertex is always kept.
/// If it coincides with the one kept before it, that one is dropped instead, unless it is the first.
/// A [too short](Lists::min_vertices) list keeps nothing, unless [TooShort::Keep].
/// With [TooShort::Null], it returns `None`, as soon as it is clear the list will be too short.
#[inline(always)]
fn push_kept_runs<const TOLERANCE_COINCIDENCE: bool, const UNIQUE_Z: bool, const UNIQUE_M: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    tolerance: Tolerance,
    lists: Lists,
    runs: &mut Vec<Range<usize>>,
) -> Option<usize> {
    let Range { start, end } = vertices;
    let first_run = runs.len();
    if end - start < 2 {
        runs.push(start..end);
        return kept_or_too_short(end - start, lists, runs, first_run);
    }
    let gives_up_early = !matches!(lists.too_short(), TooShort::Keep);
    let (mut kept, mut last_kept, mut run_start) = (end - start, start, start);
    for i in start + 1..end {
        if coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(coords, last_kept, i, tolerance) {
            kept -= 1;
            // The last vertex is only added back below when all others coincide with the first.
            let added_back = usize::from(last_kept == start && lists.keeps_both_ends());
            if gives_up_early && kept + added_back < lists.min_vertices() {
                return kept_or_too_short(kept + added_back, lists, runs, first_run);
            }
            if run_start < i {
                runs.push(run_start..i);
            }
            run_start = i + 1;
        } else {
            last_kept = i;
        }
    }
    if run_start < end {
        runs.push(run_start..end);
    } else if let Lists::Points = lists {
        // Every point after `last_kept` was dropped, which is all there is to do.
    } else if last_kept == start {
        if lists.keeps_both_ends() {
            runs.push(end - 1..end);
            kept += 1;
        }
    } else {
        // Every vertex after `last_kept` was dropped, so it ends the last run.
        let run = runs.last_mut().expect("the run of the last kept vertex");
        if run.len() == 1 {
            runs.pop();
        } else {
            run.end -= 1;
        }
        runs.push(end - 1..end);
    }
    kept_or_too_short(kept, lists, runs, first_run)
}

/// The offsets of what each of `lists` keeps by [push_kept_runs], and the coordinates it keeps.
/// Last, the lists that are too short with [TooShort::Null], which keep none.
fn without_repeated_vertices<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    lists: &ValidList,
    kind: Lists,
    coords: &ValidCoords,
    tolerance: Tolerance,
) -> (OffsetsBuffer<i64>, ValidCoords, Vec<usize>) {
    let (mut runs, mut too_short) = (Vec::new(), Vec::new());
    let lengths: Vec<usize> = lists
        .ranges()
        .enumerate()
        .map(|(j, list)| {
            push_kept_runs::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
                coords, list, tolerance, kind, &mut runs,
            )
            .unwrap_or_else(|| {
                too_short.push(j);
                0
            })
        })
        .collect();
    (
        offsets_of(lengths.into_iter()),
        coords.copy_ranges(runs.into_iter()),
        too_short,
    )
}

fn line_strings_without_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &LineStringArray,
    tolerance: Tolerance,
    options: LineOptions,
) -> LineStringArray {
    let (offsets, coords, too_short) =
        without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            array.rows.ignoring_validity(),
            Lists::Lines(options),
            &array.coords,
            tolerance,
        );
    let validity = validity_without(array.rows.validity(), array.rows.len(), &too_short);
    LineStringArray {
        rows: List::from_offsets(offsets, validity),
        coords,
    }
}

pub fn line_strings_without_repeated_points(
    array: &LineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    options: LineOptions,
) -> LineStringArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        line_strings_without_repeated_points_with(array, tolerance.unwrap_or_default(), options);
        tolerance.is_some(), unique_z, unique_m
    )
}

fn multi_points_without_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiPointArray,
    tolerance: Tolerance,
) -> MultiPointArray {
    let (rows, coords, _) = without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
        array.rows.ignoring_validity(),
        Lists::Points,
        &array.coords,
        tolerance,
    );
    MultiPointArray {
        rows: List::from_offsets(rows, array.rows.validity().cloned()),
        coords,
    }
}

pub fn multi_points_without_repeated_points(
    array: &MultiPointArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
) -> MultiPointArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_points_without_repeated_points_with(array, tolerance.unwrap_or_default());
        tolerance.is_some(), unique_z, unique_m
    )
}

fn polygons_without_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &PolygonArray,
    tolerance: Tolerance,
    too_short: TooShort,
) -> PolygonArray {
    let (rings, coords, too_short) =
        without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.rings,
            Lists::Rings { too_short },
            &array.coords,
            tolerance,
        );
    let failed: Vec<usize> = too_short
        .into_iter()
        .filter_map(|ring| list_holding(array.rows.ignoring_validity(), ring))
        .collect();
    PolygonArray {
        rows: without_rows(&array.rows, &failed),
        rings: ValidList::from_offsets(rings),
        coords,
    }
}

pub fn polygons_without_repeated_points(
    array: &PolygonArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    too_short: TooShort,
) -> PolygonArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        polygons_without_repeated_points_with(array, tolerance.unwrap_or_default(), too_short);
        tolerance.is_some(), unique_z, unique_m
    )
}

fn multi_line_strings_without_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiLineStringArray,
    tolerance: Tolerance,
    options: LineOptions,
) -> MultiLineStringArray {
    let (lines, coords, too_short) =
        without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.lines,
            Lists::Lines(options),
            &array.coords,
            tolerance,
        );
    let failed: Vec<usize> = too_short
        .into_iter()
        .filter_map(|line| list_holding(array.rows.ignoring_validity(), line))
        .collect();
    MultiLineStringArray {
        rows: without_rows(&array.rows, &failed),
        lines: ValidList::from_offsets(lines),
        coords,
    }
}

pub fn multi_line_strings_without_repeated_points(
    array: &MultiLineStringArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    options: LineOptions,
) -> MultiLineStringArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_line_strings_without_repeated_points_with(array, tolerance.unwrap_or_default(), options);
        tolerance.is_some(), unique_z, unique_m
    )
}

fn multi_polygons_without_repeated_points_with<
    const TOLERANCE_COINCIDENCE: bool,
    const UNIQUE_Z: bool,
    const UNIQUE_M: bool,
>(
    array: &MultiPolygonArray,
    tolerance: Tolerance,
    too_short: TooShort,
) -> MultiPolygonArray {
    let (rings, coords, too_short) =
        without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.rings,
            Lists::Rings { too_short },
            &array.coords,
            tolerance,
        );
    let failed: Vec<usize> = too_short
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

pub fn multi_polygons_without_repeated_points(
    array: &MultiPolygonArray,
    tolerance: Option<Tolerance>,
    unique_z: bool,
    unique_m: bool,
    too_short: TooShort,
) -> MultiPolygonArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_polygons_without_repeated_points_with(array, tolerance.unwrap_or_default(), too_short);
        tolerance.is_some(), unique_z, unique_m
    )
}
