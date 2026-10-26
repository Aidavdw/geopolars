//! Removing the points that coincide with the one before them.
//! See `remove_repeated_points` in Python for the rules.
//!
//! Every kernel copies the kept coordinates once, a run of them at a time.
//! No part is ever removed, so only the innermost list level gets new offsets.
//! A missing row is processed like any other, as its value is masked anyway.
//! A row with a ring that would be left with fewer than [MIN_RING_VERTICES] vertices becomes missing.
//! That ring keeps no vertices.
//! With `drop_short_lines`, a short linestring or ring becomes empty instead (see [Lists::short_below]).
//!
//! # Const generic flags
//!
//! Every kernel here is generic over `TOLERANCE_COINCIDENCE`,
//! as in [has_repeated_points](super::has_repeated_points).
//!
//! ## `UNIQUE_Z`
//!
//! Every kernel here is generic over `UNIQUE_Z`.
//! If `UNIQUE_Z=true`, two vertices are only repeated when their z values are too.
//! A runtime `unique_z` is ignored for coordinates without a z.
//!
//! ## `UNIQUE_M`
//!
//! Every kernel here is generic over `UNIQUE_M`.
//! If `UNIQUE_M=true`, two vertices are only repeated when their m values are too.
//! A runtime `unique_m` is ignored for coordinates without an m.

use std::ops::Range;

use geopolars_core::output::validity_without;
use geopolars_core::{
    is_close, LineStringArray, List, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PolygonArray, Tolerance, ValidCoords, ValidList, MIN_RING_VERTICES,
};
use polars_arrow::offset::OffsetsBuffer;

use super::has_repeated_points::coincide;
use super::rings::offsets_of;

/// See `collapsed_lines` of `remove_repeated_points` in Python.
#[derive(Debug, Clone, Copy)]
pub enum CollapsedLines {
    KeepEnds,
    SingleVertex,
}

/// What happens to a linestring that collapses, or that has one vertex.
/// See `remove_repeated_points` in Python.
#[derive(Debug, Clone, Copy)]
pub struct LineOptions {
    pub collapsed_lines: CollapsedLines,
    pub drop_short_lines: bool,
}

/// What the lists of vertices handed to [push_kept_runs] are.
#[derive(Clone, Copy)]
enum Lists {
    /// The points of a multipoint. These have no ends to keep.
    Points,
    Lines(LineOptions),
    /// Rings always keep both ends, so they stay closed.
    Rings {
        drop_short_lines: bool,
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

    /// A list left with fewer vertices than this becomes empty. `0` when none does.
    /// That is a linestring of a single vertex, or a ring of at most two (one place, closed).
    fn short_below(self) -> usize {
        match self {
            Lists::Lines(LineOptions {
                drop_short_lines: true,
                ..
            }) => 2,
            Lists::Rings {
                drop_short_lines: true,
            } => 3,
            Lists::Points | Lists::Lines(_) | Lists::Rings { .. } => 0,
        }
    }

    /// A list that is not [short](Self::short_below) and has fewer vertices than this
    /// makes its row missing.
    fn min_vertices(self) -> usize {
        match self {
            Lists::Rings { .. } => MIN_RING_VERTICES,
            Lists::Points | Lists::Lines(_) => 0,
        }
    }
}

/// Whether vertices `i` and `j` [coincide], with `UNIQUE_M` in m too.
#[inline(always)]
fn is_repeat<const TOLERANCE_COINCIDENCE: bool, const UNIQUE_Z: bool, const UNIQUE_M: bool>(
    coords: &ValidCoords,
    i: usize,
    j: usize,
    tolerance: Tolerance,
) -> bool {
    coincide::<TOLERANCE_COINCIDENCE, UNIQUE_Z>(coords, i, j, tolerance)
        && (!UNIQUE_M || {
            let m = coords.m().expect("an m, with `UNIQUE_M`");
            if TOLERANCE_COINCIDENCE {
                is_close(m[i], m[j], tolerance)
            } else {
                m[i] == m[j]
            }
        })
}

/// `unique_z` and `unique_m`, only for an axis `coords` has.
fn unique_axes(coords: &ValidCoords, unique_z: bool, unique_m: bool) -> (bool, bool) {
    let dim = coords.dim();
    (unique_z && dim.has_z(), unique_m && dim.has_m())
}

/// Pushes the runs of `vertices` to keep onto `runs`, and returns how many vertices they hold.
/// A vertex is dropped when it [is a repeat](is_repeat) of the one kept before it.
///
/// Unless `lists` is [Lists::Points], the last vertex is always kept.
/// If it coincides with the one kept before it, that one is dropped instead, unless it is the first.
/// A [short](Lists::short_below) list keeps nothing.
/// `None`, with nothing pushed, as soon as it is clear the list will be left with
/// fewer than [min_vertices](Lists::min_vertices) without being short.
#[inline(always)]
fn push_kept_runs<const TOLERANCE_COINCIDENCE: bool, const UNIQUE_Z: bool, const UNIQUE_M: bool>(
    coords: &ValidCoords,
    vertices: Range<usize>,
    tolerance: Tolerance,
    lists: Lists,
    runs: &mut Vec<Range<usize>>,
) -> Option<usize> {
    let Range { start, end } = vertices;
    let (short_below, min_vertices) = (lists.short_below(), lists.min_vertices());
    if end - start < 2 {
        if end - start < short_below {
            return Some(0);
        }
        runs.push(start..end);
        return Some(end - start);
    }
    let first_run = runs.len();
    let (mut kept, mut last_kept, mut run_start) = (end - start, start, start);
    for i in start + 1..end {
        if is_repeat::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(coords, last_kept, i, tolerance) {
            kept -= 1;
            // The last vertex is only added back below when all others coincide with the first,
            // which leaves 2. Only such a list can still end up short.
            if kept < min_vertices && (short_below == 0 || last_kept != start) {
                runs.truncate(first_run);
                return None;
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
    if kept < short_below {
        runs.truncate(first_run);
        return Some(0);
    }
    if kept < min_vertices {
        runs.truncate(first_run);
        return None;
    }
    Some(kept)
}

/// The offsets of what each of `lists` keeps by [push_kept_runs], and the coordinates it keeps.
/// Last, the rings that collapsed, which keep none.
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
    let (mut runs, mut collapsed) = (Vec::new(), Vec::new());
    let lengths: Vec<usize> = lists
        .ranges()
        .enumerate()
        .map(|(j, list)| {
            push_kept_runs::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
                coords, list, tolerance, kind, &mut runs,
            )
            .unwrap_or_else(|| {
                collapsed.push(j);
                0
            })
        })
        .collect();
    (
        offsets_of(lengths.into_iter()),
        coords.copy_ranges(runs.into_iter()),
        collapsed,
    )
}

/// The list of `lists` that holds item `item`, `None` if none does.
fn list_holding(lists: &ValidList, item: usize) -> Option<usize> {
    let offsets = lists.offsets().buffer();
    let i = offsets
        .partition_point(|&offset| offset as usize <= item)
        .checked_sub(1)?;
    (i < lists.len() && item < offsets[i + 1] as usize).then_some(i)
}

/// `rows`, with the rows in `failed` missing too.
fn without_rows(rows: &List, failed: &[usize]) -> List {
    let validity = validity_without(rows.validity(), rows.len(), failed);
    rows.clone().with_validity(validity)
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
    let (rows, coords, _) = without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
        array.rows.ignoring_validity(),
        Lists::Lines(options),
        &array.coords,
        tolerance,
    );
    LineStringArray {
        rows: List::from_offsets(rows, array.rows.validity().cloned()),
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
    drop_short_lines: bool,
) -> PolygonArray {
    let (rings, coords, collapsed) =
        without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.rings,
            Lists::Rings { drop_short_lines },
            &array.coords,
            tolerance,
        );
    let failed: Vec<usize> = collapsed
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
    drop_short_lines: bool,
) -> PolygonArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        polygons_without_repeated_points_with(array, tolerance.unwrap_or_default(), drop_short_lines);
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
    let (lines, coords, _) = without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
        &array.lines,
        Lists::Lines(options),
        &array.coords,
        tolerance,
    );
    MultiLineStringArray {
        rows: array.rows.clone(),
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
    drop_short_lines: bool,
) -> MultiPolygonArray {
    let (rings, coords, collapsed) =
        without_repeated_vertices::<TOLERANCE_COINCIDENCE, UNIQUE_Z, UNIQUE_M>(
            &array.rings,
            Lists::Rings { drop_short_lines },
            &array.coords,
            tolerance,
        );
    let failed: Vec<usize> = collapsed
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
    drop_short_lines: bool,
) -> MultiPolygonArray {
    let (unique_z, unique_m) = unique_axes(&array.coords, unique_z, unique_m);
    with_const_flags!(
        multi_polygons_without_repeated_points_with(array, tolerance.unwrap_or_default(), drop_short_lines);
        tolerance.is_some(), unique_z, unique_m
    )
}
