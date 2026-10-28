//! Fixing the coordinates outside the CRS's bounds.
//! See `fix_out_of_bounds` in Python for the rules.
//!
//! With [OutOfBoundsAction::Clamp], every kernel only rewrites x and y, and keeps every offset.
//! Otherwise, it marks the coordinates outside the bounds ([out_of_bounds_coords]),
//! then acts on them as in [super::selected_coords].

use geopolars_core::{
    Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray, ValidCoords,
};

use super::is_valid::{is_within, Bounds};
use super::selected_coords::{
    line_strings_with_selected, multi_line_strings_with_selected, multi_points_with_selected,
    multi_polygons_with_selected, points_with_selected, polygons_with_selected, SelectedAction,
};
use super::too_short::TooShort;

/// See `action` of `OutOfBounds` in Python.
#[derive(Debug, Clone, Copy)]
pub enum OutOfBoundsAction {
    SetEmpty,
    Drop,
    SetNull,
    Clamp,
}

impl OutOfBoundsAction {
    /// The action on the coordinates outside the bounds, `None` for [OutOfBoundsAction::Clamp].
    fn selected_action(self) -> Option<SelectedAction> {
        match self {
            OutOfBoundsAction::SetEmpty => Some(SelectedAction::SetEmpty),
            OutOfBoundsAction::Drop => Some(SelectedAction::Drop),
            OutOfBoundsAction::SetNull => Some(SelectedAction::SetNull),
            OutOfBoundsAction::Clamp => None,
        }
    }
}

/// See `OutOfBounds` in Python, with its `crs_bounds` looked up as `bounds`.
#[derive(Debug, Clone, Copy)]
pub struct OutOfBounds {
    pub bounds: Bounds,
    pub action: OutOfBoundsAction,
    pub too_short: TooShort,
}

/// Whether each of `coords` lies outside `bounds`. The empty point never does.
fn out_of_bounds_coords(coords: &ValidCoords, bounds: &Bounds) -> Vec<bool> {
    coords
        .x()
        .iter()
        .zip(coords.y())
        .map(|(&x, &y)| !is_within(bounds, x, y) & !(x.is_nan() & y.is_nan()))
        .collect()
}

/// `coords`, with x and y clamped into `bounds`. NaN stays NaN; z and m are shared.
fn clamped(coords: &ValidCoords, bounds: &Bounds) -> ValidCoords {
    let [xmin, ymin, xmax, ymax] = *bounds;
    let clamp = |axis: &[f64], min: f64, max: f64| -> Vec<f64> {
        axis.iter().map(|value| value.clamp(min, max)).collect()
    };
    coords.with_positions([
        Some(clamp(coords.x(), xmin, xmax)),
        Some(clamp(coords.y(), ymin, ymax)),
        None,
    ])
}

pub fn points_with_fixed_out_of_bounds(array: &PointArray, fix: &OutOfBounds) -> PointArray {
    let coords = array.coords.ignoring_validity();
    match fix.action.selected_action() {
        None => PointArray::new(Coords::from_valid(
            clamped(coords, &fix.bounds),
            array.coords.validity().cloned(),
        )),
        Some(action) => {
            points_with_selected(array, &out_of_bounds_coords(coords, &fix.bounds), action)
        }
    }
}

pub fn line_strings_with_fixed_out_of_bounds(
    array: &LineStringArray,
    fix: &OutOfBounds,
) -> LineStringArray {
    match fix.action.selected_action() {
        None => LineStringArray {
            rows: array.rows.clone(),
            coords: clamped(&array.coords, &fix.bounds),
        },
        Some(action) => {
            let selected = out_of_bounds_coords(&array.coords, &fix.bounds);
            line_strings_with_selected(array, &selected, action, fix.too_short)
        }
    }
}

pub fn multi_points_with_fixed_out_of_bounds(
    array: &MultiPointArray,
    fix: &OutOfBounds,
) -> MultiPointArray {
    match fix.action.selected_action() {
        None => MultiPointArray {
            rows: array.rows.clone(),
            coords: clamped(&array.coords, &fix.bounds),
        },
        Some(action) => {
            let selected = out_of_bounds_coords(&array.coords, &fix.bounds);
            multi_points_with_selected(array, &selected, action, fix.too_short)
        }
    }
}

pub fn polygons_with_fixed_out_of_bounds(array: &PolygonArray, fix: &OutOfBounds) -> PolygonArray {
    match fix.action.selected_action() {
        None => PolygonArray {
            rows: array.rows.clone(),
            rings: array.rings.clone(),
            coords: clamped(&array.coords, &fix.bounds),
        },
        Some(action) => {
            let selected = out_of_bounds_coords(&array.coords, &fix.bounds);
            polygons_with_selected(array, &selected, action, fix.too_short)
        }
    }
}

pub fn multi_line_strings_with_fixed_out_of_bounds(
    array: &MultiLineStringArray,
    fix: &OutOfBounds,
) -> MultiLineStringArray {
    match fix.action.selected_action() {
        None => MultiLineStringArray {
            rows: array.rows.clone(),
            lines: array.lines.clone(),
            coords: clamped(&array.coords, &fix.bounds),
        },
        Some(action) => {
            let selected = out_of_bounds_coords(&array.coords, &fix.bounds);
            multi_line_strings_with_selected(array, &selected, action, fix.too_short)
        }
    }
}

pub fn multi_polygons_with_fixed_out_of_bounds(
    array: &MultiPolygonArray,
    fix: &OutOfBounds,
) -> MultiPolygonArray {
    match fix.action.selected_action() {
        None => MultiPolygonArray {
            rows: array.rows.clone(),
            polygons: array.polygons.clone(),
            rings: array.rings.clone(),
            coords: clamped(&array.coords, &fix.bounds),
        },
        Some(action) => {
            let selected = out_of_bounds_coords(&array.coords, &fix.bounds);
            multi_polygons_with_selected(array, &selected, action, fix.too_short)
        }
    }
}
