//! Whether each geometry is valid, in one pass that runs the checks of
//! [crate::native] and [crate::rsgeo] one after another on every row.
//! A row stops at the first check it fails.
//! See `is_valid` in Python for the rules.
//!
//! Each gives a `bool` per row, `false` for a missing row.

use geopolars_core::{
    LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray, PointArray,
    PolygonArray,
};
use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;

use crate::native::{
    coordinates_are_valid, line_string_shape_is_valid, point_is_valid, polygon_shape_is_valid,
    Bounds,
};
use crate::per_row::per_row_bool;
use crate::rsgeo::{multi_polygon_topology_is_valid, polygon_topology_is_valid, TopologyChecks};

/// The checks to run. See `is_valid` in Python.
/// Missing values below the row are not checked here:
/// the views have to be built with them moved out to the rows (see [crate::inner_nulls]).
#[derive(Debug, Clone, Copy)]
pub struct ValidityChecks {
    /// Where x and y have to lie, if anywhere.
    pub bounds: Option<Bounds>,
    pub check_finite_coordinates: bool,
    pub check_line_string_zero_length_sections: bool,
    pub check_ring_self_intersect: bool,
    pub check_polygon_self_overlap: bool,
    /// Only read for multipolygons.
    pub check_multi_polygon_self_overlap: bool,
}

impl ValidityChecks {
    fn bounds_or_any(&self) -> Bounds {
        self.bounds.unwrap_or_default()
    }

    /// The topology checks that are on, `None` if there are none.
    /// If `multi=true`, the check on how a multipolygon's parts fit counts too.
    fn topology_checks(&self, multi: bool) -> Option<TopologyChecks> {
        let polygon = self.check_ring_self_intersect || self.check_polygon_self_overlap;
        (polygon || (multi && self.check_multi_polygon_self_overlap)).then_some(TopologyChecks {
            check_ring_self_intersect: self.check_ring_self_intersect,
            check_polygon_self_overlap: self.check_polygon_self_overlap,
            check_multi_polygon_self_overlap: self.check_multi_polygon_self_overlap,
        })
    }

    // Relating rings needs them finite, so with the topology checks on, they always are checked.
    fn has_finite_check(&self, topology: Option<&TopologyChecks>) -> bool {
        self.check_finite_coordinates || topology.is_some()
    }
}

/// See [point_is_valid] for the flags.
fn points_are_valid_with<const FINITE: bool, const BOUNDS: bool, const Z: bool, const M: bool>(
    array: &PointArray,
    bounds: &Bounds,
) -> BooleanArray {
    let coords = &array.coords;
    let values = Bitmap::from_iter((0..coords.len()).map(|i| {
        coords.is_valid(i)
            && point_is_valid::<FINITE, BOUNDS, Z, M>(coords.ignoring_validity(), i, bounds)
    }));
    BooleanArray::new(ArrowDataType::Boolean, values, coords.validity().cloned())
}

pub fn points_are_valid(array: &PointArray, checks: &ValidityChecks) -> BooleanArray {
    let dim = array.coords.dim();
    with_const_flags!(
        points_are_valid_with(array, &checks.bounds_or_any());
        checks.check_finite_coordinates, checks.bounds.is_some(), dim.has_z(), dim.has_m()
    )
}

/// See [point_is_valid] for the flags.
fn multi_points_are_valid_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
>(
    array: &MultiPointArray,
    bounds: &Bounds,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        array
            .rows
            .range(i)
            .all(|j| point_is_valid::<FINITE, BOUNDS, Z, M>(&array.coords, j, bounds))
    })
}

pub fn multi_points_are_valid(array: &MultiPointArray, checks: &ValidityChecks) -> BooleanArray {
    let dim = array.coords.dim();
    with_const_flags!(
        multi_points_are_valid_with(array, &checks.bounds_or_any());
        checks.check_finite_coordinates, checks.bounds.is_some(), dim.has_z(), dim.has_m()
    )
}

/// See [coordinates_are_valid] and [line_string_shape_is_valid] for the flags.
fn line_strings_are_valid_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
    const ZERO: bool,
>(
    array: &LineStringArray,
    bounds: &Bounds,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.rows.range(i);
        coordinates_are_valid::<FINITE, BOUNDS, Z, M>(&array.coords, vertices.clone(), bounds)
            && line_string_shape_is_valid::<ZERO, Z>(&array.coords, vertices)
    })
}

pub fn line_strings_are_valid(array: &LineStringArray, checks: &ValidityChecks) -> BooleanArray {
    let dim = array.coords.dim();
    with_const_flags!(
        line_strings_are_valid_with(array, &checks.bounds_or_any());
        checks.check_finite_coordinates, checks.bounds.is_some(), dim.has_z(), dim.has_m(),
        checks.check_line_string_zero_length_sections
    )
}

/// See [coordinates_are_valid] and [line_string_shape_is_valid] for the flags.
fn multi_line_strings_are_valid_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
    const ZERO: bool,
>(
    array: &MultiLineStringArray,
    bounds: &Bounds,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let vertices = array.lines.span(array.rows.range(i));
        coordinates_are_valid::<FINITE, BOUNDS, Z, M>(&array.coords, vertices, bounds)
            && array
                .get(i)
                .line_strings()
                .all(|line| line_string_shape_is_valid::<ZERO, Z>(line.coords, line.vertices))
    })
}

pub fn multi_line_strings_are_valid(
    array: &MultiLineStringArray,
    checks: &ValidityChecks,
) -> BooleanArray {
    let dim = array.coords.dim();
    with_const_flags!(
        multi_line_strings_are_valid_with(array, &checks.bounds_or_any());
        checks.check_finite_coordinates, checks.bounds.is_some(), dim.has_z(), dim.has_m(),
        checks.check_line_string_zero_length_sections
    )
}

/// See [coordinates_are_valid] and [polygon_shape_is_valid] for the flags.
/// `topology` holds the topology checks, if any are on.
fn polygons_are_valid_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
    const ZERO: bool,
>(
    array: &PolygonArray,
    bounds: &Bounds,
    topology: Option<&TopologyChecks>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let polygon = array.get(i);
        let vertices = array.rings.span(array.rows.range(i));
        coordinates_are_valid::<FINITE, BOUNDS, Z, M>(&array.coords, vertices, bounds)
            && polygon_shape_is_valid::<ZERO, Z>(&polygon)
            && topology.is_none_or(|checks| polygon_topology_is_valid(&polygon, checks))
    })
}

pub fn polygons_are_valid(array: &PolygonArray, checks: &ValidityChecks) -> BooleanArray {
    let dim = array.coords.dim();
    let topology = checks.topology_checks(false);
    with_const_flags!(
        polygons_are_valid_with(array, &checks.bounds_or_any(), topology.as_ref());
        checks.has_finite_check(topology.as_ref()), checks.bounds.is_some(),
        dim.has_z(), dim.has_m(), checks.check_line_string_zero_length_sections
    )
}

/// See [polygons_are_valid_with].
fn multi_polygons_are_valid_with<
    const FINITE: bool,
    const BOUNDS: bool,
    const Z: bool,
    const M: bool,
    const ZERO: bool,
>(
    array: &MultiPolygonArray,
    bounds: &Bounds,
    topology: Option<&TopologyChecks>,
) -> BooleanArray {
    per_row_bool(&array.rows, |i| {
        let multi = array.get(i);
        let vertices = array.rings.span(array.polygons.span(array.rows.range(i)));
        coordinates_are_valid::<FINITE, BOUNDS, Z, M>(&array.coords, vertices, bounds)
            && multi
                .polygons()
                .all(|polygon| polygon_shape_is_valid::<ZERO, Z>(&polygon))
            && topology.is_none_or(|checks| multi_polygon_topology_is_valid(&multi, checks))
    })
}

pub fn multi_polygons_are_valid(
    array: &MultiPolygonArray,
    checks: &ValidityChecks,
) -> BooleanArray {
    let dim = array.coords.dim();
    let topology = checks.topology_checks(true);
    with_const_flags!(
        multi_polygons_are_valid_with(array, &checks.bounds_or_any(), topology.as_ref());
        checks.has_finite_check(topology.as_ref()), checks.bounds.is_some(),
        dim.has_z(), dim.has_m(), checks.check_line_string_zero_length_sections
    )
}
