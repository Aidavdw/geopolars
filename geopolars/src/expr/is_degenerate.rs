//! Whether a geometry is degenerate, through `geopolars_compute::degeneracy`.
//! See `is_degenerate` in Python for the rules.

use geopolars_compute::degeneracy::{
    line_strings_are_degenerate, multi_line_strings_are_degenerate, multi_points_are_degenerate,
    multi_polygons_are_degenerate, points_are_degenerate, polygons_are_degenerate,
    MultiPolygonTopologyChecks, PolygonTopologyChecks,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray, Tolerance,
};
use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::kernel_output_type_same_geometry;
use super::is_valid::keep_rows;
use crate::geoarrow::{describe, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsDegenerateKwargs {
    check_repeated_points: bool,
    check_empty: bool,
    check_ring_self_intersect: bool,
    check_polygon_self_overlap: bool,
    check_multi_polygon_self_overlap: bool,
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
}

impl Default for IsDegenerateKwargs {
    fn default() -> Self {
        Self {
            check_repeated_points: true,
            check_empty: true,
            check_ring_self_intersect: true,
            check_polygon_self_overlap: true,
            check_multi_polygon_self_overlap: true,
            abs_tol: None,
            rel_tol: None,
        }
    }
}

impl IsDegenerateKwargs {
    /// `None` when neither tolerance is given; otherwise the missing one is `0.0`.
    fn tolerance(&self) -> Option<Tolerance> {
        (self.abs_tol.is_some() || self.rel_tol.is_some()).then(|| Tolerance {
            abs_tol: self.abs_tol.unwrap_or(0.0),
            rel_tol: self.rel_tol.unwrap_or(0.0),
        })
    }

    fn polygon_topology(&self) -> PolygonTopologyChecks {
        PolygonTopologyChecks {
            check_ring_self_intersect: self.check_ring_self_intersect,
            check_polygon_self_overlap: self.check_polygon_self_overlap,
        }
    }

    fn multi_polygon_topology(&self) -> MultiPolygonTopologyChecks {
        MultiPolygonTopologyChecks {
            polygons: self.polygon_topology(),
            check_multi_polygon_self_overlap: self.check_multi_polygon_self_overlap,
        }
    }
}

fn compute_err(e: impl std::fmt::Display) -> PolarsError {
    polars_err!(ComputeError: "is_degenerate {e}")
}

/// [`are_degenerate`] for one chunk of the storage of a `kind`.
/// The chunk is trusted to be valid (see `is_valid`), so its views are built directly.
fn chunk_is_degenerate(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &IsDegenerateKwargs,
) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    let (repeated, empty, tolerance) = (
        kwargs.check_repeated_points,
        kwargs.check_empty,
        kwargs.tolerance(),
    );
    Ok(match kind {
        Kind::Point => points_are_degenerate(&PointArray::new(Coords::new(chunk)?), empty),
        Kind::LineString => {
            line_strings_are_degenerate(&LineStringArray::new(rows()?)?, repeated, empty, tolerance)
        }
        Kind::MultiPoint => {
            multi_points_are_degenerate(&MultiPointArray::new(rows()?)?, repeated, empty, tolerance)
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(rows()?)?;
            multi_line_strings_are_degenerate(&array, repeated, empty, tolerance)
        }
        Kind::Polygon => polygons_are_degenerate(
            &PolygonArray::new(rows()?)?,
            repeated,
            empty,
            &kwargs.polygon_topology(),
            tolerance,
        )
        .map_err(compute_err)?,
        Kind::MultiPolygon => multi_polygons_are_degenerate(
            &MultiPolygonArray::new(rows()?)?,
            repeated,
            empty,
            &kwargs.multi_polygon_topology(),
            tolerance,
        )
        .map_err(compute_err)?,
    })
}

fn are_degenerate(
    geometries: &Series,
    kwargs: &IsDegenerateKwargs,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_degenerate(chunk.as_ref(), kind, kwargs));
    BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)
}

/// For [`is_degenerate`]: a `bool` per geometry.
fn kernel_output_type_is_degenerate(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_degenerate`.
#[polars_expr(output_type_func=kernel_output_type_is_degenerate)]
fn is_degenerate(inputs: &[Series], kwargs: IsDegenerateKwargs) -> PolarsResult<Series> {
    Ok(are_degenerate(&inputs[0], &kwargs)?.into_series())
}

/// The rows of `degenerate` that are there and not degenerate.
fn not_degenerate(degenerate: &BooleanArray) -> BooleanArray {
    let values = match degenerate.validity() {
        Some(rows) => &!degenerate.values() & rows,
        None => !degenerate.values(),
    };
    BooleanArray::new(
        ArrowDataType::Boolean,
        values,
        degenerate.validity().cloned(),
    )
}

/// See `null_out_degenerate`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn null_out_degenerate(inputs: &[Series], kwargs: IsDegenerateKwargs) -> PolarsResult<Series> {
    let degenerate = are_degenerate(&inputs[0], &kwargs)?;
    let chunks = degenerate.downcast_iter().map(not_degenerate);
    let keep = BooleanChunked::from_chunk_iter(degenerate.name().clone(), chunks);
    keep_rows(&inputs[0], &keep)
}
