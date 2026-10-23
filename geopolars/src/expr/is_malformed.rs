//! Whether a geometry is malformed, through `geopolars_compute::malformation`.
//! See `is_malformed` in Python for the rules.

use geopolars_compute::malformation::{
    line_strings_are_malformed, multi_line_strings_are_malformed, multi_points_are_malformed,
    multi_polygons_are_malformed, points_are_malformed, polygons_are_malformed,
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
pub struct IsMalformedKwargs {
    check_repeated_points: bool,
    check_empty: bool,
    check_ring_self_intersect: bool,
    check_polygon_self_overlap: bool,
    check_multi_polygon_self_overlap: bool,
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
}

impl Default for IsMalformedKwargs {
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

impl IsMalformedKwargs {
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
    polars_err!(ComputeError: "is_malformed {e}")
}

/// [`are_malformed`] for one chunk of the storage of a `kind`.
/// The chunk is trusted to be valid (see `is_valid`), so its views are built directly.
fn chunk_is_malformed(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &IsMalformedKwargs,
) -> PolarsResult<BooleanArray> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    let (repeated, empty, tolerance) = (
        kwargs.check_repeated_points,
        kwargs.check_empty,
        kwargs.tolerance(),
    );
    Ok(match kind {
        Kind::Point => points_are_malformed(&PointArray::new(Coords::new(chunk)?), empty),
        Kind::LineString => {
            line_strings_are_malformed(&LineStringArray::new(rows()?)?, repeated, empty, tolerance)
        }
        Kind::MultiPoint => {
            multi_points_are_malformed(&MultiPointArray::new(rows()?)?, repeated, empty, tolerance)
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(rows()?)?;
            multi_line_strings_are_malformed(&array, repeated, empty, tolerance)
        }
        Kind::Polygon => polygons_are_malformed(
            &PolygonArray::new(rows()?)?,
            repeated,
            empty,
            &kwargs.polygon_topology(),
            tolerance,
        )
        .map_err(compute_err)?,
        Kind::MultiPolygon => multi_polygons_are_malformed(
            &MultiPolygonArray::new(rows()?)?,
            repeated,
            empty,
            &kwargs.multi_polygon_topology(),
            tolerance,
        )
        .map_err(compute_err)?,
    })
}

fn are_malformed(geometries: &Series, kwargs: &IsMalformedKwargs) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_malformed(chunk.as_ref(), kind, kwargs));
    BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)
}

/// For [`is_malformed`]: a `bool` per geometry.
fn kernel_output_type_is_malformed(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_malformed`.
#[polars_expr(output_type_func=kernel_output_type_is_malformed)]
fn is_malformed(inputs: &[Series], kwargs: IsMalformedKwargs) -> PolarsResult<Series> {
    Ok(are_malformed(&inputs[0], &kwargs)?.into_series())
}

/// The rows of `malformed` that are there and not malformed.
fn not_malformed(malformed: &BooleanArray) -> BooleanArray {
    let values = match malformed.validity() {
        Some(rows) => &!malformed.values() & rows,
        None => !malformed.values(),
    };
    BooleanArray::new(
        ArrowDataType::Boolean,
        values,
        malformed.validity().cloned(),
    )
}

/// See `null_out_malformed`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn null_out_malformed(inputs: &[Series], kwargs: IsMalformedKwargs) -> PolarsResult<Series> {
    let malformed = are_malformed(&inputs[0], &kwargs)?;
    let chunks = malformed.downcast_iter().map(not_malformed);
    let keep = BooleanChunked::from_chunk_iter(malformed.name().clone(), chunks);
    keep_rows(&inputs[0], &keep)
}
