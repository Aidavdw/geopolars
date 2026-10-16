//! The simplification of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geoarrow_abstr::{
    GeoDimension, LineStringArrayView, MultiLineStringArrayView, MultiPolygonArrayView,
    PolygonArrayView,
};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::storage::{
    downcast, LineStringOutput, ListView, MultiLineStringOutput, MultiPolygonOutput, PolygonOutput,
    ValidCoordsView, ValidListView,
};
use crate::geoarrow::{describe, Geo, Kind};

/// Python returns a point or multipoint as it is, so it never gets here.
fn already_simple(dtype: &DataType) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "simplify expects a linestring, polygon, multilinestring or multipolygon column, got: {}",
        dtype
    )
}

/// The input's kind and metadata, with its z, but without m.
/// Also the kind of the input.
fn simplified_type(dtype: &DataType) -> PolarsResult<(DataType, Kind)> {
    let geo = describe(dtype)?;
    match geo.kind {
        Kind::LineString | Kind::MultiLineString | Kind::Polygon | Kind::MultiPolygon => {}
        Kind::Point | Kind::MultiPoint => return Err(already_simple(dtype)),
    }
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let dtype = Geo::new(geo.kind, dim, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind))
}

/// For [`simplify`].
fn kernel_output_type_simplify(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = simplified_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// See python docs for details.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplifyKwargs {
    tolerance: f64,
    preserve_topology: bool,
    ignore_per_row_errors: bool,
}

/// One chunk of `kind` geometries, simplified.
fn chunk_simplified(
    rows: &ListArray<i64>,
    kind: Kind,
    dtype: &DataType,
    kwargs: &SimplifyKwargs,
) -> PolarsResult<ListArray<i64>> {
    let SimplifyKwargs {
        tolerance,
        preserve_topology,
        ignore_per_row_errors,
    } = *kwargs;
    let values = rows.values().as_ref();
    match kind {
        Kind::LineString => LineStringArrayView {
            rows: ListView::new(rows),
            coords: ValidCoordsView::new(values)?,
        }
        .simplify::<LineStringOutput>(tolerance, preserve_topology, ignore_per_row_errors)
        .map(|output| output.0),
        Kind::MultiLineString => {
            let lines: &ListArray<i64> = downcast(values, "a list of linestrings")?;
            MultiLineStringArrayView {
                rows: ListView::new(rows),
                lines: ValidListView::new(lines),
                coords: ValidCoordsView::new(lines.values().as_ref())?,
            }
            .simplify::<MultiLineStringOutput>(tolerance, preserve_topology, ignore_per_row_errors)
            .map(|output| output.0)
        }
        Kind::Polygon => {
            let rings: &ListArray<i64> = downcast(values, "a list of rings")?;
            PolygonArrayView {
                rows: ListView::new(rows),
                rings: ValidListView::new(rings),
                coords: ValidCoordsView::new(rings.values().as_ref())?,
            }
            .simplify::<PolygonOutput>(tolerance, preserve_topology, ignore_per_row_errors)
            .map(|output| output.0)
        }
        Kind::MultiPolygon => {
            let polygons: &ListArray<i64> = downcast(values, "a list of polygons")?;
            let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
            MultiPolygonArrayView {
                rows: ListView::new(rows),
                polygons: ValidListView::new(polygons),
                rings: ValidListView::new(rings),
                coords: ValidCoordsView::new(rings.values().as_ref())?,
            }
            .simplify::<MultiPolygonOutput>(tolerance, preserve_topology, ignore_per_row_errors)
            .map(|output| output.0)
        }
        Kind::Point | Kind::MultiPoint => return Err(already_simple(dtype)),
    }
    .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// See `simplify`.
#[polars_expr(output_type_func=kernel_output_type_simplify)]
fn simplify(inputs: &[Series], kwargs: SimplifyKwargs) -> PolarsResult<Series> {
    let (dtype, kind) = simplified_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| Ok(chunk_simplified(rows, kind, inputs[0].dtype(), &kwargs)?.boxed()))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the simplified geometries are of the input's kind, in `dtype`'s dimension: its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
