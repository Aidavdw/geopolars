//! The convex hull of a polygon, forwarded to GEOS.

use geoarrow_abstr::PolygonArrayView;
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::storage::{downcast, ListView, PolygonOutput, ValidCoordsView, ValidListView};
use crate::geoarrow::{describe, GeoColumn, Kind};

/// Only polygons are hulled for now: other kinds are refused.
fn polygons(dtype: &DataType) -> PolarsResult<GeoColumn<'_>> {
    let geo = describe(dtype)?;
    match geo.kind {
        Kind::Polygon => Ok(geo),
        Kind::Point
        | Kind::LineString
        | Kind::MultiPoint
        | Kind::MultiLineString
        | Kind::MultiPolygon => polars_bail!(
            SchemaMismatch: "convex_hull expects a polygon column, got: {}", dtype
        ),
    }
}

/// For [`convex_hull`]: a polygon per polygon, with the input's dimension and metadata.
fn kernel_output_type_convex_hull(input_fields: &[Field]) -> PolarsResult<Field> {
    polygons(input_fields[0].dtype())?;
    Ok(input_fields[0].clone())
}

/// See `convex_hull`.
#[polars_expr(output_type_func=kernel_output_type_convex_hull)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = polygons(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| {
            let rings: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of rings")?;
            let array = PolygonArrayView {
                rows: ListView::new(rows),
                rings: ValidListView::new(rings),
                coords: ValidCoordsView::new(rings.values().as_ref())?,
            };
            let PolygonOutput(hulls) = array
                .convex_hulls()
                .map_err(|e| polars_err!(ComputeError: "{e}"))?;
            PolarsResult::Ok(hulls.boxed())
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the hulls are built with the input's own storage dtype.
    let storage = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };
    Ok(storage.into_extension(geo.typ.clone()))
}
