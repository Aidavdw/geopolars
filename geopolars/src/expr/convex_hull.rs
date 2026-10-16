//! The convex hull of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geoarrow_abstr::{
    LineStringArrayView, MultiLineStringArrayView, MultiPolygonArrayView, PolygonArrayView,
};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::storage::{downcast, ListView, PolygonOutput, ValidCoordsView, ValidListView};
use geoarrow_abstr::GeoDimension;

use crate::geoarrow::{describe, Geo, Kind};

/// Points enclose nothing, so they are refused.
fn encloses_nothing(dtype: &DataType) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "convex_hull expects a linestring, polygon, multilinestring or multipolygon column, got: {}",
        dtype
    )
}

/// A polygon per geometry, with the input's metadata and z, but without m.
/// Also the kind of the input.
fn hull_type(dtype: &DataType) -> PolarsResult<(DataType, Kind)> {
    let geo = describe(dtype)?;
    match geo.kind {
        Kind::LineString | Kind::MultiLineString | Kind::Polygon | Kind::MultiPolygon => {}
        Kind::Point | Kind::MultiPoint => return Err(encloses_nothing(dtype)),
    }
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let dtype = Geo::new(Kind::Polygon, dim, geo.metadata.clone()).dtype();
    Ok((dtype, geo.kind))
}

/// For [`convex_hull`].
fn kernel_output_type_convex_hull(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = hull_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// The convex hulls of one chunk of `kind` geometries, as polygons.
fn chunk_hulls(rows: &ListArray<i64>, kind: Kind, dtype: &DataType) -> PolarsResult<PolygonOutput> {
    let values = rows.values().as_ref();
    match kind {
        Kind::LineString => LineStringArrayView {
            rows: ListView::new(rows),
            coords: ValidCoordsView::new(values)?,
        }
        .convex_hulls(),
        Kind::MultiLineString => {
            let lines: &ListArray<i64> = downcast(values, "a list of linestrings")?;
            MultiLineStringArrayView {
                rows: ListView::new(rows),
                lines: ValidListView::new(lines),
                coords: ValidCoordsView::new(lines.values().as_ref())?,
            }
            .convex_hulls()
        }
        Kind::Polygon => {
            let rings: &ListArray<i64> = downcast(values, "a list of rings")?;
            PolygonArrayView {
                rows: ListView::new(rows),
                rings: ValidListView::new(rings),
                coords: ValidCoordsView::new(rings.values().as_ref())?,
            }
            .convex_hulls()
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
            .convex_hulls()
        }
        Kind::Point | Kind::MultiPoint => return Err(encloses_nothing(dtype)),
    }
    .map_err(|e| polars_err!(ComputeError: "{e}"))
}

/// See `convex_hull`.
#[polars_expr(output_type_func=kernel_output_type_convex_hull)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, kind) = hull_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| Ok(chunk_hulls(rows, kind, inputs[0].dtype())?.0.boxed()))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the hulls are polygons of `dtype`'s dimension, its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
