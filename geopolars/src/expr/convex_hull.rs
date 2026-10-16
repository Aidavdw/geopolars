//! The convex hull of a polygon or multipolygon, forwarded to GEOS.

use geoarrow_abstr::{MultiPolygonArrayView, PolygonArrayView};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::storage::{downcast, ListView, PolygonOutput, ValidCoordsView, ValidListView};
use geoarrow_abstr::GeoDimension;

use crate::geoarrow::{describe, Geo, Kind};

/// A polygon per polygon or multipolygon, with the input's metadata and z, but without m.
/// Also whether the input is a multipolygon.
fn hull_type(dtype: &DataType) -> PolarsResult<(DataType, bool)> {
    let geo = describe(dtype)?;
    let multi = match geo.kind {
        Kind::Polygon => false,
        Kind::MultiPolygon => true,
        Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => {
            polars_bail!(
                SchemaMismatch: "convex_hull expects a polygon or multipolygon column, got: {}", dtype
            )
        }
    };
    let dim = if geo.dim.has_z() {
        GeoDimension::XYZ
    } else {
        GeoDimension::XY
    };
    let dtype = Geo::new(Kind::Polygon, dim, geo.metadata.clone()).dtype();
    Ok((dtype, multi))
}

/// For [`convex_hull`].
fn kernel_output_type_convex_hull(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = hull_type(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// See `convex_hull`.
#[polars_expr(output_type_func=kernel_output_type_convex_hull)]
fn convex_hull(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, multi) = hull_type(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|rows| {
            let PolygonOutput(hulls) = if multi {
                let polygons: &ListArray<i64> =
                    downcast(rows.values().as_ref(), "a list of polygons")?;
                let rings: &ListArray<i64> =
                    downcast(polygons.values().as_ref(), "a list of rings")?;
                let array = MultiPolygonArrayView {
                    rows: ListView::new(rows),
                    polygons: ValidListView::new(polygons),
                    rings: ValidListView::new(rings),
                    coords: ValidCoordsView::new(rings.values().as_ref())?,
                };
                array.convex_hulls()
            } else {
                let rings: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of rings")?;
                let array = PolygonArrayView {
                    rows: ListView::new(rows),
                    rings: ValidListView::new(rings),
                    coords: ValidCoordsView::new(rings.values().as_ref())?,
                };
                array.convex_hulls()
            }
            .map_err(|e| polars_err!(ComputeError: "{e}"))?;
            PolarsResult::Ok(hulls.boxed())
        })
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the hulls are polygons of `dtype`'s dimension, its storage.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
