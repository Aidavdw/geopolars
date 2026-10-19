//! The simplification of a linestring or polygon, or their multi-variants, forwarded to GEOS.

use geopolars_compute::geos::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};
use geopolars_core::{
    GeoDimension, LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, Geo, Kind};
use geopolars_core::{downcast, List, ValidCoords, ValidList};

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
        Kind::LineString => simplify_line_strings(
            &LineStringArray {
                rows: List::new(rows),
                coords: ValidCoords::new(values)?,
            },
            tolerance,
            preserve_topology,
            ignore_per_row_errors,
        ),
        Kind::MultiLineString => {
            let lines: &ListArray<i64> = downcast(values, "a list of linestrings")?;
            simplify_multi_line_strings(
                &MultiLineStringArray {
                    rows: List::new(rows),
                    lines: ValidList::new(lines),
                    coords: ValidCoords::new(lines.values().as_ref())?,
                },
                tolerance,
                preserve_topology,
                ignore_per_row_errors,
            )
        }
        Kind::Polygon => {
            let rings: &ListArray<i64> = downcast(values, "a list of rings")?;
            simplify_polygons(
                &PolygonArray {
                    rows: List::new(rows),
                    rings: ValidList::new(rings),
                    coords: ValidCoords::new(rings.values().as_ref())?,
                },
                tolerance,
                preserve_topology,
                ignore_per_row_errors,
            )
        }
        Kind::MultiPolygon => {
            let polygons: &ListArray<i64> = downcast(values, "a list of polygons")?;
            let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
            simplify_multi_polygons(
                &MultiPolygonArray {
                    rows: List::new(rows),
                    polygons: ValidList::new(polygons),
                    rings: ValidList::new(rings),
                    coords: ValidCoords::new(rings.values().as_ref())?,
                },
                tolerance,
                preserve_topology,
                ignore_per_row_errors,
            )
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
