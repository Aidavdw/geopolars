//! Whether a linestring or polygon crosses or touches itself, in x and y
//! (see [`geopolars_compute::rsgeo::line_strings_self_intersect`]).
//!
//! A multi-geometry gets a list, one per part.

use geopolars_compute::rsgeo::{
    line_strings_self_intersect, multi_line_strings_self_intersect, multi_polygons_self_intersect,
    polygons_self_intersect,
};
use geopolars_core::{LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, Kind};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfIntersectsKwargs {
    ignore_per_row_errors: bool,
}

/// For [`self_intersects`]:
/// a `bool` per linestring or polygon, or a list of them (one per part) for a multi-geometry.
fn kernel_output_type_self_intersects(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = match describe(field.dtype())?.kind {
        Kind::LineString | Kind::Polygon => DataType::Boolean,
        Kind::MultiLineString | Kind::MultiPolygon => DataType::List(Box::new(DataType::Boolean)),
        kind @ (Kind::Point | Kind::MultiPoint) => {
            polars_bail!(
                SchemaMismatch: "self_intersects expects a `{}`, `{}`, `{}` or `{}` column, got: `{}`",
                Kind::LineString.name(), Kind::Polygon.name(),
                Kind::MultiLineString.name(), Kind::MultiPolygon.name(), kind.name()
            )
        }
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// See `self_intersects`.
#[polars_expr(output_type_func=kernel_output_type_self_intersects)]
fn self_intersects(inputs: &[Series], kwargs: SelfIntersectsKwargs) -> PolarsResult<Series> {
    kernel_output_type_self_intersects(&[inputs[0].field().into_owned()])?;
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    let ignore_errors = kwargs.ignore_per_row_errors;
    let compute_err = |e| polars_err!(ComputeError: "self_intersects {e}");
    let out = match kind {
        Kind::LineString => {
            let chunks = chunks.map(|rows| {
                let array = LineStringArray::new(rows)?;
                PolarsResult::Ok(line_strings_self_intersect(&array))
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Polygon => {
            let chunks = chunks.map(|rows| {
                let array = PolygonArray::new(rows)?;
                polygons_self_intersect(&array, ignore_errors).map_err(compute_err)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiLineString => {
            let chunks = chunks.map(|rows| {
                let array = MultiLineStringArray::new(rows)?;
                PolarsResult::Ok(multi_line_strings_self_intersect(&array))
            });
            ListChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiPolygon => {
            let chunks = chunks.map(|rows| {
                let array = MultiPolygonArray::new(rows)?;
                multi_polygons_self_intersect(&array, ignore_errors).map_err(compute_err)
            });
            ListChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Point | Kind::MultiPoint => {
            unreachable!("refused by `kernel_output_type_self_intersects`")
        }
    };
    Ok(out)
}
