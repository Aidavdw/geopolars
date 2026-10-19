//! Whether a geometry runs counter-clockwise, with a polygon's holes clockwise.
//! See `is_ccw` in Python for the rules.

use geopolars_compute::native::{
    line_strings_are_ccw, multi_line_strings_are_ccw, multi_polygons_are_ccw, polygons_are_ccw,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{LineStringArray, MultiLineStringArray, MultiPolygonArray, PolygonArray};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::{describe, Kind};
use geopolars_core::{downcast, List, ValidCoords, ValidList};

/// See python
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsCcwKwargs {
    holes_cw: bool,
}

fn unoriented(kind: Kind) -> PolarsError {
    polars_err!(SchemaMismatch: "is_ccw does not accept a `{}` column", kind.name())
}

/// For [`is_ccw`]: a `bool` per geometry.
fn kernel_output_type_is_ccw(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match describe(field.dtype())?.kind {
        Kind::LineString | Kind::Polygon | Kind::MultiLineString | Kind::MultiPolygon => {
            Ok(Field::new(field.name().clone(), DataType::Boolean))
        }
        kind @ (Kind::Point | Kind::MultiPoint) => Err(unoriented(kind)),
    }
}

/// See `is_ccw`.
#[polars_expr(output_type_func=kernel_output_type_is_ccw)]
fn is_ccw(inputs: &[Series], kwargs: IsCcwKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    let holes_cw = kwargs.holes_cw;
    let out = match kind {
        Kind::LineString => {
            let chunks = chunks.map(|rows| {
                let array = LineStringArray {
                    rows: List::new(rows),
                    coords: ValidCoords::new(rows.values().as_ref())?,
                };
                let out = line_strings_are_ccw(&array);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Polygon => {
            let chunks = chunks.map(|rows| {
                let rings: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of rings")?;
                let array = PolygonArray {
                    rows: List::new(rows),
                    rings: ValidList::new(rings),
                    coords: ValidCoords::new(rings.values().as_ref())?,
                };
                let out = with_const_flags!(polygons_are_ccw(&array); holes_cw);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiLineString => {
            let chunks = chunks.map(|rows| {
                let lines = downcast(rows.values().as_ref(), "a list of linestrings")?;
                let array = MultiLineStringArray {
                    rows: List::new(rows),
                    lines: ValidList::new(lines),
                    coords: ValidCoords::new(lines.values().as_ref())?,
                };
                let out = multi_line_strings_are_ccw(&array);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiPolygon => {
            let chunks = chunks.map(|rows| {
                let polygons: &ListArray<i64> =
                    downcast(rows.values().as_ref(), "a list of polygons")?;
                let rings: &ListArray<i64> =
                    downcast(polygons.values().as_ref(), "a list of rings")?;
                let array = MultiPolygonArray {
                    rows: List::new(rows),
                    polygons: ValidList::new(polygons),
                    rings: ValidList::new(rings),
                    coords: ValidCoords::new(rings.values().as_ref())?,
                };
                let out = with_const_flags!(multi_polygons_are_ccw(&array); holes_cw);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(unoriented(kind)),
    };
    Ok(out)
}
