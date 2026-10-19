//! Whether a geometry runs counter-clockwise, with a polygon's holes clockwise.
//! See `is_ccw` in Python for the rules.

use geoarrow_abstr::{
    with_const_flags, LineStringArrayView, MultiLineStringArrayView, MultiPolygonArrayView,
    PolygonArrayView,
};
use polars::prelude::*;
use polars_arrow::array::ListArray;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::storage::{downcast, BoolArray, ListView, ValidCoordsView, ValidListView};
use crate::geoarrow::{describe, Kind};

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

/// [geoarrow_abstr::native::polygons_are_ccw], into a polars array.
/// `with_const_flags!` cannot name the output type itself, so it calls this.
fn polygons_are_ccw<const HOLES_CW: bool>(
    array: &PolygonArrayView<ListView, ValidListView, ValidCoordsView>,
) -> BoolArray {
    geoarrow_abstr::native::polygons_are_ccw::<HOLES_CW, _>(array)
}

/// [geoarrow_abstr::native::multi_polygons_are_ccw], into a polars array.
/// `with_const_flags!` cannot name the output type itself, so it calls this.
fn multi_polygons_are_ccw<const HOLES_CW: bool>(
    array: &MultiPolygonArrayView<ListView, ValidListView, ValidListView, ValidCoordsView>,
) -> BoolArray {
    geoarrow_abstr::native::multi_polygons_are_ccw::<HOLES_CW, _>(array)
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
                let array = LineStringArrayView {
                    rows: ListView::new(rows),
                    coords: ValidCoordsView::new(rows.values().as_ref())?,
                };
                let BoolArray(out) = geoarrow_abstr::native::line_strings_are_ccw(&array);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Polygon => {
            let chunks = chunks.map(|rows| {
                let rings: &ListArray<i64> = downcast(rows.values().as_ref(), "a list of rings")?;
                let array = PolygonArrayView {
                    rows: ListView::new(rows),
                    rings: ValidListView::new(rings),
                    coords: ValidCoordsView::new(rings.values().as_ref())?,
                };
                let BoolArray(out) = with_const_flags!(polygons_are_ccw(&array); holes_cw);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiLineString => {
            let chunks = chunks.map(|rows| {
                let lines = downcast(rows.values().as_ref(), "a list of linestrings")?;
                let array = MultiLineStringArrayView {
                    rows: ListView::new(rows),
                    lines: ValidListView::new(lines),
                    coords: ValidCoordsView::new(lines.values().as_ref())?,
                };
                let BoolArray(out) = geoarrow_abstr::native::multi_line_strings_are_ccw(&array);
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
                let array = MultiPolygonArrayView {
                    rows: ListView::new(rows),
                    polygons: ValidListView::new(polygons),
                    rings: ValidListView::new(rings),
                    coords: ValidCoordsView::new(rings.values().as_ref())?,
                };
                let BoolArray(out) = with_const_flags!(multi_polygons_are_ccw(&array); holes_cw);
                PolarsResult::Ok(out)
            });
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(unoriented(kind)),
    };
    Ok(out)
}
