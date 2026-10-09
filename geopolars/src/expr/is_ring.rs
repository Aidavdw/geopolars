//! Whether a linestring is a ring (see [`CoordsView::is_ring`]).
//!
//! A multilinestring gets a list, one per part.

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// `output_type_func` for [`is_ring`]:
/// a `bool` per linestring, or a list of them (one per part) for a multilinestring.
fn is_ring_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = match describe(field.dtype())?.kind {
        Kind::LineString => DataType::Boolean,
        Kind::MultiLineString => DataType::List(Box::new(DataType::Boolean)),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            polars_bail!(
                SchemaMismatch: "is_ring expects a `{}` or `{}` column, got: `{}`",
                Kind::LineString.name(), Kind::MultiLineString.name(), kind.name()
            )
        }
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// Whether each of the linestrings `rows` is a ring, missing where the linestring is.
fn rings(lines: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<BooleanArray> {
    let coords = CoordsView::new(lines.values().as_ref())?;
    Ok(rows
        .map(|i| {
            let (start, end) = lines.offsets().start_end(i);
            lines.is_valid(i).then(|| coords.is_ring(start..end))
        })
        .collect())
}

/// See `is_ring`.
#[polars_expr(output_type_func=is_ring_type)]
fn is_ring(inputs: &[Series]) -> PolarsResult<Series> {
    let field = is_ring_type(&[inputs[0].field().into_owned()])?;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    if *field.dtype() == DataType::Boolean {
        let chunks = chunks.map(|chunk| rings(chunk, 0..chunk.len()));
        return Ok(BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series());
    }
    let chunks = chunks.map(|chunk| -> PolarsResult<_> {
        let lines = downcast::<ListArray<i64>>(chunk.values().as_ref(), "a list of linestrings")?;
        // Only the parts this chunk points at: it may be a slice of a larger one.
        let (first, last) = (
            *chunk.offsets().first() as usize,
            *chunk.offsets().last() as usize,
        );
        Ok(ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(ArrowDataType::Boolean),
            Offsets::try_from_lengths(chunk.offsets().lengths())?.into(),
            rings(lines, first..last)?.boxed(),
            chunk.validity().cloned(),
        ))
    });
    Ok(ListChunked::try_from_chunk_iter(name, chunks)?.into_series())
}
