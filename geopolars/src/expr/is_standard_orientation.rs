//! Whether a geometry runs in the standard orientation:
//! counter-clockwise, with a polygon's holes the other way round.
//! See `is_standard_orientation` in Python for the rules.
//!
//! A multilinestring or multipolygon gets a list, one per part.

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::area::twice_signed;
use crate::geoarrow::geotraits::LineString;
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrientationKwargs {
    /// Whether a polygon's holes have to run clockwise, rather than counter-clockwise.
    holes_clockwise: bool,
}

fn unoriented(kind: Kind) -> PolarsError {
    polars_err!(
        SchemaMismatch: "is_standard_orientation does not accept a `{}` column",
        kind.name()
    )
}

/// `output_type_func` for [`is_standard_orientation`]:
/// a `bool` per geometry, or a list of them (one per part) for a multi.
fn is_standard_orientation_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = match describe(field.dtype())?.kind {
        Kind::LineString | Kind::Polygon => DataType::Boolean,
        Kind::MultiLineString | Kind::MultiPolygon => DataType::List(Box::new(DataType::Boolean)),
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(unoriented(kind)),
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// Whether `vertices` run counter-clockwise (or clockwise, with `clockwise`).
/// No vertices at all is in either orientation.
fn runs(coords: &CoordsView, vertices: Range<usize>, clockwise: bool) -> bool {
    if vertices.is_empty() {
        return true;
    }
    if !vertices.clone().all(|i| coords.xy(i).is_some()) {
        return false;
    }
    let winding = twice_signed(&LineString {
        coords,
        start: vertices.start,
        end: vertices.end,
    });
    if clockwise {
        winding < 0.0
    } else {
        winding > 0.0
    }
}

/// Whether each of the linestrings `rows` runs counter-clockwise
fn lines(lines: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<BooleanArray> {
    let coords = CoordsView::new(lines.values().as_ref())?;
    Ok(rows
        .map(|i| {
            let (start, end) = lines.offsets().start_end(i);
            lines.is_valid(i).then(|| runs(&coords, start..end, false))
        })
        .collect())
}

/// Whether each of the polygons `rows` has its exterior counter-clockwise,
/// and its holes clockwise (or counter-clockwise, without `holes_clockwise`).
fn polygons(
    polygons: &ListArray<i64>,
    rows: Range<usize>,
    holes_clockwise: bool,
) -> PolarsResult<BooleanArray> {
    let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
    let coords = CoordsView::new(rings.values().as_ref())?;
    let ring = |ring: usize, hole: bool| -> Option<bool> {
        rings.is_valid(ring).then(|| {
            let (start, end) = rings.offsets().start_end(ring);
            runs(&coords, start..end, hole && holes_clockwise)
        })
    };
    Ok(rows
        .map(|i| {
            if polygons.is_null(i) {
                return None;
            }
            let (start, end) = polygons.offsets().start_end(i);
            // Every ring, so that a missing one makes the polygon missing.
            // No rings at all is the empty polygon, which is in either orientation.
            (start..end).try_fold(true, |all, i| Some(ring(i, i > start)? && all))
        })
        .collect())
}

fn per_part(
    chunk: &ListArray<i64>,
    parts: impl FnOnce(&ListArray<i64>, Range<usize>) -> PolarsResult<BooleanArray>,
) -> PolarsResult<ListArray<i64>> {
    let values = downcast::<ListArray<i64>>(chunk.values().as_ref(), "a list of parts")?;
    // Only the parts this chunk points at: it may be a slice of a larger one.
    let (first, last) = (
        *chunk.offsets().first() as usize,
        *chunk.offsets().last() as usize,
    );
    Ok(ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(ArrowDataType::Boolean),
        Offsets::try_from_lengths(chunk.offsets().lengths())?.into(),
        parts(values, first..last)?.boxed(),
        chunk.validity().cloned(),
    ))
}

/// See `is_standard_orientation`.
#[polars_expr(output_type_func=is_standard_orientation_type)]
fn is_standard_orientation(inputs: &[Series], kwargs: OrientationKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let holes_clockwise = kwargs.holes_clockwise;
    let out = match kind {
        Kind::LineString => {
            let chunks = storage.list()?.downcast_iter();
            let chunks = chunks.map(|chunk| lines(chunk, 0..chunk.len()));
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Polygon => {
            let chunks = storage.list()?.downcast_iter();
            let chunks = chunks.map(|chunk| polygons(chunk, 0..chunk.len(), holes_clockwise));
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiLineString => {
            let chunks = storage.list()?.downcast_iter();
            let chunks = chunks.map(|chunk| per_part(chunk, lines));
            ListChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiPolygon => {
            let chunks = storage.list()?.downcast_iter();
            let chunks = chunks
                .map(|chunk| per_part(chunk, |parts, rows| polygons(parts, rows, holes_clockwise)));
            ListChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(unoriented(kind)),
    };
    Ok(out)
}
