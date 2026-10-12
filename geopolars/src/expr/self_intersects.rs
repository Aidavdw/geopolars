//! Whether a linestring crosses or touches itself, in x and y.
//! This functionality is also exposed to the rest of this crate (for use in validity checking)

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use rsgeo::algorithm::sweep::Cross;
use rsgeo::{Coord, Intersections, Line, LineIntersection};

use super::is_standard_orientation::per_part;
use crate::geoarrow::coord::is_close;
use crate::geoarrow::geotraits::LineString;
use crate::geoarrow::storage::CoordsView;
use crate::geoarrow::{describe, Kind};

/// One segment of a linestring, numbered in order along it.
#[derive(Clone, Copy)]
struct Segment {
    index: usize,
    line: Line<f64>,
}

impl Cross for Segment {
    type Scalar = f64;

    fn line(&self) -> Line<f64> {
        self.line
    }
}

/// Whether `a` and `b` lie at almost the same place in x and y ([`is_close`]).
pub(crate) fn same_xy(a: Coord<f64>, b: Coord<f64>) -> bool {
    is_close(a.x, b.x) && is_close(a.y, b.y)
}

pub(crate) fn xy_vertices_validated(line: &LineString) -> Option<Vec<Coord<f64>>> {
    (line.start..line.end)
        .map(|i| {
            let (x, y) = line.coords.xy(i)?;
            (x.is_finite() && y.is_finite()).then_some(Coord { x, y })
        })
        .collect()
}

/// Whether `line` crosses or touches itself, in x and y.
/// `None` if one of its vertices is missing, or not finite in x or y.
/// See [`vertices_self_intersect`].
pub(crate) fn line_self_intersects(line: &LineString) -> Option<bool> {
    xy_vertices_validated(line).map(|vertices| vertices_self_intersect(&vertices))
}

/// Whether the line through `vertices` crosses or touches itself.
///
/// A segment whose ends lie at the same place is skipped.
/// Two segments in a row (and the last and the first, when the line closes)
/// may share their vertex, but not overlap; any other two may not touch at all.
pub(crate) fn vertices_self_intersect(vertices: &[Coord<f64>]) -> bool {
    let segments: Vec<Segment> = vertices
        .windows(2)
        .filter(|ends| !same_xy(ends[0], ends[1]))
        .enumerate()
        .map(|(index, ends)| Segment {
            index,
            line: Line::new(ends[0], ends[1]),
        })
        .collect();
    let n = segments.len();
    if n < 2 {
        return false;
    }
    let closed = same_xy(vertices[0], vertices[vertices.len() - 1]);
    let adjacent = |a: &Segment, b: &Segment| {
        let (low, high) = (a.index.min(b.index), a.index.max(b.index));
        high - low == 1 || (closed && low == 0 && high == n - 1)
    };
    Intersections::from_iter(segments)
        .any(|(a, b, hit)| !adjacent(&a, &b) || matches!(hit, LineIntersection::Collinear { .. }))
}

/// `output_type_func` for [`self_intersects`]:
/// a `bool` per linestring, or a list of them (one per part) for a multilinestring.
fn self_intersects_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = match describe(field.dtype())?.kind {
        Kind::LineString => DataType::Boolean,
        Kind::MultiLineString => DataType::List(Box::new(DataType::Boolean)),
        kind @ (Kind::Point | Kind::Polygon | Kind::MultiPoint | Kind::MultiPolygon) => {
            polars_bail!(
                SchemaMismatch: "self_intersects expects a `{}` or `{}` column, got: `{}`",
                Kind::LineString.name(), Kind::MultiLineString.name(), kind.name()
            )
        }
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// Whether each of the linestrings `rows` intersects itself,
/// missing where the linestring is, or one of its vertices.
fn lines(lines: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<BooleanArray> {
    let coords = CoordsView::new(lines.values().as_ref())?;
    Ok(rows
        .map(|i| {
            let (start, end) = lines.offsets().start_end(i);
            let line = LineString {
                coords: &coords,
                start,
                end,
            };
            lines
                .is_valid(i)
                .then(|| line_self_intersects(&line))
                .flatten()
        })
        .collect())
}

/// See `self_intersects`.
#[polars_expr(output_type_func=self_intersects_type)]
fn self_intersects(inputs: &[Series]) -> PolarsResult<Series> {
    let field = self_intersects_type(&[inputs[0].field().into_owned()])?;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    if *field.dtype() == DataType::Boolean {
        let chunks = chunks.map(|chunk| lines(chunk, 0..chunk.len()));
        return Ok(BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series());
    }
    let chunks = chunks.map(|chunk| per_part(chunk, lines));
    Ok(ListChunked::try_from_chunk_iter(name, chunks)?.into_series())
}
