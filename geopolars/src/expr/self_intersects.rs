//! Whether a linestring crosses or touches itself, in x and y.
//! A polygon is forwarded per ring, and its holes to `rings_overlap`.
//! This functionality is also exposed to the rest of this crate (for use in validity checking)

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;
use rsgeo::algorithm::sweep::Cross;
use rsgeo::{Coord, Intersections, Line, LineIntersection, Polygon, PreparedGeometry, Relate};

use super::rings_overlap::{overlap, ring};
use crate::geoarrow::{describe, Kind};
use geopolars_core::is_close;
use geopolars_core::{downcast, ScalarLineString, ValidCoords};

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

/// The x and y of every vertex of `line`.
/// `None` if one of them is not finite.
pub(crate) fn xy_vertices_validated(line: &ScalarLineString) -> Option<Vec<Coord<f64>>> {
    line.positions::<false>()
        .map(|[x, y, _]| (x.is_finite() && y.is_finite()).then_some(Coord { x, y }))
        .collect()
}

/// Whether `line` crosses or touches itself, in x and y.
/// `None` if one of its vertices is not finite in x or y.
/// See [`vertices_self_intersect`].
pub(crate) fn line_self_intersects(line: &ScalarLineString) -> Option<bool> {
    xy_vertices_validated(line).map(|vertices| vertices_self_intersect(&vertices))
}

/// Whether the line through `vertices` crosses or touches itself.
///
/// A segment whose ends lie at the same place is skipped.
/// Two segments in a row (and the last and the first, when the line closes)
/// may share their vertex, but not overlap; any other two may not touch at all.
pub(crate) fn vertices_self_intersect(vertices: &[Coord<f64>]) -> bool {
    let ends = || {
        vertices
            .windows(2)
            .filter(|ends| !same_xy(ends[0], ends[1]))
    };
    // Counted rather than collected: the sweep collects the segments itself.
    let n = ends().count();
    if n < 2 {
        return false;
    }
    let closed = same_xy(vertices[0], vertices[vertices.len() - 1]);
    let adjacent = |a: &Segment, b: &Segment| {
        let (low, high) = (a.index.min(b.index), a.index.max(b.index));
        high - low == 1 || (closed && low == 0 && high == n - 1)
    };
    let segments = ends().enumerate().map(|(index, ends)| Segment {
        index,
        line: Line::new(ends[0], ends[1]),
    });
    Intersections::from_iter(segments)
        .any(|(a, b, hit)| !adjacent(&a, &b) || matches!(hit, LineIntersection::Collinear { .. }))
}

/// Whether a polygon with `rings` intersects itself:
/// one of them crosses itself, or two of its holes (every ring after the first) overlap.
fn polygon_self_intersects(rings: Vec<Polygon<f64>>) -> bool {
    if rings
        .iter()
        .any(|ring| vertices_self_intersect(&ring.exterior().0))
    {
        return true;
    }
    // Prepare every hole once, as it is related to every other.
    let holes: Vec<_> = rings
        .into_iter()
        .skip(1)
        .map(PreparedGeometry::from)
        .collect();
    holes.iter().enumerate().any(|(i, hole)| {
        holes[i + 1..]
            .iter()
            .any(|other| overlap(&hole.relate(other), false))
    })
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

/// Whether each of the linestrings `rows` intersects itself,
/// missing where the linestring is, or one of its vertices.
fn lines(lines: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<BooleanArray> {
    let coords = ValidCoords::new(lines.values().as_ref())?;
    Ok(rows
        .map(|i| {
            let (start, end) = lines.offsets().start_end(i);
            let line = ScalarLineString {
                coords: &coords,
                vertices: start..end,
            };
            lines
                .is_valid(i)
                .then(|| line_self_intersects(&line))
                .flatten()
        })
        .collect())
}

/// Whether each of the polygons `rows` intersects itself ([`polygon_self_intersects`]),
/// missing where the polygon is, or one of its rings or their vertices.
/// Errors on a ring that is not one, as `rings_overlap` does.
fn polygons(polygons: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<BooleanArray> {
    let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
    let coords = ValidCoords::new(rings.values().as_ref())?;
    rows.map(|i| {
        if polygons.is_null(i) {
            return Ok(None);
        }
        let (start, end) = polygons.offsets().start_end(i);
        // Every ring, so that a missing one makes the polygon missing.
        let of_polygon = (start..end)
            .map(|r| {
                if rings.is_null(r) {
                    return Ok(None);
                }
                let (start, end) = rings.offsets().start_end(r);
                ring(&coords, start, end)
            })
            .collect::<PolarsResult<Option<Vec<_>>>>()?;
        Ok(of_polygon.map(polygon_self_intersects))
    })
    .collect()
}

/// One list per row of `chunk`, holding `parts` of the parts it points at.
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

/// See `self_intersects`.
#[polars_expr(output_type_func=kernel_output_type_self_intersects)]
fn self_intersects(inputs: &[Series]) -> PolarsResult<Series> {
    kernel_output_type_self_intersects(&[inputs[0].field().into_owned()])?;
    let kind = describe(inputs[0].dtype())?.kind;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();
    let chunks = storage.list()?.downcast_iter();
    let out = match kind {
        Kind::LineString => {
            let chunks = chunks.map(|chunk| lines(chunk, 0..chunk.len()));
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Polygon => {
            let chunks = chunks.map(|chunk| polygons(chunk, 0..chunk.len()));
            BooleanChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiLineString => {
            let chunks = chunks.map(|chunk| per_part(chunk, lines));
            ListChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::MultiPolygon => {
            let chunks = chunks.map(|chunk| per_part(chunk, polygons));
            ListChunked::try_from_chunk_iter(name, chunks)?.into_series()
        }
        Kind::Point | Kind::MultiPoint => {
            unreachable!("refused by `kernel_output_type_self_intersects`")
        }
    };
    Ok(out)
}
