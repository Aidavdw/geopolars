//! Whether a geometry is valid
//! See `is_valid` in Python for the rules.
//!
//! [are_valid] can also be used by other modules

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::same_geometry;
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
type Bounds = [f64; 4];

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsValidKwargs {
    bounds: Option<Bounds>,
}

/// Checks the coordinates of one chunk,
/// and the ranges of them that make up a line or ring.
struct Checker<'a> {
    coords: CoordsView<'a>,
    bounds: Option<Bounds>,
}

impl<'a> Checker<'a> {
    fn new(coords: &'a dyn Array, bounds: Option<Bounds>) -> PolarsResult<Self> {
        Ok(Self {
            coords: CoordsView::new(coords)?,
            bounds,
        })
    }

    fn coord(&self, i: usize) -> bool {
        self.coords.present(i)
            && self.coords.values(i).all(f64::is_finite)
            && self.bounds.is_none_or(|[xmin, ymin, xmax, ymax]| {
                let (x, y) = (self.coords.nth(i, 0), self.coords.nth(i, 1));
                (xmin..=xmax).contains(&x) && (ymin..=ymax).contains(&y)
            })
    }

    /// A valid coordinate, or the empty point
    fn point(&self, i: usize) -> bool {
        (self.coords.present(i) && self.coords.values(i).all(f64::is_nan)) || self.coord(i)
    }

    /// No vertices, or at least two valid ones, no two in a row at the same place.
    fn line(&self, vertices: Range<usize>) -> bool {
        vertices.len() != 1
            && vertices.clone().all(|i| self.coord(i))
            && vertices
                .clone()
                .zip(vertices.skip(1))
                .all(|(a, b)| !self.coords.same_place(a, b))
    }

    /// A valid line that closes, around at least a triangle.
    fn ring(&self, vertices: Range<usize>) -> bool {
        self.line(vertices.clone()) && self.coords.is_ring(vertices)
    }

    /// Check every ring
    /// either all of them empty, or all of them valid rings.
    fn polygon(&self, rings: &ListArray<i64>, polygon: Range<usize>) -> bool {
        let mut rings_of = polygon.map(|ring| part(rings, ring));
        if rings_of
            .clone()
            .all(|ring| ring.is_some_and(|ring| ring.is_empty()))
        {
            return true;
        }
        rings_of.all(|ring| ring.is_some_and(|ring| self.ring(ring)))
    }
}

fn part(list: &ListArray<i64>, i: usize) -> Option<Range<usize>> {
    list.is_valid(i).then(|| {
        let (start, end) = list.offsets().start_end(i);
        start..end
    })
}

fn per_row(rows: &dyn Array, valid: impl Fn(usize) -> bool) -> BooleanArray {
    (0..rows.len())
        .map(|i| rows.is_valid(i).then(|| valid(i)))
        .collect()
}

/// [`are_valid`] for one chunk of the storage of a `kind`.
fn chunk_is_valid(
    chunk: &dyn Array,
    kind: Kind,
    bounds: Option<Bounds>,
) -> PolarsResult<BooleanArray> {
    if kind == Kind::Point {
        let check = Checker::new(chunk, bounds)?;
        return Ok(per_row(chunk, |i| check.point(i)));
    }
    let rows: &ListArray<i64> = downcast(chunk, "a list of parts")?;
    let parts = rows.values().as_ref();
    let row = |i: usize| part(rows, i).expect("only called on rows that are there");
    Ok(match kind {
        Kind::Point => unreachable!("handled above"),
        Kind::LineString => {
            let check = Checker::new(parts, bounds)?;
            per_row(chunk, |i| check.line(row(i)))
        }
        Kind::MultiPoint => {
            let check = Checker::new(parts, bounds)?;
            per_row(chunk, |i| row(i).all(|point| check.point(point)))
        }
        Kind::Polygon => {
            let rings: &ListArray<i64> = downcast(parts, "a list of rings")?;
            let check = Checker::new(rings.values().as_ref(), bounds)?;
            per_row(chunk, |i| check.polygon(rings, row(i)))
        }
        Kind::MultiLineString => {
            let lines: &ListArray<i64> = downcast(parts, "a list of linestrings")?;
            let check = Checker::new(lines.values().as_ref(), bounds)?;
            per_row(chunk, |i| {
                row(i).all(|line| part(lines, line).is_some_and(|line| check.line(line)))
            })
        }
        Kind::MultiPolygon => {
            let polygons: &ListArray<i64> = downcast(parts, "a list of polygons")?;
            let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
            let check = Checker::new(rings.values().as_ref(), bounds)?;
            per_row(chunk, |i| {
                row(i).all(|polygon| {
                    part(polygons, polygon).is_some_and(|polygon| check.polygon(rings, polygon))
                })
            })
        }
    })
}

/// Whether each geometry in `geometries` is valid, missing where the geometry is.
/// One chunk out for every chunk in.
pub(crate) fn are_valid(
    geometries: &Series,
    bounds: Option<Bounds>,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_valid(chunk.as_ref(), kind, bounds));
    BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)
}

/// `output_type_func` for [`is_valid`]: a `bool` per geometry.
fn is_valid_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_valid`.
#[polars_expr(output_type_func=is_valid_type)]
fn is_valid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    Ok(are_valid(&inputs[0], kwargs.bounds)?.into_series())
}

/// See `null_out_invalid`.
#[polars_expr(output_type_func=same_geometry)]
fn null_out_invalid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let valid = are_valid(geometries, kwargs.bounds)?;
    // Missing rows do not count, and stay missing.
    if valid.all() {
        return Ok(geometries.clone());
    }

    let geo = describe(geometries.dtype())?;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .zip(valid.downcast_iter())
        .map(|(chunk, valid)| {
            // A missing row is `false` in `valid`'s values, so they alone say what stays.
            chunk.with_validity(Some(valid.values().clone()))
        })
        .collect();
    // SAFETY: only the outer validity changed, so every chunk is still the storage.
    let storage = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };
    Ok(storage.into_extension(geo.typ.clone()))
}
