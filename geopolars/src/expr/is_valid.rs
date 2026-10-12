//! Whether a geometry is valid
//! See `is_valid` in Python for the rules.
//! A polygon's rings are also checked against each other, in x and y, through rsgeo's `Relate`,
//! and so are a multipolygon's parts.
//!
//! [are_valid] can also be used by other modules

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use pyo3_polars::derive::polars_expr;
use rsgeo::coordinate_position::CoordPos;
use rsgeo::dimensions::Dimensions;
use rsgeo::relate::IntersectionMatrix;
use rsgeo::{Coord, LineString, Polygon, PreparedGeometry, Relate};
use serde::Deserialize;

use super::coords::same_geometry;
use super::self_intersects::{vertices_self_intersect, xy_vertices_validated};
use crate::geoarrow::geotraits::LineString as GeoLineString;
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
type Bounds = [f64; 4];

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsValidKwargs {
    bounds: Option<Bounds>,
    /// Whether every coordinate has to be there, with every axis.
    check_missing_coordinates: bool,
    /// Whether every axis of a coordinate that is there has to be finite.
    check_finite_coordinates: bool,
    /// Whether every coordinate that is there has to lie inside `bounds`.
    check_crs_bounds: bool,
    /// Whether no two vertices in a row of a line or ring may lie at the same place.
    check_line_string_zero_length_sections: bool,
    /// Whether a polygon's rings may not cross themselves.
    check_ring_self_intersect: bool,
    /// Whether a polygon's rings have to [fit](rings_fit) together.
    check_polygon_self_overlap: bool,
    /// Whether a multipolygon's parts have to [fit](parts_fit) together.
    check_multi_polygon_self_overlap: bool,
}

impl Default for IsValidKwargs {
    fn default() -> Self {
        Self {
            bounds: None,
            check_missing_coordinates: true,
            check_finite_coordinates: true,
            check_crs_bounds: true,
            check_line_string_zero_length_sections: true,
            check_ring_self_intersect: true,
            check_polygon_self_overlap: true,
            check_multi_polygon_self_overlap: true,
        }
    }
}

/// Checks the coordinates of one chunk,
/// and the ranges of them that make up a line or ring.
struct Checker<'a> {
    coords: CoordsView<'a>,
    kwargs: &'a IsValidKwargs,
}

impl<'a> Checker<'a> {
    fn new(coords: &'a dyn Array, kwargs: &'a IsValidKwargs) -> PolarsResult<Self> {
        Ok(Self {
            coords: CoordsView::new(coords)?,
            kwargs,
        })
    }

    fn coord(&self, i: usize) -> bool {
        let kwargs = self.kwargs;
        // A missing coordinate has no values to check.
        if !self.coords.present(i) {
            return !kwargs.check_missing_coordinates;
        }
        (!kwargs.check_finite_coordinates || self.coords.values(i).all(f64::is_finite))
            && (!kwargs.check_crs_bounds
                || kwargs.bounds.is_none_or(|[xmin, ymin, xmax, ymax]| {
                    let (x, y) = (self.coords.nth(i, 0), self.coords.nth(i, 1));
                    (xmin..=xmax).contains(&x) && (ymin..=ymax).contains(&y)
                }))
    }

    /// A valid coordinate, or the empty point
    fn point(&self, i: usize) -> bool {
        (self.coords.present(i) && self.coords.values(i).all(f64::is_nan)) || self.coord(i)
    }

    /// No vertices, or at least two valid ones, no two in a row at the same place
    /// (the latter only when the kwargs ask for it).
    fn line(&self, vertices: Range<usize>) -> bool {
        vertices.len() != 1
            && vertices.clone().all(|i| self.coord(i))
            && (!self.kwargs.check_line_string_zero_length_sections
                || vertices
                    .clone()
                    .zip(vertices.skip(1))
                    .all(|(a, b)| !self.coords.same_place(a, b)))
    }

    /// A valid line that closes, around at least a triangle.
    fn ring(&self, vertices: Range<usize>) -> bool {
        self.line(vertices.clone()) && self.coords.is_ring(vertices)
    }

    /// Check every ring
    /// either all of them empty, or all of them valid rings that do not cross themselves,
    /// and that [fit](rings_fit) together (each of those two only when the kwargs ask for it).
    fn polygon(&self, rings: &ListArray<i64>, polygon: Range<usize>) -> bool {
        self.polygon_rings(rings, polygon, false).is_some()
    }

    /// The rings of a [valid](Self::polygon) polygon in x and y, `None` if it is invalid.
    /// The empty polygon has no rings,
    /// and neither does any other when neither `xy` nor a ring check needs them read.
    fn polygon_rings(
        &self,
        rings: &ListArray<i64>,
        polygon: Range<usize>,
        xy: bool,
    ) -> Option<Vec<Vec<Coord<f64>>>> {
        let mut rings_of = polygon.map(|ring| part(rings, ring));
        if rings_of
            .clone()
            .all(|ring| ring.is_some_and(|ring| ring.is_empty()))
        {
            return Some(vec![]);
        }
        let IsValidKwargs {
            check_ring_self_intersect,
            check_polygon_self_overlap,
            ..
        } = *self.kwargs;
        // Without topology, the rings need not be read into vertices at all.
        if !check_ring_self_intersect && !check_polygon_self_overlap && !xy {
            return rings_of
                .all(|ring| ring.is_some_and(|ring| self.ring(ring)))
                .then(Vec::new);
        }
        let rings = rings_of
            .map(|ring| {
                let ring = ring.filter(|ring| self.ring(ring.clone()))?;
                let vertices = xy_vertices_validated(&GeoLineString {
                    coords: &self.coords,
                    start: ring.start,
                    end: ring.end,
                })?;
                (!check_ring_self_intersect || !vertices_self_intersect(&vertices))
                    .then_some(vertices)
            })
            .collect::<Option<Vec<_>>>()?;
        (!check_polygon_self_overlap || rings_fit(&rings)).then_some(rings)
    }
}

fn touch_at_points(matrix: &IntersectionMatrix) -> bool {
    matrix.get(CoordPos::OnBoundary, CoordPos::OnBoundary) != Dimensions::OneDimensional
}

/// Validate for polygon's rings.
/// First is exterior, others is holes.
/// See python for rules.
fn rings_fit(rings: &[Vec<Coord<f64>>]) -> bool {
    // Re-use matrix, as getting that is the expensive part.
    if rings.len() < 2 {
        return true;
    }
    let mut prepared = rings
        .iter()
        .map(|ring| PreparedGeometry::from(Polygon::new(LineString::new(ring.clone()), vec![])));
    let exterior = prepared.next().expect("at least two rings");
    let holes: Vec<_> = prepared.collect();
    holes.iter().enumerate().all(|(i, hole)| {
        let matrix = exterior.relate(hole);
        matrix.is_contains()
            && touch_at_points(&matrix)
            && holes[i + 1..].iter().all(|other| {
                let matrix = hole.relate(other);
                matrix.get(CoordPos::Inside, CoordPos::Inside) == Dimensions::Empty
                    && touch_at_points(&matrix)
            })
    })
}

/// Validate for a multipolygon's parts, each given as the rings of a valid polygon.
/// No two parts may share any area, and they may touch each other only at points.
/// See python for rules.
fn parts_fit(parts: Vec<Vec<Vec<Coord<f64>>>>) -> bool {
    let prepared: Vec<_> = parts
        .into_iter()
        .filter(|rings| !rings.is_empty())
        .map(|rings| {
            let mut rings = rings.into_iter().map(LineString::new);
            let exterior = rings.next().expect("not empty");
            PreparedGeometry::from(Polygon::new(exterior, rings.collect()))
        })
        .collect();
    prepared.iter().enumerate().all(|(i, part)| {
        prepared[i + 1..].iter().all(|other| {
            let matrix = part.relate(other);
            matrix.get(CoordPos::Inside, CoordPos::Inside) == Dimensions::Empty
                && touch_at_points(&matrix)
        })
    })
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
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanArray> {
    if kind == Kind::Point {
        let check = Checker::new(chunk, kwargs)?;
        return Ok(per_row(chunk, |i| check.point(i)));
    }
    let rows: &ListArray<i64> = downcast(chunk, "a list of parts")?;
    let parts = rows.values().as_ref();
    let row = |i: usize| part(rows, i).expect("only called on rows that are there");
    Ok(match kind {
        Kind::Point => unreachable!("handled above"),
        Kind::LineString => {
            let check = Checker::new(parts, kwargs)?;
            per_row(chunk, |i| check.line(row(i)))
        }
        Kind::MultiPoint => {
            let check = Checker::new(parts, kwargs)?;
            per_row(chunk, |i| row(i).all(|point| check.point(point)))
        }
        Kind::Polygon => {
            let rings: &ListArray<i64> = downcast(parts, "a list of rings")?;
            let check = Checker::new(rings.values().as_ref(), kwargs)?;
            per_row(chunk, |i| check.polygon(rings, row(i)))
        }
        Kind::MultiLineString => {
            let lines: &ListArray<i64> = downcast(parts, "a list of linestrings")?;
            let check = Checker::new(lines.values().as_ref(), kwargs)?;
            per_row(chunk, |i| {
                row(i).all(|line| part(lines, line).is_some_and(|line| check.line(line)))
            })
        }
        Kind::MultiPolygon => {
            let polygons: &ListArray<i64> = downcast(parts, "a list of polygons")?;
            let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
            let check = Checker::new(rings.values().as_ref(), kwargs)?;
            if !kwargs.check_multi_polygon_self_overlap {
                return Ok(per_row(chunk, |i| {
                    row(i).all(|polygon| {
                        part(polygons, polygon).is_some_and(|polygon| check.polygon(rings, polygon))
                    })
                }));
            }
            per_row(chunk, |i| {
                row(i)
                    .map(|polygon| {
                        part(polygons, polygon)
                            .and_then(|polygon| check.polygon_rings(rings, polygon, true))
                    })
                    .collect::<Option<Vec<_>>>()
                    .is_some_and(parts_fit)
            })
        }
    })
}

/// Whether each geometry in `geometries` is valid, missing where the geometry is.
/// One chunk out for every chunk in.
pub(crate) fn are_valid(
    geometries: &Series,
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_valid(chunk.as_ref(), kind, kwargs));
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
    Ok(are_valid(&inputs[0], &kwargs)?.into_series())
}

/// See `null_out_invalid`.
#[polars_expr(output_type_func=same_geometry)]
fn null_out_invalid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let valid = are_valid(geometries, &kwargs)?;
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
