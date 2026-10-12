//! Whether two rings overlap, which is when the areas they enclose share area.

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use rsgeo::coordinate_position::CoordPos;
use rsgeo::dimensions::Dimensions;
use rsgeo::relate::IntersectionMatrix;
use rsgeo::{LineString, Polygon, Relate};
use serde::Deserialize;

use super::self_intersects::{same_xy, xy_vertices_validated};
use crate::geoarrow::geotraits::LineString as GeoLineString;
use crate::geoarrow::storage::{CoordsView, MIN_RING_VERTICES};
use crate::geoarrow::{describe, ensure_same_crs, Kind};

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RingsOverlapKwargs {
    /// Whether rings that only touch overlap too.
    include_touching: bool,
}

/// `output_type_func` for [`rings_overlap`]: a `bool`, named after the first ring.
///
/// Both have to be linestrings, in the same CRS (or neither declaring one).
fn rings_overlap_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let [a, b] = input_fields else {
        polars_bail!(ComputeError: "rings_overlap takes two rings, got {} inputs", input_fields.len());
    };
    let (geo_a, geo_b) = (describe(a.dtype())?, describe(b.dtype())?);
    for (field, geo) in [(a, &geo_a), (b, &geo_b)] {
        polars_ensure!(
            geo.kind == Kind::LineString,
            SchemaMismatch: "rings_overlap expects two `{}` columns, got: {}",
            Kind::LineString.name(), field.dtype()
        );
    }
    ensure_same_crs(&geo_a, &geo_b)?;
    Ok(Field::new(a.name().clone(), DataType::Boolean))
}

/// The ring at `start..end` as the polygon it encloses, in x and y.
/// `None` if one of its vertices is missing, or not finite in x or y.
/// Errors if it is not a ring.
fn ring(coords: &CoordsView, start: usize, end: usize) -> PolarsResult<Option<Polygon<f64>>> {
    let Some(vertices) = xy_vertices_validated(&GeoLineString { coords, start, end }) else {
        return Ok(None);
    };
    polars_ensure!(
        vertices.len() >= MIN_RING_VERTICES && same_xy(vertices[0], vertices[vertices.len() - 1]),
        ComputeError: "rings_overlap expects rings (at least {MIN_RING_VERTICES} vertices, \
        the last at the first), got a linestring of {} vertices that is not one",
        vertices.len()
    );
    Ok(Some(Polygon::new(LineString::new(vertices), vec![])))
}

/// One side: its rings and their coordinates, as a single chunk.
struct Rings<'a> {
    rings: &'a ListArray<i64>,
    coords: CoordsView<'a>,
}

impl<'a> Rings<'a> {
    fn new(rings: &'a ListArray<i64>) -> PolarsResult<Self> {
        let coords = CoordsView::new(rings.values().as_ref())?;
        Ok(Self { rings, coords })
    }

    /// Ring `i` (or the only one, for a side of one row), `None` where it is missing.
    fn get(&self, i: usize) -> PolarsResult<Option<Polygon<f64>>> {
        let i = if self.rings.len() == 1 { 0 } else { i };
        if !self.rings.is_valid(i) {
            return Ok(None);
        }
        let (start, end) = self.rings.offsets().start_end(i);
        ring(&self.coords, start, end)
    }
}

/// Gets the [IntersectionMatrix] between two rings.
pub(crate) fn relate_rings(
    a: &Series,
    b: &Series,
) -> PolarsResult<Vec<Option<IntersectionMatrix>>> {
    let a = a.ext()?.storage().list()?.rechunk();
    let b = b.ext()?.storage().list()?.rechunk();
    let (a, b) = (
        Rings::new(a.downcast_as_array())?,
        Rings::new(b.downcast_as_array())?,
    );
    let len = match (a.rings.len(), b.rings.len()) {
        (n, m) if n == m => n,
        (1, m) => m,
        (n, 1) => n,
        (n, m) => polars_bail!(ShapeMismatch: "cannot compare {n} rings against {m}"),
    };

    (0..len)
        .map(|i| {
            let (Some(a), Some(b)) = (a.get(i)?, b.get(i)?) else {
                return Ok(None);
            };
            Ok(Some(a.relate(&b)))
        })
        .collect()
}

/// See `rings_overlap`.
#[polars_expr(output_type_func=rings_overlap_type)]
fn rings_overlap(inputs: &[Series], kwargs: RingsOverlapKwargs) -> PolarsResult<Series> {
    let out: BooleanChunked = relate_rings(&inputs[0], &inputs[1])?
        .into_iter()
        .map(|matrix| {
            matrix.map(|matrix| {
                if kwargs.include_touching {
                    matrix.is_intersects()
                } else {
                    matrix.get(CoordPos::Inside, CoordPos::Inside) != Dimensions::Empty
                }
            })
        })
        .collect();
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}
