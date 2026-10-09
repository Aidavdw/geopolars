//! The centroid of a geometry, in one pass over the storage:
//! - a linestring's is the midpoint of every segment, weighted by its length,
//! - a polygon's is the shoelace formula, over the planar area,
//! - a multi-geometry gets a list of them, one per part.

use std::ops::Range;

use geo_traits::PolygonTrait;
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use super::area::planar_area;
use crate::geoarrow::geotraits::{LineString, Polygon};
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Geo, GeoDimension, Kind};

// TODO: right now, for multi* it returns a list of centroids.
// We need an operation which gives a shared centroid too.

/// The centroids of rows `rows` of a list of linestrings or polygons,
/// as their coordinate struct.
type PartCentroids = fn(&ListArray<i64>, Range<usize>) -> PolarsResult<Box<dyn Array>>;

/// How to find the centroids of a column.
struct Centroids {
    /// A point carrying the input's metadata, or a list of them for a multi-geometry.
    dtype: DataType,
    /// Each row is a list of parts, rather than one.
    multi: bool,
    part: PartCentroids,
}

/// Decided once per column: which fields there are, and so which loop runs.
/// A (multi)linestring's centroid keeps its dimension, a (multi)polygon's is a `PointXY`.
fn centroids_of(dtype: &DataType) -> PolarsResult<Centroids> {
    let geo = describe(dtype)?;
    let (dim, part): (_, PartCentroids) = match (geo.kind, geo.dim) {
        (Kind::Point | Kind::MultiPoint, _) => polars_bail!(
            SchemaMismatch: "centroid expects a (multi)linestring or (multi)polygon column, got: {}",
            dtype
        ),
        (Kind::Polygon | Kind::MultiPolygon, _) => (GeoDimension::XY, polygons),
        (_, dim @ GeoDimension::XY) => (dim, linestrings::<2, 2>),
        (_, dim @ GeoDimension::XYZ) => (dim, linestrings::<3, 3>),
        (_, dim @ GeoDimension::XYM) => (dim, linestrings::<3, 2>),
        (_, dim @ GeoDimension::XYZM) => (dim, linestrings::<4, 3>),
    };
    let multi = matches!(geo.kind, Kind::MultiLineString | Kind::MultiPolygon);
    let point = Geo::new(Kind::Point, dim, geo.metadata.clone()).dtype();
    Ok(Centroids {
        dtype: if multi {
            DataType::List(Box::new(point))
        } else {
            point
        },
        multi,
        part,
    })
}

/// `output_type_func` for [`centroid`].
fn centroid_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let centroids = centroids_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), centroids.dtype))
}

/// A coordinate struct of `dtype` out of `centroids`, missing where there is none.
fn points<const N: usize>(
    dtype: ArrowDataType,
    centroids: impl Iterator<Item = Option<[f64; N]>>,
) -> Box<dyn Array> {
    let mut fields: [Vec<f64>; N] = std::array::from_fn(|_| Vec::new());
    let mut valid = Vec::new();
    for centroid in centroids {
        for (field, value) in fields.iter_mut().zip(centroid.unwrap_or([0.0; N])) {
            field.push(value);
        }
        valid.push(centroid.is_some());
    }
    let fields = fields.map(|field| PrimitiveArray::from_vec(field).boxed());
    StructArray::new(
        dtype,
        valid.len(),
        fields.into(),
        Some(Bitmap::from_iter(valid)),
    )
    .boxed()
}

struct Coords<'a, const N: usize> {
    coords: &'a StructArray,
    fields: [&'a PrimitiveArray<f64>; N],
}

impl<'a, const N: usize> Coords<'a, N> {
    fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let fields = coords.values()[..N]
            .iter()
            .map(|field| downcast(field.as_ref(), "f64 coordinates"))
            .collect::<PolarsResult<Vec<_>>>()?;
        let fields = fields.try_into().unwrap_or_else(|_| unreachable!());
        Ok(Self { coords, fields })
    }

    /// Coordinate `i`, `None` if it or any of its fields is missing.
    fn get(&self, i: usize) -> Option<[f64; N]> {
        let present = self.coords.is_valid(i) && self.fields.iter().all(|f| f.is_valid(i));
        present.then(|| self.fields.map(|f| f.value(i)))
    }
}

/// The centroid of the coordinates `start..end`, as a line:
/// Σ len (a + b) / 2 Σ len over its segments `ab`,
/// where `len` is measured over the first `S` fields (xy, or xyz),
/// and every field (incl. m) is averaged by it.
/// `None` if a coordinate is missing, or the line has no length.
fn line_centroid<const N: usize, const S: usize>(
    coords: &Coords<N>,
    start: usize,
    end: usize,
) -> Option<[f64; N]> {
    let mut line = (start..end).map(|i| coords.get(i));
    let mut previous = line.next()??;
    let (mut sum, mut total) = ([0.0; N], 0.0);
    for here in line {
        let here = here?;
        let len = (0..S)
            .map(|k| (here[k] - previous[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        for k in 0..N {
            sum[k] += len * (previous[k] + here[k]);
        }
        total += len;
        previous = here;
    }
    (total > 0.0).then(|| sum.map(|s| s / (2.0 * total)))
}

/// See [`PartCentroids`], for linestrings: their centroids have the same fields.
/// `N` and `S` are fixed per dimension, so the loop never asks which fields there are.
fn linestrings<const N: usize, const S: usize>(
    lines: &ListArray<i64>,
    rows: Range<usize>,
) -> PolarsResult<Box<dyn Array>> {
    let coords = Coords::<N>::new(lines.values().as_ref())?;
    Ok(points(
        lines.values().dtype().clone(),
        rows.map(|i| {
            let (start, end) = lines.offsets().start_end(i);
            lines
                .is_valid(i)
                .then(|| line_centroid::<N, S>(&coords, start, end))?
        }),
    ))
}

/// Σ cross, Σ (x_i + x_{i+1}) cross and Σ (y_i + y_{i+1}) cross over one closed ring,
/// with cross = x_i y_{i+1} - x_{i+1} y_i. Signed by which way the ring runs.
fn ring_moments(ring: &LineString) -> (f64, f64, f64) {
    let (x, y) = (|i| ring.coords.nth(i, 0), |i| ring.coords.nth(i, 1));
    // The repeated last coordinate is only ever the `i + 1`.
    (ring.start..ring.end.saturating_sub(1)).fold((0.0, 0.0, 0.0), |(a, mx, my), i| {
        let cross = x(i) * y(i + 1) - x(i + 1) * y(i);
        (
            a + cross,
            mx + (x(i) + x(i + 1)) * cross,
            my + (y(i) + y(i + 1)) * cross,
        )
    })
}

/// A polygon's moments: its exterior ring, less the holes inside it,
/// whichever way each of them runs.
fn moments(polygon: &Polygon) -> (f64, f64) {
    let Some(exterior) = polygon.exterior() else {
        return (0.0, 0.0);
    };
    let oriented = |ring: &LineString| {
        let (a, mx, my) = ring_moments(ring);
        (a.signum() * mx, a.signum() * my)
    };
    polygon
        .interiors()
        .fold(oriented(&exterior), |(mx, my), hole| {
            let (hx, hy) = oriented(&hole);
            (mx - hx, my - hy)
        })
}

/// The formula: C = M / 6A. Nothing enclosed has no centroid.
fn polygon_centroid(polygon: &Polygon) -> Option<[f64; 2]> {
    let ((mx, my), area) = (moments(polygon), planar_area(polygon));
    (area > 0.0).then(|| [mx / (6.0 * area), my / (6.0 * area)])
}

/// See [`PartCentroids`], for polygons: their centroids are xy.
fn polygons(polygons: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<Box<dyn Array>> {
    let rings = downcast::<ListArray<i64>>(polygons.values().as_ref(), "a list of rings")?;
    let coords = CoordsView::new(rings.values().as_ref())?;
    let xy = GeoDimension::XY
        .coordinates()
        .to_arrow(CompatLevel::newest());
    Ok(points(
        xy,
        rows.map(|i| {
            let (start, end) = polygons.offsets().start_end(i);
            let polygon = Polygon {
                rings,
                coords: &coords,
                start,
                end,
            };
            polygons.is_valid(i).then(|| polygon_centroid(&polygon))?
        }),
    ))
}

/// See `centroid`.
#[polars_expr(output_type_func=centroid_type)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let Centroids { dtype, multi, part } = centroids_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();

    let chunks = storage.list()?.downcast_iter().map(|chunk| {
        if !multi {
            return part(chunk, 0..chunk.len());
        }
        // A multi-geometry's parts sit one list down, and so do their centroids.
        let parts = downcast::<ListArray<i64>>(chunk.values().as_ref(), "a list of parts")?;
        // Only the parts this chunk points at: it may be a slice of a larger one.
        let (first, last) = (
            *chunk.offsets().first() as usize,
            *chunk.offsets().last() as usize,
        );
        let points = part(parts, first..last)?;
        Ok(ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(points.dtype().clone()),
            Offsets::try_from_lengths(chunk.offsets().lengths())?.into(),
            points,
            chunk.validity().cloned(),
        )
        .boxed())
    });
    let chunks = chunks.collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `dtype` describes.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
