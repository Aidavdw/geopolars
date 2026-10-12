//! The centroid of a geometry, in one pass over the storage:
//! - a linestring's is the midpoint of every segment, weighted by its length,
//! - a polygon's is the shoelace formula, over the planar area,
//! - a multi-geometry's is that of its parts together, each weighted by its length or area.

use std::iter::Sum;
use std::ops::{Add, Range, Sub};

use geo_traits::PolygonTrait;
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::geotraits::{LineString, Polygon};
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Geo, GeoDimension, Kind};

/// The centroids of every row of a chunk of (multi)linestrings or (multi)polygons,
/// as their coordinate struct. The flag says whether each row is a list of parts.
type ChunkCentroids = fn(&ListArray<i64>, bool) -> PolarsResult<Box<dyn Array>>;

/// How to find the centroids of a column.
struct Centroids {
    /// A point carrying the input's metadata.
    dtype: DataType,
    /// Each row is a list of parts, rather than one.
    multi: bool,
    chunk: ChunkCentroids,
}

/// Decided once per column: which fields there are, and so which loop runs.
/// A (multi)linestring's centroid keeps its dimension, a (multi)polygon's is a `PointXY`.
fn centroids_of(dtype: &DataType) -> PolarsResult<Centroids> {
    let geo = describe(dtype)?;
    let (dim, chunk): (_, ChunkCentroids) = match (geo.kind, geo.dim) {
        (Kind::Point | Kind::MultiPoint, _) => polars_bail!(
            SchemaMismatch: "centroid expects a (multi)linestring or (multi)polygon column, got: {}",
            dtype
        ),
        (Kind::Polygon | Kind::MultiPolygon, _) => (GeoDimension::XY, polygon_centroids),
        (_, dim @ GeoDimension::XY) => (dim, linestring_centroids::<2, 2>),
        (_, dim @ GeoDimension::XYZ) => (dim, linestring_centroids::<3, 3>),
        (_, dim @ GeoDimension::XYM) => (dim, linestring_centroids::<3, 2>),
        (_, dim @ GeoDimension::XYZM) => (dim, linestring_centroids::<4, 3>),
    };
    let multi = matches!(geo.kind, Kind::MultiLineString | Kind::MultiPolygon);
    Ok(Centroids {
        dtype: Geo::new(Kind::Point, dim, geo.metadata.clone()).dtype(),
        multi,
        chunk,
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

/// How much a geometry weighs (its length or area), and where.
/// (deconstructed centroid)
#[derive(Clone, Copy)]
struct Moment<const N: usize> {
    /// Σ weight * centre
    sum: [f64; N],
    /// Σ weight
    weight: f64,
}

impl<const N: usize> Moment<N> {
    /// What weighs nothing, such as an empty geometry.
    const ZERO: Self = Self {
        sum: [0.0; N],
        weight: 0.0,
    };

    /// The centroid: Σ weight * centre / Σ weight.
    /// What weighs nothing has none.
    fn centre(self) -> Option<[f64; N]> {
        (self.weight > 0.0).then(|| self.sum.map(|s| s / self.weight))
    }
}

impl<const N: usize> Add for Moment<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            sum: std::array::from_fn(|k| self.sum[k] + other.sum[k]),
            weight: self.weight + other.weight,
        }
    }
}

/// Takes a part out, such as a polygon's hole.
impl<const N: usize> Sub for Moment<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            sum: std::array::from_fn(|k| self.sum[k] - other.sum[k]),
            weight: self.weight - other.weight,
        }
    }
}

impl<const N: usize> Sum for Moment<N> {
    fn sum<I: Iterator<Item = Self>>(parts: I) -> Self {
        parts.fold(Self::ZERO, Add::add)
    }
}

/// The geometries of `chunk`: the chunk itself, or a multi-geometry's parts one list down.
fn geometries(chunk: &ListArray<i64>, multi: bool) -> PolarsResult<&ListArray<i64>> {
    if multi {
        downcast(chunk.values().as_ref(), "a list of parts")
    } else {
        Ok(chunk)
    }
}

/// The centroid of every row of `chunk`, as a coordinate struct of `dtype`,
/// out of the `moment` of each of its [`geometries`].
/// A multi-geometry's is that of its parts together.
/// Only a row can be missing: valid storage has no missing parts.
fn row_centroids<const N: usize>(
    dtype: ArrowDataType,
    chunk: &ListArray<i64>,
    multi: bool,
    moment: impl Fn(usize) -> Moment<N>,
) -> Box<dyn Array> {
    points(
        dtype,
        (0..chunk.len()).map(|i| {
            if chunk.is_null(i) {
                return None;
            }
            let moment = if multi {
                let (start, end) = chunk.offsets().start_end(i);
                (start..end).map(&moment).sum()
            } else {
                moment(i)
            };
            moment.centre()
        }),
    )
}

/// The first `N` fields of one chunk of coordinates
struct Coords<'a, const N: usize> {
    fields: [&'a [f64]; N],
}

impl<'a, const N: usize> Coords<'a, N> {
    fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let fields = coords.values()[..N]
            .iter()
            .map(|field| {
                let field: &PrimitiveArray<f64> = downcast(field.as_ref(), "f64 coordinates")?;
                Ok(field.values().as_slice())
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let fields = fields.try_into().unwrap_or_else(|_| unreachable!());
        Ok(Self { fields })
    }

    /// Coordinate `i`.
    #[inline(always)]
    fn get(&self, i: usize) -> [f64; N] {
        self.fields.map(|field| field[i])
    }
}

/// The moment of the coordinates `vertices`, as a line:
/// Σ len (a + b) / 2 and Σ len over its segments `ab`,
/// where `len` is measured over the first `S` fields (xy, or xyz),
/// and every field (incl. m) is averaged by it.
fn line_moment<const N: usize, const S: usize>(
    coords: &Coords<N>,
    vertices: Range<usize>,
) -> Moment<N> {
    let mut line = vertices.map(|i| coords.get(i));
    let Some(mut previous) = line.next() else {
        return Moment::ZERO;
    };
    let (mut sum, mut total) = ([0.0; N], 0.0);
    for here in line {
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
    Moment {
        sum: sum.map(|s| s / 2.0),
        weight: total,
    }
}

/// See [`ChunkCentroids`], for (multi)linestrings: their centroids have the same fields.
/// `N` and `S` are fixed per dimension, so the loop never asks which fields there are.
fn linestring_centroids<const N: usize, const S: usize>(
    chunk: &ListArray<i64>,
    multi: bool,
) -> PolarsResult<Box<dyn Array>> {
    let lines = geometries(chunk, multi)?;
    let coords = Coords::<N>::new(lines.values().as_ref())?;
    Ok(row_centroids(
        lines.values().dtype().clone(),
        chunk,
        multi,
        |i| {
            let (start, end) = lines.offsets().start_end(i);
            line_moment::<N, S>(&coords, start..end)
        },
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

/// A polygon's moment, weighted by its planar area:
/// its exterior ring, less the holes inside it, whichever way each of them runs.
/// The formula is C = M / 3a, with `a` twice the area: weighted by the area, M / 6.
fn polygon_moment(polygon: &Polygon) -> Moment<2> {
    let Some(exterior) = polygon.exterior() else {
        return Moment::ZERO;
    };
    let oriented = |ring: &LineString| {
        let (a, mx, my) = ring_moments(ring);
        Moment {
            sum: [mx, my].map(|m| a.signum() * m / 6.0),
            weight: a.abs() / 2.0,
        }
    };
    polygon
        .interiors()
        .fold(oriented(&exterior), |polygon, hole| {
            polygon - oriented(&hole)
        })
}

/// See [`ChunkCentroids`], for (multi)polygons: their centroids are xy.
fn polygon_centroids(chunk: &ListArray<i64>, multi: bool) -> PolarsResult<Box<dyn Array>> {
    let polygons = geometries(chunk, multi)?;
    let rings = downcast::<ListArray<i64>>(polygons.values().as_ref(), "a list of rings")?;
    let coords = CoordsView::new(rings.values().as_ref())?;
    let xy = GeoDimension::XY
        .coordinates()
        .to_arrow(CompatLevel::newest());
    Ok(row_centroids(xy, chunk, multi, |i| {
        let (start, end) = polygons.offsets().start_end(i);
        let polygon = Polygon {
            rings,
            coords: &coords,
            start,
            end,
        };
        polygon_moment(&polygon)
    }))
}

/// See python `centroid`.
#[polars_expr(output_type_func=centroid_type)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let Centroids {
        dtype,
        multi,
        chunk: centroids,
    } = centroids_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();

    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|chunk| centroids(chunk, multi))
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `dtype` describes.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
