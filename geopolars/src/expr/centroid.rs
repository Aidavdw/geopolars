//! The centroid of a geometry, in one pass over the storage:
//! - a linestring's is the midpoint of every segment, weighted by its length,
//! - a polygon's is the shoelace formula, over the planar area.

use geo_traits::PolygonTrait;
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;

use super::area::planar_area;
use crate::geoarrow::geotraits::{for_each_geometry, Geometry, LineString, Polygon};
use crate::geoarrow::storage::downcast;
use crate::geoarrow::{describe, Geo, GeoDimension, Kind};

/// The point a centroid of `dtype` is, carrying its metadata:
/// a linestring's keeps its dimension, a polygon's is a `PointXY`.
fn centroid_of(dtype: &DataType) -> PolarsResult<Geo> {
    let geo = describe(dtype)?;
    let dim = match geo.kind {
        Kind::LineString => geo.dim,
        Kind::Polygon => GeoDimension::XY,
        Kind::Point | Kind::MultiPoint | Kind::MultiLineString | Kind::MultiPolygon => {
            polars_bail!(
                SchemaMismatch: "centroid expects a `{}` or `{}` column, got: {}",
                Kind::LineString.name(), Kind::Polygon.name(), dtype
            )
        }
    };
    Ok(Geo::new(Kind::Point, dim, geo.metadata.clone()))
}

/// `output_type_func` for [`centroid`].
fn centroid_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let point = centroid_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
}

struct Coords<'a, const N: usize> {
    coords: &'a StructArray,
    fields: [&'a PrimitiveArray<f64>; N],
}

impl<'a, const N: usize> Coords<'a, N> {
    fn new(coords: &'a dyn Array) -> PolarsResult<Self> {
        let coords: &StructArray = downcast(coords, "a coordinate struct")?;
        let mut fields = Vec::with_capacity(N);
        for field in &coords.values()[..N] {
            fields.push(downcast(field.as_ref(), "f64 coordinates")?);
        }
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
    if start == end {
        return None;
    }
    let mut previous = coords.get(start)?;
    let (mut sum, mut total) = ([0.0; N], 0.0);
    for i in start + 1..end {
        let here = coords.get(i)?;
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

/// The centroid of every linestring, into `out` (one `Vec` per field) and `valid`.
/// `N` and `S` are fixed per dimension, so the loop never asks which fields there are.
fn linestrings<const N: usize, const S: usize>(
    storage: &Series,
    out: &mut [Vec<f64>],
    valid: &mut Vec<bool>,
) -> PolarsResult<()> {
    for lines in storage.list()?.downcast_iter() {
        let coords = Coords::<N>::new(lines.values().as_ref())?;
        for i in 0..lines.len() {
            let (start, end) = lines.offsets().start_end(i);
            let centroid = lines
                .is_valid(i)
                .then(|| line_centroid::<N, S>(&coords, start, end))
                .flatten();
            let point = centroid.unwrap_or([0.0; N]);
            for (out, value) in out.iter_mut().zip(point) {
                out.push(value);
            }
            valid.push(centroid.is_some());
        }
    }
    Ok(())
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
fn from_moments((mx, my): (f64, f64), area: f64) -> Option<(f64, f64)> {
    (area > 0.0).then(|| (mx / (6.0 * area), my / (6.0 * area)))
}

/// The centroid of every polygon, into `out` (x and y) and `valid`.
fn polygons(storage: &Series, out: &mut [Vec<f64>], valid: &mut Vec<bool>) -> PolarsResult<()> {
    for chunk in storage.chunks() {
        for_each_geometry(chunk.as_ref(), Kind::Polygon, |geometry| {
            let centroid = match geometry {
                Some(Geometry::Polygon(polygon)) => {
                    from_moments(moments(&polygon), planar_area(&polygon))
                }
                _ => None,
            };
            let (cx, cy) = centroid.unwrap_or((0.0, 0.0));
            out[0].push(cx);
            out[1].push(cy);
            valid.push(centroid.is_some());
            Ok(())
        })?;
    }
    Ok(())
}

/// See `centroid`.
#[polars_expr(output_type_func=centroid_type)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let point = centroid_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let names = point.dim().field_names();

    let mut out = vec![Vec::with_capacity(storage.len()); names.len()];
    let mut valid = Vec::with_capacity(storage.len());
    // Which fields there are is decided here, once: each arm is its own loop.
    match (describe(inputs[0].dtype())?.kind, point.dim()) {
        (Kind::LineString, GeoDimension::XY) => linestrings::<2, 2>(storage, &mut out, &mut valid),
        (Kind::LineString, GeoDimension::XYZ) => linestrings::<3, 3>(storage, &mut out, &mut valid),
        (Kind::LineString, GeoDimension::XYM) => linestrings::<3, 2>(storage, &mut out, &mut valid),
        (Kind::LineString, GeoDimension::XYZM) => {
            linestrings::<4, 3>(storage, &mut out, &mut valid)
        }
        _ => polygons(storage, &mut out, &mut valid),
    }?;

    let fields = names
        .iter()
        .zip(out)
        .map(|(name, values)| Float64Chunked::from_vec((*name).into(), values).into_series())
        .collect::<Vec<_>>();
    let coords = StructChunked::from_series(inputs[0].name().clone(), valid.len(), fields.iter())?
        .with_outer_validity(Some(Bitmap::from_iter(valid)));
    Ok(coords.into_series().into_extension(point.instance()))
}
