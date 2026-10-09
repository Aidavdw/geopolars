//! The centroid of a polygon: the shoelace formula, over the planar area,
//! in one pass over the storage.

use geo_traits::PolygonTrait;
use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;

use super::area::planar_area;
use crate::geoarrow::geotraits::{for_each_geometry, Geometry, LineString, Polygon};
use crate::geoarrow::{describe, Geo, GeoDimension, Kind};

/// The `PointXY` a centroid of `dtype` is, carrying its metadata.
fn centroid_of(dtype: &DataType) -> PolarsResult<Geo> {
    let geo = describe(dtype)?;
    polars_ensure!(
        geo.kind == Kind::Polygon,
        SchemaMismatch: "centroid expects a `{}` column, got: {}", Kind::Polygon.name(), dtype
    );
    Ok(Geo::new(
        Kind::Point,
        GeoDimension::XY,
        geo.metadata.clone(),
    ))
}

/// `output_type_func` for [`centroid`].
fn centroid_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let point = centroid_of(input_fields[0].dtype())?;
    Ok(Field::new(input_fields[0].name().clone(), point.dtype()))
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

/// See `centroid`.
#[polars_expr(output_type_func=centroid_type)]
fn centroid(inputs: &[Series]) -> PolarsResult<Series> {
    let point = centroid_of(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();

    let mut x = Vec::with_capacity(storage.len());
    let mut y = Vec::with_capacity(storage.len());
    let mut valid = Vec::with_capacity(storage.len());
    for chunk in storage.chunks() {
        for_each_geometry(chunk.as_ref(), Kind::Polygon, |geometry| {
            let centroid = match geometry {
                Some(Geometry::Polygon(polygon)) => {
                    from_moments(moments(&polygon), planar_area(&polygon))
                }
                _ => None,
            };
            let (cx, cy) = centroid.unwrap_or((0.0, 0.0));
            x.push(cx);
            y.push(cy);
            valid.push(centroid.is_some());
            Ok(())
        })?;
    }

    let n = valid.len();
    let x = Float64Chunked::from_vec("x".into(), x).into_series();
    let y = Float64Chunked::from_vec("y".into(), y).into_series();
    let coords = StructChunked::from_series(inputs[0].name().clone(), n, [x, y].iter())?
        .with_outer_validity(Some(Bitmap::from_iter(valid)));
    Ok(coords.into_series().into_extension(point.instance()))
}
