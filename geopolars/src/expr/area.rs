//! The area a geometry encloses:
//! - on the plane, computed by `rsgeo`,
//! - along the ellipsoid, for geometries that declare a CRS.
//!
//! Planar areas are also written in plain Polars expressions, on the Python side.

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use rsgeo::Area;

use super::distance::GeodesicMetric;
use crate::geoarrow::to_rsgeo::{downcast, Coords};
use crate::geoarrow::{describe, to_rsgeo, Kind};

// TODO: Move this up into a shareable thing?
/// `output_type_func` for an operation that gives one `f64` per geometry.
fn float_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Float64))
}

/// The planar area of every geometry. Only a polygon encloses anything.
/// Everything else is `0.0`, and a missing geometry has no area.
#[polars_expr(output_type_func=float_output)]
fn area_rsgeo(inputs: &[Series]) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let storage = inputs[0].ext()?.storage();
    let name = inputs[0].name().clone();

    let out = match geo.kind {
        Kind::Polygon => Float64Chunked::from_iter_options(
            name,
            to_rsgeo::polygons(storage)?.map(|polygon| polygon.map(|p| p.unsigned_area())),
        ),
        Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => {
            nothing_enclosed(storage)
        }
    };
    Ok(out.into_series())
}

/// `0.0` for every geometry that is there, for kinds that cannot enclose anything.
fn nothing_enclosed(storage: &Series) -> Float64Chunked {
    Float64Chunked::from_iter_options(
        storage.name().clone(),
        storage
            .is_not_null()
            .iter()
            .map(|present| (present == Some(true)).then_some(0.0)),
    )
}

/// The CRS of a column to measure an area in. Type operation only.
fn crs_of(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geo.metadata.crs()
}

/// `output_type_func` for [`area_geodesic`]: an `f64`, in metres².
fn square_metres(input_fields: &[Field]) -> PolarsResult<Field> {
    crs_of(&input_fields[0])?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

/// The area of every polygon along its CRS's ellipsoid:
/// its exterior ring, less the holes inside it.
fn polygons(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.list()?.downcast_iter() {
        let rings: &ListArray<i64> = downcast(chunk.values().as_ref(), "a list of rings")?;
        let coords = Coords::new(rings.values().as_ref())?;
        let ring_area = |ring: usize| {
            if rings.is_null(ring) {
                return Ok(None);
            }
            let (start, end) = rings.offsets().start_end(ring);
            metric.ring_area((start..end).map(|i| coords.xy(i)))
        };

        for i in 0..chunk.len() {
            if chunk.is_null(i) {
                out.push(None);
                continue;
            }
            let (start, end) = chunk.offsets().start_end(i);
            let areas: Option<Vec<f64>> =
                (start..end).map(ring_area).collect::<PolarsResult<_>>()?;
            // No rings at all is an empty polygon, which encloses nothing.
            out.push(areas.map(|areas| match areas.split_first() {
                None => 0.0,
                Some((exterior, holes)) => exterior - holes.iter().sum::<f64>(),
            }));
        }
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// The area of every geometry along its CRS's ellipsoid, in metres².
/// Only a polygon encloses anything; `z` and `m` are left out.
/// Everything else is `0.0`, and a missing geometry has no area.
#[polars_expr(output_type_func=square_metres)]
fn area_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let crs = crs_of(&inputs[0].field())?;
    let storage = inputs[0].ext()?.storage();

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::Polygon => polygons(storage, &*GeodesicMetric::of(&crs)?)?,
        // Still refuses a CRS that has no ellipsoid, as a polygon in it would be.
        Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => {
            GeodesicMetric::of(&crs)?;
            nothing_enclosed(storage)
        }
    };
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}
