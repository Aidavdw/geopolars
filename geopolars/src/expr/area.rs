//! The area a geometry encloses along the ellipsoid, for geometries that declare a CRS.
//!
//! Planar areas are written in plain Polars expressions instead, on the Python side.

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use super::distance::GeodesicMetric;
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

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

/// `output_type_func` for [`area_geodesic`]: an `f64`, in the unit of the CRS squared,
/// or a list of them for a multipolygon.
fn areas(input_fields: &[Field]) -> PolarsResult<Field> {
    crs_of(&input_fields[0])?;
    let dtype = match describe(input_fields[0].dtype())?.kind {
        Kind::MultiPolygon => DataType::List(Box::new(DataType::Float64)),
        _ => DataType::Float64,
    };
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// View over polygons
struct Polygons<'a> {
    polygons: &'a ListArray<i64>,
    rings: &'a ListArray<i64>,
    coords: CoordsView<'a>,
}

impl<'a> Polygons<'a> {
    fn new(polygons: &'a ListArray<i64>) -> PolarsResult<Self> {
        let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
        let coords = CoordsView::new(rings.values().as_ref())?;
        Ok(Self {
            polygons,
            rings,
            coords,
        })
    }

    fn ring_area(&self, ring: usize, metric: &GeodesicMetric) -> PolarsResult<Option<f64>> {
        if self.rings.is_null(ring) {
            return Ok(None);
        }
        let (start, end) = self.rings.offsets().start_end(ring);
        metric.ring_area((start..end).map(|i| self.coords.xy(i)))
    }

    /// The area of polygon `i`: its exterior ring, less the holes inside it.
    /// `None` if it, or any of its rings or coordinates, is missing.
    fn area(&self, i: usize, metric: &GeodesicMetric) -> PolarsResult<Option<f64>> {
        if self.polygons.is_null(i) {
            return Ok(None);
        }
        let (start, end) = self.polygons.offsets().start_end(i);
        let areas: Option<Vec<f64>> = (start..end)
            .map(|ring| self.ring_area(ring, metric))
            .collect::<PolarsResult<_>>()?;
        // No rings at all is an empty polygon, which encloses nothing.
        Ok(areas.map(|areas| match areas.split_first() {
            None => 0.0,
            Some((exterior, holes)) => exterior - holes.iter().sum::<f64>(),
        }))
    }
}

/// The area of every polygon along its CRS's ellipsoid.
fn polygons(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.list()?.downcast_iter() {
        let polygons = Polygons::new(chunk)?;
        for i in 0..chunk.len() {
            out.push(polygons.area(i, metric)?);
        }
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// The area of each part of every multipolygon, in order.
/// A missing part renders the entire operation as missing.
fn multipolygons(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<ListChunked> {
    let name = storage.name().clone();
    let chunks = storage
        .list()?
        .downcast_iter()
        .map(|chunk| -> PolarsResult<_> {
            let polygons = Polygons::new(downcast(chunk.values().as_ref(), "a list of polygons")?)?;
            // Only the parts this chunk points at: it may be a slice of a larger one.
            let (first, last) = (
                *chunk.offsets().first() as usize,
                *chunk.offsets().last() as usize,
            );
            let parts = (first..last)
                .map(|polygon| polygons.area(polygon, metric))
                .collect::<PolarsResult<PrimitiveArray<f64>>>()?;
            let offsets = Offsets::try_from_lengths(chunk.offsets().lengths())?;
            Ok(ListArray::<i64>::new(
                ListArray::<i64>::default_datatype(ArrowDataType::Float64),
                offsets.into(),
                parts.boxed(),
                chunk.validity().cloned(),
            ))
        });
    ListChunked::try_from_chunk_iter(name, chunks)
}

/// See `area`.
#[polars_expr(output_type_func=areas)]
fn area_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let crs = crs_of(&inputs[0].field())?;
    let storage = inputs[0].ext()?.storage();

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::Polygon => polygons(storage, &*GeodesicMetric::of(&crs)?)?.into_series(),
        Kind::MultiPolygon => multipolygons(storage, &*GeodesicMetric::of(&crs)?)?.into_series(),
        // Still refuses a CRS that has no ellipsoid, as a polygon in it would be.
        Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString => {
            GeodesicMetric::of(&crs)?;
            nothing_enclosed(storage).into_series()
        }
    };
    Ok(out.with_name(inputs[0].name().clone()))
}
