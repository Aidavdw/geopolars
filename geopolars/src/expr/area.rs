//! The area a geometry encloses:
//! - on the plane its coordinates lie in ([`area_planar`]),
//! - along the ellipsoid of its CRS ([`area_geodesic`]).

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use geo_traits::{MultiPolygonTrait, PolygonTrait};

use super::distance::GeodesicMetric;
use crate::geoarrow::geotraits::{for_each_geometry, Geometry, LineString, Polygon};
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// Twice the signed area of one closed ring, positive when it runs CCW.
/// Like `_twice_signed` in Python: relative to its first coordinate,
/// and without the repeated last one.
fn twice_signed(ring: &LineString) -> f64 {
    if ring.end - ring.start < 2 {
        return 0.0;
    }
    let (x, y) = (|i| ring.coords.nth(i, 0), |i| ring.coords.nth(i, 1));
    let (x0, y0) = (x(ring.start), y(ring.start));
    (ring.start..ring.end - 1)
        .map(|i| (x(i) - x0) * (y(i + 1) - y0) - (x(i + 1) - x0) * (y(i) - y0))
        .sum()
}

/// The area of a polygon on the plane: its exterior ring, less the holes inside it.
pub(crate) fn planar_area(polygon: &Polygon) -> f64 {
    let Some(exterior) = polygon.exterior() else {
        return 0.0;
    };
    let holes: f64 = polygon
        .interiors()
        .map(|ring| twice_signed(&ring).abs())
        .sum();
    (twice_signed(&exterior).abs() - holes) / 2.0
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

/// Points are refused rather than given an area of `0.0`.
fn points_have_no_area(kind: Kind) -> PolarsError {
    polars_err!(SchemaMismatch: "area does not accept a `{}` column", kind.name())
}

/// An `f64` per geometry, or a list of them (one per part) for a multipolygon.
fn area_dtype(field: &Field) -> PolarsResult<Field> {
    let dtype = match describe(field.dtype())?.kind {
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(points_have_no_area(kind)),
        Kind::MultiPolygon => DataType::List(Box::new(DataType::Float64)),
        _ => DataType::Float64,
    };
    Ok(Field::new(field.name().clone(), dtype))
}

/// `output_type_func` for [`area_planar`]: in the unit of the coordinates squared.
fn areas_planar(input_fields: &[Field]) -> PolarsResult<Field> {
    area_dtype(&input_fields[0])
}

/// `output_type_func` for [`area_geodesic`]: in the unit of the CRS squared.
fn areas(input_fields: &[Field]) -> PolarsResult<Field> {
    // Check if the CRS can be used by PROJ. If not, raise error at plan time rather than at runtime.
    GeodesicMetric::of(&crs_of(&input_fields[0])?)?;
    area_dtype(&input_fields[0])
}

/// The planar area of every polygon.
fn polygons_planar(storage: &Series) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.chunks() {
        for_each_geometry(chunk.as_ref(), Kind::Polygon, |geometry| {
            out.push(match geometry {
                Some(Geometry::Polygon(polygon)) => Some(planar_area(&polygon)),
                _ => None,
            });
            Ok(())
        })?;
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// The planar area of each part of every multipolygon, in order.
fn multipolygons_planar(storage: &Series) -> PolarsResult<ListChunked> {
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        storage.name().clone(),
        storage.len(),
        storage.len(),
        DataType::Float64,
    );
    for chunk in storage.chunks() {
        for_each_geometry(chunk.as_ref(), Kind::MultiPolygon, |geometry| {
            match geometry {
                Some(Geometry::MultiPolygon(multipolygon)) => {
                    let parts: Vec<f64> = multipolygon
                        .polygons()
                        .map(|polygon| planar_area(&polygon))
                        .collect();
                    builder.append_slice(&parts);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;
    }
    Ok(builder.finish())
}

/// See `area_planar`.
#[polars_expr(output_type_func=areas_planar)]
fn area_planar(inputs: &[Series]) -> PolarsResult<Series> {
    let storage = inputs[0].ext()?.storage();
    let out = match describe(inputs[0].dtype())?.kind {
        Kind::Polygon => polygons_planar(storage)?.into_series(),
        Kind::MultiPolygon => multipolygons_planar(storage)?.into_series(),
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(points_have_no_area(kind)),
        Kind::LineString | Kind::MultiLineString => nothing_enclosed(storage).into_series(),
    };
    Ok(out.with_name(inputs[0].name().clone()))
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
        kind @ (Kind::Point | Kind::MultiPoint) => return Err(points_have_no_area(kind)),
        // Still refuses a CRS that has no ellipsoid, as a polygon in it would be.
        Kind::LineString | Kind::MultiLineString => {
            GeodesicMetric::of(&crs)?;
            nothing_enclosed(storage).into_series()
        }
    };
    Ok(out.with_name(inputs[0].name().clone()))
}
