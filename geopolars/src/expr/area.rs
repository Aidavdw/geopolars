//! The area a geometry encloses:
//! - on the plane its coordinates lie in ([`area_planar`]),
//! - along the ellipsoid of its CRS ([`area_geodesic`]).

use geoarrow_abstr::{GeoArrowCoordsShared, GeoArrowCoordsView};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;

use geo_traits::{MultiPolygonTrait, PolygonTrait};

use super::distance::GeodesicMetric;
use crate::geoarrow::geotraits::{for_each_geometry, Geometry, LineString, Polygon};
use crate::geoarrow::storage::{downcast, CoordsView};
use crate::geoarrow::{describe, Kind};

/// Twice the signed area of one closed ring, positive when it runs CCW.
/// Like `_twice_signed` in Python: relative to its first coordinate,
/// and without the repeated last one.
pub(crate) fn twice_signed(ring: &LineString) -> f64 {
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

/// The CRS of a column to measure an area in. Type operation only.
fn crs_of(field: &Field) -> PolarsResult<String> {
    let geo = describe(field.dtype())?;
    polars_ensure!(
        geo.metadata.declares_crs(),
        SchemaMismatch: "the geometry declares no CRS, so there is no ellipsoid to measure on"
    );
    geo.metadata.crs().map(str::to_owned)
}

/// Only polygons and multipolygons enclose anything: other kinds are refused
/// rather than given an area of `0.0`.
fn encloses_nothing(kind: Kind) -> PolarsError {
    polars_err!(SchemaMismatch: "area does not accept a `{}` column", kind.name())
}

/// An `f64` per geometry: for a multipolygon, the sum of its parts.
fn area_dtype(field: &Field) -> PolarsResult<Field> {
    match describe(field.dtype())?.kind {
        Kind::Polygon | Kind::MultiPolygon => {
            Ok(Field::new(field.name().clone(), DataType::Float64))
        }
        kind @ (Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString) => {
            Err(encloses_nothing(kind))
        }
    }
}

/// For [`area_planar`]: in the unit of the coordinates squared.
fn kernel_output_type_area_planar(input_fields: &[Field]) -> PolarsResult<Field> {
    area_dtype(&input_fields[0])
}

/// For [`area_geodesic`]: in the unit of the CRS squared.
fn kernel_output_type_area_geodesic(input_fields: &[Field]) -> PolarsResult<Field> {
    // Check if the CRS can be used by PROJ. If not, raise error at plan time rather than at runtime.
    GeodesicMetric::of(&crs_of(&input_fields[0])?)?;
    area_dtype(&input_fields[0])
}

/// The planar area of every polygon, or of every multipolygon as the sum of its parts.
fn planar(storage: &Series, kind: Kind) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.chunks() {
        for_each_geometry(chunk.as_ref(), kind, |geometry| {
            out.push(match geometry {
                Some(Geometry::Polygon(polygon)) => Some(planar_area(&polygon)),
                Some(Geometry::MultiPolygon(multipolygon)) => Some(
                    multipolygon
                        .polygons()
                        .map(|polygon| planar_area(&polygon))
                        .sum(),
                ),
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

/// See `area_planar`.
#[polars_expr(output_type_func=kernel_output_type_area_planar)]
fn area_planar(inputs: &[Series]) -> PolarsResult<Series> {
    let storage = inputs[0].ext()?.storage();
    let out = match describe(inputs[0].dtype())?.kind {
        kind @ (Kind::Polygon | Kind::MultiPolygon) => planar(storage, kind)?,
        kind @ (Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString) => {
            return Err(encloses_nothing(kind))
        }
    };
    Ok(out.into_series().with_name(inputs[0].name().clone()))
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

/// The area of every multipolygon: the sum of its parts.
/// A missing part renders the entire multipolygon as missing.
fn multipolygons(storage: &Series, metric: &GeodesicMetric) -> PolarsResult<Float64Chunked> {
    let mut out = Vec::with_capacity(storage.len());
    for chunk in storage.list()?.downcast_iter() {
        let polygons = Polygons::new(downcast(chunk.values().as_ref(), "a list of polygons")?)?;
        for i in 0..chunk.len() {
            if chunk.is_null(i) {
                out.push(None);
                continue;
            }
            let (start, end) = chunk.offsets().start_end(i);
            let parts: Option<Vec<f64>> = (start..end)
                .map(|polygon| polygons.area(polygon, metric))
                .collect::<PolarsResult<_>>()?;
            out.push(parts.map(|parts| parts.iter().sum()));
        }
    }
    Ok(Float64Chunked::from_iter_options(
        storage.name().clone(),
        out.into_iter(),
    ))
}

/// See `area`.
#[polars_expr(output_type_func=kernel_output_type_area_geodesic)]
fn area_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let crs = crs_of(&inputs[0].field())?;
    let storage = inputs[0].ext()?.storage();

    let out = match describe(inputs[0].dtype())?.kind {
        Kind::Polygon => polygons(storage, &*GeodesicMetric::of(&crs)?)?.into_series(),
        Kind::MultiPolygon => multipolygons(storage, &*GeodesicMetric::of(&crs)?)?.into_series(),
        kind @ (Kind::Point | Kind::LineString | Kind::MultiPoint | Kind::MultiLineString) => {
            return Err(encloses_nothing(kind))
        }
    };
    Ok(out.with_name(inputs[0].name().clone()))
}
