//! How far apart two points are along the ellipsoid, for points that declare a CRS.
//!
//! Points without a CRS are measured in plain Polars expressions instead, on the Python side.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rsgeo::proj::Proj;
use rsgeo::{Distance, Geodesic, Point};

use crate::geoarrow::crs::ExtensionMetadata;
use crate::geoarrow::{describe, Kind};

// FIXME: distance should always be measured on the given ellipsoid.
/// Longitude/latitude on WGS 84: what [`Geodesic`] measures on.
const LONLAT: &str = "EPSG:4326";

/// Measures coordinates in one CRS against each other along the WGS 84 ellipsoid.
///
/// Kept apart from the expression, so the length of a linestring can be
/// measured segment by segment with the same thing.
///
/// Every CRS is measured on WGS 84, whatever ellipsoid it is defined on.
/// Its datum puts both coordinates off by (nearly) the same amount,
/// so this changes the distance between them far less than it changes where they are.
pub struct GeodesicMetric {
    to_lonlat: Proj,
}

impl GeodesicMetric {
    /// `crs` is anything PROJ accepts, as [`ExtensionMetadata::crs`] gives it.
    pub fn new(crs: &str) -> PolarsResult<Self> {
        let to_lonlat = Proj::new_known_crs(crs, LONLAT, None).map_err(
            |e| polars_err!(ComputeError: "cannot measure along the ellipsoid in {crs}: {e}"),
        )?;
        Ok(Self { to_lonlat })
    }

    fn lonlat(&self, (x, y): (f64, f64)) -> PolarsResult<Point<f64>> {
        let (lon, lat) = self
            .to_lonlat
            .convert((x, y))
            .map_err(|e| polars_err!(ComputeError: "failed to reproject ({x}, {y}): {e}"))?;
        Ok(Point::new(lon, lat))
    }

    /// The square of the geodesic distance between `a` and `b`, in metres².
    pub fn distance_squared(&self, a: (f64, f64), b: (f64, f64)) -> PolarsResult<f64> {
        let distance = Geodesic.distance(self.lonlat(a)?, self.lonlat(b)?);
        Ok(distance * distance)
    }
}

/// The CRS two point columns share. Type operation only.
///
/// Distances between points in different CRSs would have to pick one of them,
/// so they are refused rather than silently reprojected.
fn shared_crs(fields: &[Field]) -> PolarsResult<String> {
    let [a, b] = fields else {
        polars_bail!(ComputeError: "a distance is between two points, got {} inputs", fields.len());
    };
    let mut metadata = Vec::with_capacity(2);
    for field in [a, b] {
        let geo = describe(field.dtype())?;
        polars_ensure!(
            geo.kind == Kind::Point,
            SchemaMismatch: "a distance is measured between two `{}` columns, got: {}",
            Kind::Point.name(), field.dtype()
        );
        metadata.push(ExtensionMetadata::parse(
            geo.typ.serialize_metadata().as_deref(),
        )?);
    }

    match (metadata[0].declares_crs(), metadata[1].declares_crs()) {
        (true, true) => {}
        (true, false) | (false, true) => polars_bail!(
            SchemaMismatch: "only one of the points declares a CRS, so they cannot be measured \
            against each other; declare the same one on both"
        ),
        (false, false) => polars_bail!(
            SchemaMismatch: "neither point declares a CRS, so there is no ellipsoid to measure on"
        ),
    }
    let (crs_a, crs_b) = (metadata[0].crs()?, metadata[1].crs()?);
    polars_ensure!(
        crs_a == crs_b,
        SchemaMismatch: "the points declare different CRSs, {crs_a} and {crs_b}; \
        use `to_crs` to bring one onto the other first"
    );
    Ok(crs_a)
}

/// `output_type_func` for [`distance_squared_geodesic`]: an `f64`, named after the first point.
fn squared_metres(input_fields: &[Field]) -> PolarsResult<Field> {
    shared_crs(input_fields)?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

/// The `x` and `y` of every point, `None` where the point is missing.
fn points(storage: &Series) -> PolarsResult<Vec<Option<(f64, f64)>>> {
    let fields = storage.struct_()?;
    let x = fields.field_by_name("x")?;
    let y = fields.field_by_name("y")?;
    // A missing point can hold anything in its fields. Never hand that to PROJ.
    let present = storage.is_not_null();
    Ok(present
        .iter()
        .zip(x.f64()?.iter())
        .zip(y.f64()?.iter())
        .map(|((present, x), y)| match (present, x, y) {
            (Some(true), Some(x), Some(y)) => Some((x, y)),
            _ => None,
        })
        .collect())
}

/// The square of the geodesic distance between two points, row by row, in metres².
/// Either side can be a single point, which is then measured against every row of the other.
#[polars_expr(output_type_func=squared_metres)]
fn distance_squared_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let fields = [
        inputs[0].field().into_owned(),
        inputs[1].field().into_owned(),
    ];
    let metric = GeodesicMetric::new(&shared_crs(&fields)?)?;

    let a = points(inputs[0].ext()?.storage())?;
    let b = points(inputs[1].ext()?.storage())?;
    let len = match (a.len(), b.len()) {
        (n, m) if n == m => n,
        (1, m) => m,
        (n, 1) => n,
        (n, m) => polars_bail!(ShapeMismatch: "cannot measure {n} points against {m}"),
    };
    let at = |side: &[Option<(f64, f64)>], i: usize| side[if side.len() == 1 { 0 } else { i }];

    let out = (0..len)
        .map(|i| match (at(&a, i), at(&b, i)) {
            (Some(a), Some(b)) => metric.distance_squared(a, b).map(Some),
            _ => Ok(None),
        })
        .collect::<PolarsResult<Float64Chunked>>()?;
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}
