//! How far apart two points are along the ellipsoid, for points that declare a CRS.
//!
//! Points without a CRS are measured in plain Polars expressions instead, on the Python side.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use geographiclib_rs::{Geodesic, InverseGeodesic, PolygonArea, Winding};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rsgeo::proj::Proj;

use crate::geoarrow::geodetic::GeodeticCrs;
use crate::geoarrow::{describe, Kind};

/// Measures coordinates in one CRS against each other along the ellipsoid that CRS lies on.
///
/// Kept apart from the expression, so the length of a linestring
/// and the area of a polygon can be measured with the same thing.
pub struct GeodesicMetric {
    /// To the CRS's own longitude/latitude, so no datum is shifted.
    to_lonlat: Proj,
    ellipsoid: Geodesic,
}

/// How many CRSes to keep in cache.
/// If more are used, it will start over.
const CACHED_CRSS: usize = 16;

thread_local! {
    /// A cache of CRSes.
    /// call to PROJ by parsing costs more than measuring a small morsel, so cached.
    /// Per thread, because a [`Proj`] can't be shared between threads.
    /// With this, CRS can be re-used over multiple morsels.
    static METRICS: RefCell<HashMap<String, Rc<GeodesicMetric>>> = RefCell::new(HashMap::new());
}

impl GeodesicMetric {
    /// Populates or gets the metric for `crs` from the cache.
    ///
    /// `crs` is anything PROJ accepts, as [`ExtensionMetadata::crs`](crate::geoarrow::crs::ExtensionMetadata::crs) gives it.
    pub fn of(crs: &str) -> PolarsResult<Rc<Self>> {
        // A CRS that fails to build is not cached, and fails again the next time.
        if let Some(metric) = METRICS.with_borrow(|metrics| metrics.get(crs).cloned()) {
            return Ok(metric);
        }
        // Built outside the borrow, so nothing is held while PROJ runs.
        let metric = Rc::new(Self::new(crs)?);
        METRICS.with_borrow_mut(|metrics| {
            if metrics.len() >= CACHED_CRSS {
                metrics.clear();
            }
            metrics.insert(crs.to_owned(), Rc::clone(&metric));
        });
        Ok(metric)
    }

    fn new(crs: &str) -> PolarsResult<Self> {
        let geodetic = GeodeticCrs::of(crs)?;
        let to_lonlat = Proj::new_known_crs(crs, &geodetic.definition, None).map_err(
            |e| polars_err!(ComputeError: "cannot measure along the ellipsoid in {crs}: {e}"),
        )?;
        let ellipsoid = Geodesic::new(geodetic.semi_major, geodetic.flattening);
        Ok(Self {
            to_lonlat,
            ellipsoid,
        })
    }

    /// `(lon, lat)` in degrees, on this CRS's own datum.
    ///
    /// Reprojecting is the expensive part of measuring,
    /// so something measuring many segments should do this once per vertex.
    pub fn lonlat(&self, (x, y): (f64, f64)) -> PolarsResult<(f64, f64)> {
        self.to_lonlat
            .convert((x, y))
            .map_err(|e| polars_err!(ComputeError: "failed to reproject ({x}, {y}): {e}"))
    }

    /// The geodesic distance between two coordinates already in [`lonlat`](Self::lonlat), in metres.
    pub fn between(&self, (lon_a, lat_a): (f64, f64), (lon_b, lat_b): (f64, f64)) -> f64 {
        // CHECK: is this squared? if so, then change name.
        self.ellipsoid.inverse(lat_a, lon_a, lat_b, lon_b)
    }

    /// The square of the geodesic distance between `a` and `b`, in metres².
    pub fn distance_squared(&self, a: (f64, f64), b: (f64, f64)) -> PolarsResult<f64> {
        let distance = self.between(self.lonlat(a)?, self.lonlat(b)?);
        Ok(distance * distance)
    }

    /// The length of the path through `coords`.
    pub fn length(
        &self,
        coords: impl IntoIterator<Item = Option<(f64, f64)>>,
    ) -> PolarsResult<Option<f64>> {
        self.path(coords, |xy| self.lonlat(xy), |a, b| self.between(a, b))
    }

    /// The length of the path through `coords`, climbing and descending with their `z`.
    ///
    /// Each segment is its geodesic and its difference in height, by Pythagoras,
    /// like [`distance_squared`](Self::distance_squared) between two points with a `z`.
    pub fn length_with_height(
        &self,
        coords: impl IntoIterator<Item = Option<WithHeight>>,
    ) -> PolarsResult<Option<f64>> {
        self.path(
            coords,
            |(xy, z)| Ok((self.lonlat(xy)?, z)),
            |(a, z_a), (b, z_b)| self.between(a, b).hypot(z_b - z_a),
        )
    }

    /// The area the closed ring through `coords` encloses on the ellipsoid, in metres².
    /// Based on `geographiclib_rs`
    pub fn ring_area(
        &self,
        coords: impl IntoIterator<Item = Option<(f64, f64)>>,
    ) -> PolarsResult<Option<f64>> {
        let mut ring = PolygonArea::new(&self.ellipsoid, Winding::CounterClockwise);
        for coord in coords {
            let Some(coord) = coord else { return Ok(None) };
            let (lon, lat) = self.lonlat(coord)?;
            // The repeated closing vertex adds an edge of length zero, which encloses nothing.
            ring.add_point(lat, lon);
        }
        // Signed, then made positive: unsigned would take a clockwise ring
        // to enclose everything outside of it.
        let (_perimeter, area, _vertices) = ring.compute(true);
        Ok(Some(area.abs()))
    }

    /// Adds up `segment` between every two consecutive `coords`, once they are `prepared`.
    ///
    /// Generic so the 2D and 3D paths each get a loop of their own,
    /// and 2D never looks at a height.
    fn path<C, P: Copy>(
        &self,
        coords: impl IntoIterator<Item = Option<C>>,
        prepare: impl Fn(C) -> PolarsResult<P>,
        segment: impl Fn(P, P) -> f64,
    ) -> PolarsResult<Option<f64>> {
        // Each coordinate is reprojected once, and re-used for the next call.
        // Each segment is measured as a distance rather than squared and rooted again.
        let mut total = 0.0;
        let mut previous = None;
        for coord in coords {
            let Some(coord) = coord else { return Ok(None) };
            let here = prepare(coord)?;
            if let Some(previous) = previous {
                total += segment(previous, here);
            }
            previous = Some(here);
        }
        Ok(Some(total))
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
        metadata.push(geo.metadata);
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

/// A point's `(x, y)` and its `z`, kept apart so the `(x, y)` can go to PROJ as is.
pub type WithHeight = ((f64, f64), f64);

/// The `x` and `y` of every point with its `z`, `None` where either is missing.
fn points_with_height(storage: &Series) -> PolarsResult<Vec<Option<WithHeight>>> {
    let z = storage.struct_()?.field_by_name("z")?;
    Ok(points(storage)?
        .into_iter()
        .zip(z.f64()?.iter())
        .map(|(xy, z)| Some((xy?, z?)))
        .collect())
}

/// `measure` applied row by row to two sides of the same length,
/// or one side of a single point against every row of the other.
fn pairwise<P: Copy>(
    a: &[Option<P>],
    b: &[Option<P>],
    measure: impl Fn(P, P) -> PolarsResult<f64>,
) -> PolarsResult<Float64Chunked> {
    let len = match (a.len(), b.len()) {
        (n, m) if n == m => n,
        (1, m) => m,
        (n, 1) => n,
        (n, m) => polars_bail!(ShapeMismatch: "cannot measure {n} points against {m}"),
    };
    let at = |side: &[Option<P>], i: usize| side[if side.len() == 1 { 0 } else { i }];

    (0..len)
        .map(|i| match (at(a, i), at(b, i)) {
            (Some(a), Some(b)) => measure(a, b).map(Some),
            _ => Ok(None),
        })
        .collect()
}

/// See `distance_squared`.
#[polars_expr(output_type_func=squared_metres)]
fn distance_squared_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let fields = [
        inputs[0].field().into_owned(),
        inputs[1].field().into_owned(),
    ];
    let metric = GeodesicMetric::of(&shared_crs(&fields)?)?;
    let both_have_z =
        describe(fields[0].dtype())?.dim.has_z() && describe(fields[1].dtype())?.dim.has_z();

    let (a, b) = (inputs[0].ext()?.storage(), inputs[1].ext()?.storage());
    // Chosen once per call, so the far more common 2D points
    // go through a loop that never looks at a height.
    let out = if both_have_z {
        pairwise(
            &points_with_height(a)?,
            &points_with_height(b)?,
            |(a, z_a), (b, z_b)| Ok(metric.distance_squared(a, b)? + (z_b - z_a).powi(2)),
        )?
    } else {
        pairwise(&points(a)?, &points(b)?, |a, b| {
            metric.distance_squared(a, b)
        })?
    };
    Ok(out.with_name(inputs[0].name().clone()).into_series())
}

#[cfg(test)]
mod tests {
    //! What the Python suite cannot see: whether a metric was built again.

    use super::*;

    #[test]
    fn a_thread_builds_the_metric_for_a_crs_once() {
        let first = GeodesicMetric::of("EPSG:4326").unwrap();

        assert!(Rc::ptr_eq(
            &first,
            &GeodesicMetric::of("EPSG:4326").unwrap()
        ));
        assert!(!Rc::ptr_eq(
            &first,
            &GeodesicMetric::of("EPSG:4289").unwrap()
        ));
    }

    #[test]
    fn a_crs_that_fails_is_not_cached() {
        assert!(GeodesicMetric::of("EPSG:4978").is_err());
        assert!(GeodesicMetric::of("EPSG:4978").is_err());
    }

    #[test]
    fn the_cache_starts_over_rather_than_growing() {
        for zone in 1..=CACHED_CRSS + 1 {
            GeodesicMetric::of(&format!("EPSG:{}", 32600 + zone)).unwrap();
        }

        assert!(METRICS.with_borrow(|metrics| metrics.len()) <= CACHED_CRSS);
    }
}
