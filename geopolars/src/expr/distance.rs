//! How far apart two points are along the ellipsoid, for points that declare a CRS.
//!
//! Points without a CRS are measured in plain Polars expressions instead, on the Python side.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use geographiclib_rs::{Geodesic, InverseGeodesic, PolygonArea, Winding};
use polars::prelude::*;
use polars_arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use polars_arrow::datatypes::ArrowDataType;
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use rsgeo::proj::Proj;

use crate::geoarrow::geodetic::GeodeticCrs;
use crate::geoarrow::storage::CoordsView;
use crate::geoarrow::{describe, Kind};

/// Measures coordinates in one CRS against each other along the ellipsoid that CRS lies on.
///
/// Kept apart from the expression, so the distance between two points, the length of a linestring
/// (both through [`Metric`]) and the area of a polygon can be measured with the same thing.
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

    /// The geodesic distance between two coordinates already in [`lonlat`](Self::lonlat),
    /// in the unit of the CRS's `x` and `y` (metres if those are degrees),
    /// which is the unit [`GeodeticCrs::semi_major`] gives the ellipsoid in.
    pub fn between(&self, (lon_a, lat_a): (f64, f64), (lon_b, lat_b): (f64, f64)) -> f64 {
        self.ellipsoid.inverse(lat_a, lon_a, lat_b, lon_b)
    }

    /// The area the closed ring through `coords` encloses on the ellipsoid,
    /// in the CRS's unit squared, like [`between`](Self::between).
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
}

/// How a segment is measured: on the plane, or along an ellipsoid.
/// Generic, so every metric and dimension gets a loop of its own.
pub(crate) trait Metric {
    /// A vertex, ready to measure from (reprojected once, for the ellipsoid).
    type Vertex: Copy;
    /// The vertex at `position`. Its height only counts in [`segment`](Self::segment) with `Z`.
    fn vertex(&self, position: [f64; 3]) -> PolarsResult<Self::Vertex>;
    /// How long the segment `ab` is.
    fn segment<const Z: bool>(&self, a: Self::Vertex, b: Self::Vertex) -> f64;
}

/// In the space the coordinates lie in, by Pythagoras.
pub(crate) struct Planar;

impl Metric for Planar {
    type Vertex = [f64; 3];

    #[inline(always)]
    fn vertex(&self, position: [f64; 3]) -> PolarsResult<[f64; 3]> {
        Ok(position)
    }

    #[inline(always)]
    fn segment<const Z: bool>(&self, [ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3]) -> f64 {
        let (dx, dy) = (bx - ax, by - ay);
        if Z {
            let dz = bz - az;
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            dx.hypot(dy)
        }
    }
}

/// Along the ellipsoid, climbing and descending with `z`:
/// each segment is its geodesic and its difference in height, by Pythagoras.
impl Metric for GeodesicMetric {
    type Vertex = WithHeight;

    #[inline(always)]
    fn vertex(&self, [x, y, z]: [f64; 3]) -> PolarsResult<WithHeight> {
        Ok((self.lonlat((x, y))?, z))
    }

    #[inline(always)]
    fn segment<const Z: bool>(&self, (a, z_a): WithHeight, (b, z_b): WithHeight) -> f64 {
        let along = self.between(a, b);
        if Z {
            along.hypot(z_b - z_a)
        } else {
            along
        }
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

/// For [`distance_geodesic`]: an `f64`, named after the first point.
fn kernel_output_type_distance_geodesic(input_fields: &[Field]) -> PolarsResult<Field> {
    // So a CRS PROJ cannot measure in fails while the plan is built.
    GeodesicMetric::of(&shared_crs(input_fields)?)?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

/// A point's `(x, y)` and its `z`, kept apart so the `(x, y)` can go to PROJ as is.
pub type WithHeight = ((f64, f64), f64);

/// The distance between the points of `a` and `b`
fn pairwise<const Z: bool>(
    metric: &impl Metric,
    a: &CoordsView,
    b: &CoordsView,
) -> PolarsResult<PrimitiveArray<f64>> {
    polars_ensure!(a.len() == b.len(), ComputeError: "Chunks must be same size");
    let mut distances = MutablePrimitiveArray::with_capacity(a.len());
    for i in 0..a.len() {
        // A invalid/null point can hold anything in its fields. Never hand that to PROJ.
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        match (a.position::<true>(i), b.position::<true>(i)) {
            (Some(here), Some(there)) => distances
                .push_value(metric.segment::<Z>(metric.vertex(here)?, metric.vertex(there)?)),
            _ => distances.push(None),
        }
    }
    Ok(distances.into())
}

/// The distance between the single point at `one` and every point of `many`.
/// `one` is prepared (reprojected) once, rather than on every row.
fn against_one<const Z: bool>(
    metric: &impl Metric,
    one: [f64; 3],
    many: &CoordsView,
) -> PolarsResult<PrimitiveArray<f64>> {
    let one = metric.vertex(one)?;
    let mut distances = MutablePrimitiveArray::with_capacity(many.len());
    for i in 0..many.len() {
        // A invalid/null point can hold anything in its fields. Never hand that to PROJ.
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        match many.position::<true>(i) {
            Some(there) => distances.push_value(metric.segment::<Z>(one, metric.vertex(there)?)),
            None => distances.push(None),
        }
    }
    Ok(distances.into())
}

/// [`against_one`] for every chunk of `many`. Distance goes both ways,
/// so this serves a single point on either side.
fn every_against_one(
    metric: &impl Metric,
    one: &StructChunked,
    many: &StructChunked,
    z: bool,
) -> PolarsResult<Vec<PrimitiveArray<f64>>> {
    let one = one.rechunk();
    let one = one
        .downcast_iter()
        .next()
        .expect("a single point is one chunk");
    let Some(one) = CoordsView::new(one)?.position::<true>(0) else {
        return Ok(many
            .downcast_iter()
            .map(|many| PrimitiveArray::new_null(ArrowDataType::Float64, many.len()))
            .collect());
    };
    many.downcast_iter()
        .map(|many| with_const_flags!(against_one(metric, one, &CoordsView::new(many)?); z))
        .collect()
}

/// See `distance`.
#[polars_expr(output_type_func=kernel_output_type_distance_geodesic)]
fn distance_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
    let fields = [
        inputs[0].field().into_owned(),
        inputs[1].field().into_owned(),
    ];
    let metric = GeodesicMetric::of(&shared_crs(&fields)?)?;
    let z = describe(fields[0].dtype())?.dim.has_z() && describe(fields[1].dtype())?.dim.has_z();

    let a = inputs[0].ext()?.storage().struct_()?;
    let b = inputs[1].ext()?.storage().struct_()?;
    let chunks = match (a.len(), b.len()) {
        (n, m) if n == m => {
            let (a, b) = align_chunks_binary(a, b);
            a.downcast_iter()
                .zip(b.downcast_iter())
                .map(|(a, b)| {
                    let (a, b) = (CoordsView::new(a)?, CoordsView::new(b)?);
                    with_const_flags!(pairwise(&*metric, &a, &b); z)
                })
                .collect::<PolarsResult<Vec<_>>>()?
        }
        (1, _) => every_against_one(&*metric, a, b, z)?,
        (_, 1) => every_against_one(&*metric, b, a, z)?,
        (n, m) => polars_bail!(ShapeMismatch: "cannot measure {n} points against {m}"),
    };
    Ok(Float64Chunked::from_chunk_iter(inputs[0].name().clone(), chunks).into_series())
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
