//! How far apart two points are along the ellipsoid, for points that declare a CRS.
//!
//! Points without a CRS are measured in plain Polars expressions instead, on the Python side.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use geographiclib_rs::{Geodesic, InverseGeodesic, PolygonArea, Winding};
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray, StructArray};
use polars_arrow::compute::utils::combine_validities_and;
use polars_arrow::datatypes::ArrowDataType;
use polars_core::utils::align_chunks_binary;
use pyo3_polars::derive::polars_expr;
use rsgeo::proj::Proj;

use crate::geoarrow::geodetic::GeodeticCrs;
use crate::geoarrow::storage::Axes;
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
        // CHECK: is this squared? if so, then change name.
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
    /// Vertex `i`. Its height only counts with `Z`.
    fn vertex<const Z: bool>(&self, axes: &Axes, i: usize) -> PolarsResult<Self::Vertex>;
    /// How long the segment `ab` is.
    fn segment<const Z: bool>(&self, a: Self::Vertex, b: Self::Vertex) -> f64;
}

/// In the space the coordinates lie in, by Pythagoras.
pub(crate) struct Planar;

impl Metric for Planar {
    type Vertex = [f64; 3];

    #[inline(always)]
    fn vertex<const Z: bool>(&self, axes: &Axes, i: usize) -> PolarsResult<[f64; 3]> {
        Ok([axes.x[i], axes.y[i], if Z { axes.z[i] } else { 0.0 }])
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
    fn vertex<const Z: bool>(&self, axes: &Axes, i: usize) -> PolarsResult<WithHeight> {
        let lonlat = self.lonlat((axes.x[i], axes.y[i]))?;
        Ok((lonlat, if Z { axes.z[i] } else { 0.0 }))
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

/// `output_type_func` for [`distance_squared_geodesic`]: an `f64`, named after the first point.
fn squared_distance(input_fields: &[Field]) -> PolarsResult<Field> {
    // So a CRS PROJ cannot measure in fails while the plan is built.
    GeodesicMetric::of(&shared_crs(input_fields)?)?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Float64,
    ))
}

/// A point's `(x, y)` and its `z`, kept apart so the `(x, y)` can go to PROJ as is.
pub type WithHeight = ((f64, f64), f64);

/// The squared distance between the points of `a` and `b`, row by row,
/// over two chunks of the same length.
fn pairwise<const Z: bool>(
    metric: &impl Metric,
    a: &StructArray,
    b: &StructArray,
) -> PolarsResult<PrimitiveArray<f64>> {
    let ((axes_a, _), (axes_b, _)) = (Axes::new(a)?, Axes::new(b)?);
    let validity = combine_validities_and(a.validity(), b.validity());
    let values = (0..a.len())
        .map(|i| {
            // A missing point can hold anything in its fields. Never hand that to PROJ.
            if validity.as_ref().is_some_and(|valid| !valid.get_bit(i)) {
                return Ok(0.0);
            }
            let here = metric.vertex::<Z>(&axes_a, i)?;
            let there = metric.vertex::<Z>(&axes_b, i)?;
            Ok(metric.segment::<Z>(here, there).powi(2))
        })
        .collect::<PolarsResult<Vec<f64>>>()?;
    Ok(PrimitiveArray::from_vec(values).with_validity(validity))
}

/// The squared distance between the single point `one` and every point of `many`.
/// `one` is prepared (reprojected) once, rather than on every row.
fn against_one<const Z: bool>(
    metric: &impl Metric,
    one: &StructArray,
    many: &StructArray,
) -> PolarsResult<PrimitiveArray<f64>> {
    if one.is_null(0) {
        return Ok(PrimitiveArray::new_null(ArrowDataType::Float64, many.len()));
    }
    let one = metric.vertex::<Z>(&Axes::new(one)?.0, 0)?;
    let (axes, _) = Axes::new(many)?;
    let values = (0..many.len())
        .map(|i| {
            // A missing point can hold anything in its fields. Never hand that to PROJ.
            if many.is_null(i) {
                return Ok(0.0);
            }
            Ok(metric
                .segment::<Z>(one, metric.vertex::<Z>(&axes, i)?)
                .powi(2))
        })
        .collect::<PolarsResult<Vec<f64>>>()?;
    Ok(PrimitiveArray::from_vec(values).with_validity(many.validity().cloned()))
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
    many.downcast_iter()
        .map(|many| with_const_flags!(against_one(metric, one, many); z))
        .collect()
}

/// See `distance_squared`.
#[polars_expr(output_type_func=squared_distance)]
fn distance_squared_geodesic(inputs: &[Series]) -> PolarsResult<Series> {
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
                .map(|(a, b)| with_const_flags!(pairwise(&*metric, a, b); z))
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
