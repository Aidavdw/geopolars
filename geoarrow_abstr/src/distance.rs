//! How far apart points are, as a [DistanceMetric] measures them.
//! Functionality for:
//! - distance between arrays of two points
//! - distance between one fixed and an array of others
//! - n*n distances (between each and all other)

use std::convert::Infallible;

use crate::{
    BaseMetric, GeoArrowCoordsView, GeoArrowF64Array, GeodesicMetric, Planar, ReprojectError,
    WithHeight,
};

/// How a segment is measured: on the plane, or along an ellipsoid.
/// Generic, so every metric and dimension gets a loop of its own.
pub trait DistanceMetric: BaseMetric {
    /// A vertex, ready to measure from (reprojected once, for the ellipsoid).
    type Vertex: Copy;
    /// Why a vertex could not be prepared (e.g. a failed reprojection).
    type Error;
    /// Sometimes it is faster to just *do* a calculation and then
    /// mask out the result *afterwards* (`TRUE`),
    /// rather than first checking for validity,
    /// and if the value is marked as invalid on the bitflag,
    /// skipping the calculation all together (`FALSE`).
    /// This flag switches between the two behaviours.
    ///
    /// If `FALSE`,
    /// That slot can hold anything in its fields,
    /// so only a metric that measures anything harmlessly should!
    /// One that must never hand it on (e.g. to PROJ) skips missing points.
    const MEASURES_NULL_SLOTS: bool;
    /// The vertex at `position`. Its height only counts in [`segment`](Self::segment) with `Z`.
    fn vertex(&self, position: [f64; 3]) -> Result<Self::Vertex, Self::Error>;
    /// How long the segment `ab` is.
    fn segment<const Z: bool>(&self, a: Self::Vertex, b: Self::Vertex) -> f64;
}

impl DistanceMetric for Planar {
    type Vertex = [f64; 3];
    type Error = Infallible;
    const MEASURES_NULL_SLOTS: bool = true;

    #[inline(always)]
    fn vertex(&self, position: [f64; 3]) -> Result<[f64; 3], Infallible> {
        Ok(position)
    }

    #[inline(always)]
    fn segment<const Z: bool>(&self, [ax, ay, az]: [f64; 3], [bx, by, bz]: [f64; 3]) -> f64 {
        let (dx, dy) = (bx - ax, by - ay);
        if Z {
            let dz = bz - az;
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            (dx * dx + dy * dy).sqrt()
        }
    }
}

/// Along the ellipsoid, climbing and descending with `z`:
/// each segment is its geodesic and its difference in height, by Pythagoras.
impl DistanceMetric for GeodesicMetric {
    type Vertex = WithHeight;
    type Error = ReprojectError;
    const MEASURES_NULL_SLOTS: bool = false;

    #[inline(always)]
    fn vertex(&self, [x, y, z]: [f64; 3]) -> Result<WithHeight, ReprojectError> {
        Ok((self.lonlat((x, y))?, z))
    }

    #[inline(always)]
    fn segment<const Z: bool>(&self, (a, z_a): WithHeight, (b, z_b): WithHeight) -> f64 {
        let along = self.between(a, b);
        if Z {
            let dz = z_b - z_a;
            (along * along + dz * dz).sqrt()
        } else {
            along
        }
    }
}

/// The distance between the points of `a` and `b`, which are the same length.
/// On an error, with `STOP_ON_ERROR` it stops and returns it;
/// without, it marks that row missing and carries on, so it never fails.
pub fn pairwise<const Z: bool, const STOP_ON_ERROR: bool, M: DistanceMetric, V, O>(
    metric: &M,
    a: &V,
    b: &V,
) -> Result<O, M::Error>
where
    V: GeoArrowCoordsView,
    O: GeoArrowF64Array<Validity = V::Validity>,
{
    debug_assert!(a.len() == b.len(), "Chunks must be the same size");
    // Can't just pass the M::MEASURES_NULL_SLOTS into the function on rust stable,
    // so use a silly if-statement. I know we use nightly at polars,
    // but this is nicer if other people would want to use this crate :)
    let (values, failed) = if M::MEASURES_NULL_SLOTS {
        pairwise_values::<false, Z, STOP_ON_ERROR, _>(metric, a, b)?
    } else {
        pairwise_values::<true, Z, STOP_ON_ERROR, _>(metric, a, b)?
    };
    Ok(O::from_values_and(
        values,
        a.validity(),
        b.validity(),
        &failed,
    ))
}

/// [pairwise]'s distances, and the rows that failed (only without `STOP_ON_ERROR`),
/// measuring missing points only without `NULLS`.
/// A missing point's distance is masked by the validity afterwards.
#[inline(always)]
fn pairwise_values<const NULLS: bool, const Z: bool, const STOP_ON_ERROR: bool, M>(
    metric: &M,
    a: &impl GeoArrowCoordsView,
    b: &impl GeoArrowCoordsView,
) -> Result<(Vec<f64>, Vec<usize>), M::Error>
where
    M: DistanceMetric,
{
    // Filled by index, not collected,
    // because collecting through `Result` loses the length,
    // so the Vec would grow as it goes, checking its capacity on every row.
    let mut values = vec![0.0; a.len()];
    let mut failed = Vec::with_capacity(a.len());
    for (i, value) in values.iter_mut().enumerate() {
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        if let (Some(here), Some(there)) = (a.position::<NULLS, Z>(i), b.position::<NULLS, Z>(i)) {
            let distance = metric
                .vertex(here)
                .and_then(|here| Ok(metric.segment::<Z>(here, metric.vertex(there)?)));
            match distance {
                Ok(distance) => *value = distance,
                Err(e) if STOP_ON_ERROR => return Err(e),
                Err(_) => failed.push(i),
            }
        }
        // A missing point keeps its 0.0, which the validity masks out.
    }
    Ok((values, failed))
}

/// The distance between the single point at `one` and every point of `many`.
/// `one` is prepared (reprojected) once, rather than on every row.
/// On an error, with `STOP_ON_ERROR` it stops and returns it;
/// without, it marks that row missing (every row, for `one`) and carries on, so it never fails.
pub fn against_one<
    const Z: bool,
    const STOP_ON_ERROR: bool,
    M: DistanceMetric,
    V: GeoArrowCoordsView,
    O: GeoArrowF64Array<Validity = V::Validity>,
>(
    metric: &M,
    one: [f64; 3],
    many: &V,
) -> Result<O, M::Error> {
    // If any of the 'one' is NaN, then everything will become NaN anyway,
    // so we might as well exit early.
    // Missing points stay missing.
    if one[..if Z { 3 } else { 2 }]
        .iter()
        .any(|axis| axis.is_nan())
    {
        return Ok(O::from_values(vec![f64::NAN; many.len()], many.validity()));
    }
    let one = match metric.vertex(one) {
        Ok(one) => one,
        Err(e) if STOP_ON_ERROR => return Err(e),
        Err(_) => {
            let every_row = (0..many.len()).collect::<Vec<_>>();
            return Ok(O::from_values_and(
                vec![0.0; many.len()],
                None,
                None,
                &every_row,
            ));
        }
    };
    let (values, failed) = if M::MEASURES_NULL_SLOTS {
        against_one_values::<false, Z, STOP_ON_ERROR, _>(metric, one, many)?
    } else {
        against_one_values::<true, Z, STOP_ON_ERROR, _>(metric, one, many)?
    };
    Ok(O::from_values_and(values, many.validity(), None, &failed))
}

/// [against_one]'s distances, and the rows that failed (only without `STOP_ON_ERROR`),
/// measuring missing points only without `NULLS`.
/// A missing point's distance is masked by the validity afterwards.
#[inline(always)]
fn against_one_values<const NULLS: bool, const Z: bool, const STOP_ON_ERROR: bool, M>(
    metric: &M,
    one: M::Vertex,
    many: &impl GeoArrowCoordsView,
) -> Result<(Vec<f64>, Vec<usize>), M::Error>
where
    M: DistanceMetric,
{
    // Filled by index, as in `pairwise_values`.
    let mut values = vec![0.0; many.len()];
    let mut failed = Vec::new();
    for (i, value) in values.iter_mut().enumerate() {
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        if let Some(there) = many.position::<NULLS, Z>(i) {
            match metric.vertex(there) {
                Ok(there) => *value = metric.segment::<Z>(one, there),
                Err(e) if STOP_ON_ERROR => return Err(e),
                Err(_) => failed.push(i),
            }
        }
        // A missing point keeps its 0.0, which the validity masks out.
    }
    Ok((values, failed))
}
