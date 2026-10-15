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
pub fn pairwise<const Z: bool, M: DistanceMetric, V: GeoArrowCoordsView, O>(
    metric: &M,
    a: &V,
    b: &V,
) -> Result<O, M::Error>
where
    O: GeoArrowF64Array<Validity = V::Validity>,
{
    debug_assert!(a.len() == b.len(), "Chunks must be the same size");
    let values = if M::MEASURES_NULL_SLOTS {
        pairwise_values::<false, Z, _>(metric, a, b)?
    } else {
        pairwise_values::<true, Z, _>(metric, a, b)?
    };
    Ok(O::from_values_and(values, a.validity(), b.validity()))
}

/// [pairwise]'s distances, measuring missing points only without `NULLS`.
/// A missing point's distance is masked by the validity afterwards.
#[inline(always)]
fn pairwise_values<const NULLS: bool, const Z: bool, M: DistanceMetric>(
    metric: &M,
    a: &impl GeoArrowCoordsView,
    b: &impl GeoArrowCoordsView,
) -> Result<Vec<f64>, M::Error> {
    // Filled by index, not collected: collecting through `Result` loses the length,
    // so the Vec would grow as it goes, checking its capacity on every row.
    let mut values = vec![0.0; a.len()];
    for (i, value) in values.iter_mut().enumerate() {
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        if let (Some(here), Some(there)) = (a.position::<NULLS, Z>(i), b.position::<NULLS, Z>(i)) {
            *value = metric.segment::<Z>(metric.vertex(here)?, metric.vertex(there)?);
        }
        // A missing point keeps its 0.0, which the validity masks out.
    }
    Ok(values)
}

/// The distance between the single point at `one` and every point of `many`.
/// `one` is prepared (reprojected) once, rather than on every row.
pub fn against_one<const Z: bool, M: DistanceMetric, V: GeoArrowCoordsView, O>(
    metric: &M,
    one: [f64; 3],
    many: &V,
) -> Result<O, M::Error>
where
    O: GeoArrowF64Array<Validity = V::Validity>,
{
    // If any of the 'one' is NaN, then everything will become NaN anyway,
    // so we might as well exit early.
    // Missing points stay missing.
    if one[..if Z { 3 } else { 2 }]
        .iter()
        .any(|axis| axis.is_nan())
    {
        return Ok(O::from_values(vec![f64::NAN; many.len()], many.validity()));
    }
    let one = metric.vertex(one)?;
    let values = if M::MEASURES_NULL_SLOTS {
        against_one_values::<false, Z, _>(metric, one, many)?
    } else {
        against_one_values::<true, Z, _>(metric, one, many)?
    };
    Ok(O::from_values(values, many.validity()))
}

/// [against_one]'s distances, measuring missing points only without `NULLS`.
/// A missing point's distance is masked by the validity afterwards.
#[inline(always)]
fn against_one_values<const NULLS: bool, const Z: bool, M: DistanceMetric>(
    metric: &M,
    one: M::Vertex,
    many: &impl GeoArrowCoordsView,
) -> Result<Vec<f64>, M::Error> {
    // Filled by index, as in `pairwise_values`.
    let mut values = vec![0.0; many.len()];
    for (i, value) in values.iter_mut().enumerate() {
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        if let Some(there) = many.position::<NULLS, Z>(i) {
            *value = metric.segment::<Z>(one, metric.vertex(there)?);
        }
        // A missing point keeps its 0.0, which the validity masks out.
    }
    Ok(values)
}
