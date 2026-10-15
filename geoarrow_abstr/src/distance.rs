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
    /// The vertex at `position`. Its height only counts in [`segment`](Self::segment) with `Z`.
    fn vertex(&self, position: [f64; 3]) -> Result<Self::Vertex, Self::Error>;
    /// How long the segment `ab` is.
    fn segment<const Z: bool>(&self, a: Self::Vertex, b: Self::Vertex) -> f64;
}

impl DistanceMetric for Planar {
    type Vertex = [f64; 3];
    type Error = Infallible;

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
pub fn pairwise<const Z: bool, M: DistanceMetric, O: GeoArrowF64Array>(
    metric: &M,
    a: &impl GeoArrowCoordsView,
    b: &impl GeoArrowCoordsView,
) -> Result<O, M::Error> {
    debug_assert!(a.len() == b.len(), "Chunks must be the same size");
    O::try_from_options::<true, _>((0..a.len()).map(|i| {
        // A invalid/null point can hold anything in its fields. Never hand that to PROJ.
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        match (a.position::<true, Z>(i), b.position::<true, Z>(i)) {
            (Some(here), Some(there)) => Ok(Some(
                metric.segment::<Z>(metric.vertex(here)?, metric.vertex(there)?),
            )),
            _ => Ok(None),
        }
    }))
}

/// The distance between the single point at `one` and every point of `many`.
/// `one` is prepared (reprojected) once, rather than on every row.
pub fn against_one<const Z: bool, M: DistanceMetric, O: GeoArrowF64Array>(
    metric: &M,
    one: [f64; 3],
    many: &impl GeoArrowCoordsView,
) -> Result<O, M::Error> {
    // If any of the 'one' is NaN, then everything will become NaN anyway,
    // so we might as well exit early.
    // Missing points stay missing.
    if one[..if Z { 3 } else { 2 }]
        .iter()
        .any(|axis| axis.is_nan())
    {
        return O::try_from_options::<true, _>(
            (0..many.len()).map(|i| Ok(many.position::<true, Z>(i).map(|_| f64::NAN))),
        );
    }
    let one = metric.vertex(one)?;
    O::try_from_options::<true, _>((0..many.len()).map(|i| {
        // A invalid/null point can hold anything in its fields. Never hand that to PROJ.
        // An empty point (NaN coordinates) does go through, and comes out as a NaN distance:
        // empty points are assumed rare enough not to warrant an extra check on every valid point.
        match many.position::<true, Z>(i) {
            Some(there) => Ok(Some(metric.segment::<Z>(one, metric.vertex(there)?))),
            None => Ok(None),
        }
    }))
}
