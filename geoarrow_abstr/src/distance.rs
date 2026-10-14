//! How far apart points are, as a [Metric] measures them.
//! Functionality for:
//! - distance between arrays of two points
//! - distance between one fixed and an array of others
//! - n*n distances (between each and all other)

use crate::{GeoArrowCoordsView, GeoArrowF64Array, Metric};

/// The distance between the points of `a` and `b`, which are the same length.
pub fn pairwise<const Z: bool, M: Metric, O: GeoArrowF64Array>(
    metric: &M,
    a: &impl GeoArrowCoordsView,
    b: &impl GeoArrowCoordsView,
) -> Result<O, M::Error> {
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
pub fn against_one<const Z: bool, M: Metric, O: GeoArrowF64Array>(
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
