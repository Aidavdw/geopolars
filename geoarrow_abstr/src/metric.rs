//! Different methods for measuring distance.
//! Multiple methods are possible:
//! - Planar
//! - Geodesic

// TODO: A couple more methods of calculating distance:
// - Euclidian (fast like planar, but now uses the CRS units),
//   which should have a usage note that it might have a large error for long distances.
// - Haversine, errors up to 0.5%
// - Rhumb, distance along const lat or long bearing
// - Projected, which is just Geodesic but first projected to WGS84, and uses Karney's algorithm.

use std::convert::Infallible;
use std::fmt;

use geographiclib_rs::{Geodesic, InverseGeodesic, PolygonArea, Winding};
use proj::{Proj, ProjError};

use crate::{GeoArrowValidCoordsView, LineString};

/// How a segment, or the area a ring encloses, is measured: on the plane, or along an ellipsoid.
/// Generic, so every metric and dimension gets a loop of its own.
pub trait Metric {
    /// A vertex, ready to measure from (reprojected once, for the ellipsoid).
    type Vertex: Copy;
    /// Why a vertex could not be prepared (e.g. a failed reprojection).
    type Error;
    /// The vertex at `position`. Its height only counts in [`segment`](Self::segment) with `Z`.
    fn vertex(&self, position: [f64; 3]) -> Result<Self::Vertex, Self::Error>;
    /// How long the segment `ab` is.
    fn segment<const Z: bool>(&self, a: Self::Vertex, b: Self::Vertex) -> f64;
}

/// In the space the coordinates lie in, by Pythagoras.
pub struct Planar;

impl Metric for Planar {
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
            dx.hypot(dy)
        }
    }
}

/// A point's `(x, y)` and its `z`, kept apart so the `(x, y)` can go to PROJ as is.
pub type WithHeight = ((f64, f64), f64);

/// A coordinate PROJ could not bring to longitude/latitude.
#[derive(Debug)]
pub struct ReprojectError {
    pub position: (f64, f64),
    pub source: ProjError,
}

impl fmt::Display for ReprojectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (x, y) = self.position;
        write!(f, "failed to reproject ({x}, {y}): {}", self.source)
    }
}

impl std::error::Error for ReprojectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Measures coordinates in one CRS against each other along the ellipsoid that CRS lies on.
///
/// Kept apart from the kernels, so the distance between two points, the length of a linestring
/// and the area of a polygon (all through [`Metric`]) can be measured with the same thing.
pub struct GeodesicMetric {
    /// To the CRS's own longitude/latitude, so no datum is shifted.
    to_lonlat: Proj,
    pub(crate) ellipsoid: Geodesic,
}

impl GeodesicMetric {
    /// `to_lonlat` goes from the CRS to its own longitude/latitude, so no datum is shifted.
    /// `semi_major` is in the unit of the CRS's `x` and `y` (metres if those are degrees),
    /// which is the unit everything measured comes out in.
    pub fn new(to_lonlat: Proj, semi_major: f64, flattening: f64) -> Self {
        Self {
            to_lonlat,
            ellipsoid: Geodesic::new(semi_major, flattening),
        }
    }

    /// `(lon, lat)` in degrees, on this CRS's own datum.
    ///
    /// Reprojecting is the expensive part of measuring,
    /// so something measuring many segments should do this once per vertex.
    pub fn lonlat(&self, (x, y): (f64, f64)) -> Result<(f64, f64), ReprojectError> {
        self.to_lonlat
            .convert((x, y))
            .map_err(|source| ReprojectError {
                position: (x, y),
                source,
            })
    }

    /// The geodesic distance between two coordinates already in [`lonlat`](Self::lonlat),
    /// in the unit the ellipsoid was given in (see [`new`](Self::new)).
    pub fn between(&self, (lon_a, lat_a): (f64, f64), (lon_b, lat_b): (f64, f64)) -> f64 {
        self.ellipsoid.inverse(lat_a, lon_a, lat_b, lon_b)
    }
}

/// Along the ellipsoid, climbing and descending with `z`:
/// each segment is its geodesic and its difference in height, by Pythagoras.
impl Metric for GeodesicMetric {
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
            along.hypot(z_b - z_a)
        } else {
            along
        }
    }
}
