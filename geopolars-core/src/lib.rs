//! Views over the Polars storage of GeoArrow geometries,
//! and the arrays kernels build their output in.
//!
//! See the README.

mod coords;
mod dimension;
mod linestrings;
mod lists;
pub mod output;
mod points;
mod polygons;

use polars_arrow::array::Array;
use polars_core::prelude::{polars_err, PolarsResult};

pub use coords::{
    is_close, same_position, AxisArrays, Coords, ValidCoords, ABS_TOL, MIN_RING_VERTICES, REL_TOL,
};
pub use dimension::GeoDimension;
pub use linestrings::{LineString, LineStringArray, MultiLineString, MultiLineStringArray};
pub use lists::{List, ValidList};
pub use points::{MultiPointArray, Point, PointArray};
pub use polygons::{MultiPolygon, MultiPolygonArray, Polygon, PolygonArray};

/// Downcast one level of the storage to the array it has to be.
/// The dtype was already checked, so this failing is a bug.
pub fn downcast<'a, T: 'static>(array: &'a dyn Array, what: &str) -> PolarsResult<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(
        || polars_err!(ComputeError: "expected {what} in geometry storage, got: {:?}", array.dtype()),
    )
}
