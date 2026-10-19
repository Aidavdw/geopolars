//! Abstraction traits for accessing GeoArrow coordinates.
//! Extends geo-traits.
//! Also holds the kernels written against those traits, so they work whatever storage backs them:
//! [native] ones of our own, and [geos] ones forwarding to GEOS, each behind its feature.
//!
//! See the expression implementation guide on Array abstractions for GeoArrow.

/// Dispatches runtime `bool` flags into generic implementations,
/// so that the hot path never branches on them.
/// `with_const_flags!(f(args); a, b)` calls `f::<A, B>(args)`,
/// with each runtime flag turned into a const `bool`, in order.
#[macro_export]
macro_rules! with_const_flags {
    // `done` holds the flags already turned into consts.
    (@ $f:ident [$($done:tt)*] ($($arg:expr),*)) => {
        $f::<$($done)*>($($arg),*)
    };
    (@ $f:ident [$($done:tt)*] ($($arg:expr),*) $flag:expr $(, $rest:expr)*) => {
        if $flag {
            $crate::with_const_flags!(@ $f [$($done)* true,] ($($arg),*) $($rest),*)
        } else {
            $crate::with_const_flags!(@ $f [$($done)* false,] ($($arg),*) $($rest),*)
        }
    };
    ($f:ident($($arg:expr),*); $($flag:expr),+) => {
        $crate::with_const_flags!(@ $f [] ($($arg),*) $($flag),+)
    };
}

mod arrays;
mod coords;
mod dimension;
#[cfg(feature = "geos")]
pub mod geos;
mod linestrings;
mod lists;
mod metric;
#[cfg(feature = "native")]
pub mod native;
mod points;
mod polygons;

pub use arrays::{
    GeoArrowBoolArray, GeoArrowBoolListArray, GeoArrowF64Array, GeoArrowLineStringOutput,
    GeoArrowMultiLineStringOutput, GeoArrowMultiPolygonOutput, GeoArrowPolygonOutput, OwnedAxes,
};
pub use coords::{GeoArrowCoordsShared, GeoArrowCoordsView, GeoArrowValidCoordsView};
pub use dimension::GeoDimension;
pub use linestrings::{
    LineStringArrayView, LineStringView, MultiLineStringArrayView, MultiLineStringView,
};
pub use lists::{GeoArrowListShared, GeoArrowListView, GeoArrowValidListView};
pub use metric::{BaseMetric, GeodesicMetric, Planar, ReprojectError, WithHeight};
pub use points::{MultiPointArrayView, PointArrayView, PointView};
pub use polygons::{MultiPolygonArrayView, MultiPolygonView, PolygonArrayView, PolygonView};
