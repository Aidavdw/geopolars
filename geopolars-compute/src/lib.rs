//! Kernels on GeoArrow, taking the views of `geopolars-core`:
//! [native] ones of our own, and [geos], [rsgeo] and [proj] ones forwarding to those libraries.
//! [validity_checking] combines native and rsgeo checks.
//!
//! See the README.

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

mod error;
#[cfg(feature = "geos")]
pub mod geos;
mod metric;
#[cfg(feature = "native")]
pub mod native;
pub mod proj;
#[cfg(feature = "rsgeo")]
pub mod rsgeo;
#[cfg(feature = "rsgeo")]
pub mod validity_checking;

use error::row_or_missing;
pub use error::{GeoComputeError, GeoComputeResult};
pub use metric::{BaseMetric, GeodesicMetric, Planar, WithHeight};
