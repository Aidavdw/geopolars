//! Kernels on GeoArrow, taking the views of `geopolars-core`:
//! [native] ones of our own, and [geos] ones forwarding to GEOS, each behind its feature.
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

#[cfg(feature = "geos")]
pub mod geos;
mod metric;
#[cfg(feature = "native")]
pub mod native;

pub use metric::{BaseMetric, GeodesicMetric, Planar, ReprojectError, WithHeight};
