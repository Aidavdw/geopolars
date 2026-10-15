//! Expressions over GeoArrow geometry columns.
//!
//! Polars finds the expression functions by doing a `dlsym` for `_polars_plugin_*`
//! to get symbols out of this compiled library, so how they are split across
//! modules here is for readers only -- nothing has to be registered.
//!
//! Nothing in here names a concrete geometry. An expression asks
//! [`geoarrow::describe`](crate::geoarrow::describe) what it was handed and then
//! either does not care (see [`coords::map_coords`]) or matches on the
//! [`Kind`](crate::geoarrow::Kind) exhaustively.

/// Dispatches runtime `bool` flags into generic implementations,
/// so that the hot path never branches on them.
/// `with_const_flags!(f(args); a, b)` calls `f::<A, B>(args)`,
/// with each runtime flag turned into a const `bool`, in order.
macro_rules! with_const_flags {
    // `done` holds the flags already turned into consts.
    (@ $f:ident [$($done:tt)*] ($($arg:expr),*)) => {
        $f::<$($done)*>($($arg),*)
    };
    (@ $f:ident [$($done:tt)*] ($($arg:expr),*) $flag:expr $(, $rest:expr)*) => {
        if $flag {
            with_const_flags!(@ $f [$($done)* true,] ($($arg),*) $($rest),*)
        } else {
            with_const_flags!(@ $f [$($done)* false,] ($($arg),*) $($rest),*)
        }
    };
    ($f:ident($($arg:expr),*); $($flag:expr),+) => {
        with_const_flags!(@ $f [] ($($arg),*) $($flag),+)
    };
}

pub mod affine;
pub mod area;
pub mod centroid;
pub mod construct;
pub mod convex_hull;
pub mod coords;
pub mod crs;
pub mod distance;
pub mod is_ring;
pub mod is_standard_orientation;
pub mod is_valid;
pub mod length;
pub mod rings;
pub mod rings_overlap;
pub mod self_intersects;
pub mod wkb;
pub mod wkt;
