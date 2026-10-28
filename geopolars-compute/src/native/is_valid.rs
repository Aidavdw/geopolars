//! The ring and shape checks of `is_valid_geoarrow` and `is_valid_ogc`.
//! See them in Python for the rules.
//! The coordinate checks are in [super::coordinate_selection].
//!
//! # Const generic flags
//!
//! ## `Z`
//!
//! The ring closure is generic over `Z`.
//! If `Z=true`, a ring has to close in z as well.

#[cfg(feature = "geos")]
use std::ops::Range;

use geopolars_core::{same_place, ScalarPolygon};
#[cfg(feature = "geos")]
use geopolars_core::{ValidCoords, MIN_RING_VERTICES};

/// Whether every ring of `polygon` is empty or closed, as GeoArrow asks.
#[inline(always)]
pub(crate) fn polygon_rings_are_closed<const Z: bool>(polygon: &ScalarPolygon) -> bool {
    polygon.rings().all(|ring| {
        let vertices = ring.vertices;
        vertices.is_empty() || same_place::<Z>(ring.coords, vertices.start, vertices.end - 1)
    })
}

/// Whether `vertices` hold at least `min` vertices that lie elsewhere in x and y than the one before.
#[cfg(feature = "geos")]
#[inline(always)]
fn has_at_least_distinct_vertices(
    coords: &ValidCoords,
    vertices: Range<usize>,
    min: usize,
) -> bool {
    let start = vertices.start;
    vertices
        .filter(|&i| i == start || !same_place::<false>(coords, i - 1, i))
        .take(min)
        .count()
        == min
}

/// Whether `vertices` make a linestring OGC accepts: none, or at least two distinct ones.
#[cfg(feature = "geos")]
#[inline(always)]
pub(crate) fn line_string_shape_is_valid(coords: &ValidCoords, vertices: Range<usize>) -> bool {
    vertices.is_empty() || has_at_least_distinct_vertices(coords, vertices, 2)
}

/// Whether `polygon` has a shape OGC accepts:
/// every ring empty, or every ring with at least 4 distinct vertices.
#[cfg(feature = "geos")]
#[inline(always)]
pub(crate) fn polygon_shape_is_valid(polygon: &ScalarPolygon) -> bool {
    if polygon.has_empty_exterior() {
        return polygon.rings().all(|ring| ring.vertices.is_empty());
    }
    polygon
        .rings()
        .all(|ring| has_at_least_distinct_vertices(ring.coords, ring.vertices, MIN_RING_VERTICES))
}
