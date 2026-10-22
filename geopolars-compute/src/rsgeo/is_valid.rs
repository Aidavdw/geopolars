//! Topology checks used in `is_valid`
//! Answers how the rings of a polygon, and the parts of a multipolygon, fit together in x and y.
//! See `is_valid` in Python for the rules.
//!
//! Every polygon is assumed to have a valid shape already
//! (see [crate::native::polygon_shape_is_valid]),
//! and to be finite in x and y (see [crate::native::coordinates_are_valid]).

use ::rsgeo::coordinate_position::CoordPos;
use ::rsgeo::dimensions::Dimensions;
use ::rsgeo::relate::IntersectionMatrix;
use ::rsgeo::{Polygon, PreparedGeometry, Relate};
use geopolars_core::{ScalarLineString, ScalarPolygon};

use super::rings_overlap::overlap;
use super::self_intersects::finite_line_self_intersects;
use super::to_rsgeo_ring;

fn touch_only_at_points(matrix: &IntersectionMatrix) -> bool {
    matrix.get(CoordPos::OnBoundary, CoordPos::OnBoundary) != Dimensions::OneDimensional
}

/// Whether a polygon's rings fit together, each given as the polygon it encloses.
/// The first is the exterior, the others are holes.
/// See `is_valid` in Python for the rules.
fn polygon_rings_fit(rings: &[Polygon<f64>]) -> bool {
    // Re-use matrix, as getting that is the expensive part.
    let [exterior, holes @ ..] = rings else {
        return true;
    };
    let exterior = PreparedGeometry::from(exterior);
    let holes: Vec<_> = holes.iter().map(PreparedGeometry::from).collect();
    holes.iter().enumerate().all(|(i, hole)| {
        let matrix = exterior.relate(hole);
        matrix.is_contains()
            && touch_only_at_points(&matrix)
            && holes[i + 1..].iter().all(|other| {
                let matrix = hole.relate(other);
                !overlap::<false>(&matrix) && touch_only_at_points(&matrix)
            })
    })
}

/// Whether a multipolygon's parts fit together:
/// no two parts share any area, and they only touch at points.
pub(crate) fn multi_polygon_parts_fit(parts: &[Polygon<f64>]) -> bool {
    let prepared: Vec<_> = parts.iter().map(PreparedGeometry::from).collect();
    prepared.iter().enumerate().all(|(i, part)| {
        prepared[i + 1..].iter().all(|other| {
            let matrix = part.relate(other);
            !overlap::<false>(&matrix) && touch_only_at_points(&matrix)
        })
    })
}

/// The checks on how a polygon's rings, and a multipolygon's parts, fit together.
/// See `is_valid` in Python.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FitChecks {
    pub check_ring_self_intersect: bool,
    pub check_polygon_self_overlap: bool,
    /// Only read for multipolygons.
    pub check_multi_polygon_self_overlap: bool,
}

/// Whether `ring` does not cross itself, if `check_ring_self_intersect=true`.
fn ring_is_valid(ring: &ScalarLineString, checks: &FitChecks) -> bool {
    !(checks.check_ring_self_intersect && finite_line_self_intersects(ring))
}

/// Whether the rings of `polygon`, which passed the shape checks, are valid and fit together.
/// If `keep=true`, they are returned as one rsgeo polygon to relate further.
/// `Err(())` if they are not valid, `Ok(None)` for the empty polygon (or without `keep`).
pub(crate) fn polygon_fits(
    polygon: &ScalarPolygon,
    checks: &FitChecks,
    keep: bool,
) -> Result<Option<Polygon<f64>>, ()> {
    // The shape checks left either every ring empty, or none.
    if polygon
        .rings()
        .next()
        .is_none_or(|exterior| exterior.vertices.is_empty())
    {
        return Ok(None);
    }
    if !polygon.rings().all(|ring| ring_is_valid(&ring, checks)) {
        return Err(());
    }
    // A lone exterior has nothing to fit with.
    let fit = checks.check_polygon_self_overlap && polygon.parts.len() > 1;
    if !(fit || keep) {
        return Ok(None);
    }
    // Handed to rsgeo, which needs them owned: the only copy made of the vertices.
    let rings: Vec<_> = polygon.rings().map(|ring| to_rsgeo_ring(&ring)).collect();
    if fit && !polygon_rings_fit(&rings) {
        return Err(());
    }
    Ok(keep.then(|| {
        let mut rings = rings.into_iter().map(|ring| ring.into_inner().0);
        let exterior = rings
            .next()
            .expect("a polygon that is not empty has an exterior");
        Polygon::new(exterior, rings.collect())
    }))
}
