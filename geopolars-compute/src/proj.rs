//! Interfacing with PROJ, to move coordinates from one CRS to another.
//!
//! PROJ takes one `(x, y)` at a time, so only `x` and `y` are reprojected.
//! `z` and `m` keep their buffers.

use geopolars_core::{Coords, ValidCoords};
use proj::Proj;

use crate::{GeoComputeError, GeoComputeResult};

/// For a column of points
/// A missing point is never handed to PROJ, and keeps its `x` and `y`.
pub fn reproject_coords(proj: &Proj, coords: &Coords) -> GeoComputeResult<Coords> {
    let xy = with_const_flags!(reprojected_points(proj, coords); !coords.is_all_valid())?;
    let axes = coords.ignoring_validity().with_positions(xy);
    Ok(Coords::from_valid(axes, coords.validity().cloned()))
}

/// For the coordinates of a nested kind, which are never missing.
pub fn reproject_valid_coords(proj: &Proj, coords: &ValidCoords) -> GeoComputeResult<ValidCoords> {
    let (x, y) = (coords.x(), coords.y());
    let mut xs = Vec::with_capacity(coords.len());
    let mut ys = Vec::with_capacity(coords.len());
    for (&x, &y) in x.iter().zip(y) {
        let (x, y) = reproject_xy(proj, (x, y))?;
        xs.push(x);
        ys.push(y);
    }
    Ok(coords.with_positions([Some(xs), Some(ys), None]))
}

/// The new `x` and `y` of [`reproject_coords`].
/// If `NULLS=false`, every point is assumed to be there.
fn reprojected_points<const NULLS: bool>(
    proj: &Proj,
    coords: &Coords,
) -> GeoComputeResult<[Option<Vec<f64>>; 3]> {
    let values = coords.ignoring_validity();
    let (x, y) = (values.x(), values.y());
    let mut xs = Vec::with_capacity(coords.len());
    let mut ys = Vec::with_capacity(coords.len());
    for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
        let (x, y) = if !NULLS || coords.is_valid(i) {
            reproject_xy(proj, (x, y))?
        } else {
            (x, y)
        };
        xs.push(x);
        ys.push(y);
    }
    Ok([Some(xs), Some(ys), None])
}

#[inline(always)]
fn reproject_xy(proj: &Proj, (x, y): (f64, f64)) -> GeoComputeResult<(f64, f64)> {
    proj.convert((x, y))
        .map_err(|source| GeoComputeError::Reproject {
            position: (x, y),
            source,
        })
}
