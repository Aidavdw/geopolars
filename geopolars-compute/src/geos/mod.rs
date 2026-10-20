//! Interfacing with the `geos` crate.
//!
//! GEOS, as a C++ library, owns its coordinate storage,
//! so arrow buffers can't be borrowed zero-copy.
//! The copy is one memcpy-like op, so still not very heavy.

mod convex_hull;
mod linestrings;
mod polygons;
mod simplify;

pub use convex_hull::{
    line_string_convex_hulls, multi_line_string_convex_hulls, multi_point_convex_hulls,
    multi_polygon_convex_hulls, polygon_convex_hulls,
};
pub use polygons::{polygons_from_geos, GeosPolygons};
pub use simplify::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};

use std::ops::Range;

use geopolars_core::output::OwnedAxes;
use geopolars_core::{Coords, GeoDimension, Point, PointArray, ValidCoords};
use geos::{CoordSeq, CoordType, Geom, Geometry};

/// Converts a view into GEOS geometries.
/// A single geometry gives one GEOS geometry.
/// An array gives one per row, with `None` where the row is missing.
pub trait ToGeos {
    type Geos;

    fn to_geos(&self) -> geos::GResult<Self::Geos>;
}

fn coord_type(dim: GeoDimension) -> CoordType {
    match dim {
        GeoDimension::XY => CoordType::XY,
        GeoDimension::XYZ => CoordType::XYZ,
        GeoDimension::XYM => CoordType::XYM,
        GeoDimension::XYZM => CoordType::XYZM,
    }
}

/// `None` if the point is empty (missing).
impl ToGeos for Point<'_> {
    type Geos = Option<Geometry>;

    fn to_geos(&self) -> geos::GResult<Option<Geometry>> {
        let dim = self.coords.dim();
        point_to_geos(
            self.coords,
            self.index,
            dim.field_names().len(),
            coord_type(dim),
        )
    }
}

impl ToGeos for PointArray {
    type Geos = Vec<Option<Geometry>>;

    fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        // GEOS has no batch constructor, so only the per-column work is shared.
        let dim = self.coords.dim();
        let len = dim.field_names().len();
        let coord_type = coord_type(dim);
        let mut geometries = Vec::with_capacity(self.coords.len());
        for i in 0..self.coords.len() {
            geometries.push(point_to_geos(&self.coords, i, len, coord_type)?);
        }
        Ok(geometries)
    }
}

/// Point `i` of `coords`, which carry `len` axes of `coord_type`.
/// `None` if it is missing.
#[inline(always)]
fn point_to_geos(
    coords: &Coords,
    i: usize,
    len: usize,
    coord_type: CoordType,
) -> geos::GResult<Option<Geometry>> {
    let Some((x, y)) = coords.xy(i) else {
        return Ok(None);
    };
    // GeoArrow's empty point. GEOS only counts a point without coordinates as empty.
    if x.is_nan() && y.is_nan() {
        return Geometry::create_empty_point().map(Some);
    }
    let mut buffer = [0.0; 4];
    for (n, value) in buffer[..len].iter_mut().enumerate() {
        *value = coords.nth(i, n);
    }
    let seq = CoordSeq::new_from_buffer(&buffer[..len], 1, coord_type)?;
    Geometry::create_point(seq).map(Some)
}

/// The x, y, z and m values of every coordinate, as [`CoordSeq::new_from_arrays`] takes them.
type Axes<'a> = (&'a [f64], &'a [f64], Option<&'a [f64]>, Option<&'a [f64]>);

fn axes(coords: &ValidCoords) -> Axes<'_> {
    let (x, y) = (coords.axis(0), coords.axis(1));
    match coords.dim() {
        GeoDimension::XY => (x, y, None, None),
        GeoDimension::XYZ => (x, y, Some(coords.axis(2)), None),
        GeoDimension::XYM => (x, y, None, Some(coords.axis(2))),
        GeoDimension::XYZM => (x, y, Some(coords.axis(2)), Some(coords.axis(3))),
    }
}

/// The coordinates at `vertices`, copied from `axes` in one go.
#[inline(always)]
fn coord_seq(axes: &Axes, vertices: Range<usize>) -> geos::GResult<CoordSeq> {
    let (x, y, z, m) = axes;
    CoordSeq::new_from_arrays(
        &x[vertices.clone()],
        &y[vertices.clone()],
        z.map(|z| &z[vertices.clone()]),
        m.map(|m| &m[vertices]),
    )
}

/// Appends the coordinates of `geometry`, a linestring or a ring, to the axes.
/// If `z` is given, they keep their z.
fn append_coords(
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    z: Option<&mut Vec<f64>>,
    geometry: &impl Geom,
) -> geos::GResult<()> {
    let dim = match z {
        Some(_) => GeoDimension::XYZ,
        None => GeoDimension::XY,
    };
    let buffer = geometry.get_coord_seq()?.as_buffer(Some(coord_type(dim)))?;
    let width = dim.field_names().len();
    // Once per linestring or ring, rather than per coordinate.
    match z {
        Some(z) => buffer.chunks_exact(width).for_each(|coord| {
            x.push(coord[0]);
            y.push(coord[1]);
            z.push(coord[2]);
        }),
        None => buffer.chunks_exact(width).for_each(|coord| {
            x.push(coord[0]);
            y.push(coord[1]);
        }),
    }
    Ok(())
}

/// Contains arrow-ified output of a GEOS function.
/// - Coordinates (x, y, and z if there is one)
/// - the list levels above them, built from one GEOS geometry after the other.
///
/// The output of GEOS never contains m values.
pub(super) struct Buffers {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Option<Vec<f64>>,
    /// One per list level, innermost first.
    offsets: Vec<Vec<i64>>,
}

/// How far [`Buffers`] was filled, to go back to with [`Buffers::truncate`].
/// Sized for the most list levels a geometry has (a multipolygon's three),
/// so taking it allocates nothing.
pub(super) struct Filled {
    coords: usize,
    offsets: [usize; 3],
}

impl Buffers {
    /// With `has_z=true` the coordinates keep their z.
    pub(super) fn new(has_z: bool, levels: usize, coords: usize) -> Self {
        Self {
            x: Vec::with_capacity(coords),
            y: Vec::with_capacity(coords),
            z: has_z.then(|| Vec::with_capacity(coords)),
            offsets: vec![vec![0]; levels],
        }
    }

    fn dim(&self) -> GeoDimension {
        match self.z {
            Some(_) => GeoDimension::XYZ,
            None => GeoDimension::XY,
        }
    }

    /// Appends the coordinates of `geometry`, a linestring or a ring.
    pub(super) fn push_coords(&mut self, geometry: &impl Geom) -> geos::GResult<()> {
        append_coords(&mut self.x, &mut self.y, self.z.as_mut(), geometry)
    }

    /// Ends the current list at `level`, after everything pushed to the level below it.
    #[inline(always)]
    pub(super) fn close(&mut self, level: usize) {
        let end = match level {
            0 => self.x.len(),
            _ => self.offsets[level - 1].len() - 1,
        };
        self.offsets[level].push(end as i64);
    }

    /// Ends the current row: the list at the outermost level.
    #[inline(always)]
    pub(super) fn close_row(&mut self) {
        self.close(self.offsets.len() - 1);
    }

    pub(super) fn filled(&self) -> Filled {
        Filled {
            coords: self.x.len(),
            offsets: std::array::from_fn(|level| self.offsets.get(level).map_or(0, Vec::len)),
        }
    }

    /// Drops everything pushed since `filled`.
    pub(super) fn truncate(&mut self, filled: &Filled) {
        self.x.truncate(filled.coords);
        self.y.truncate(filled.coords);
        if let Some(z) = &mut self.z {
            z.truncate(filled.coords);
        }
        for (offsets, &len) in self.offsets.iter_mut().zip(&filled.offsets) {
            offsets.truncate(len);
        }
    }

    /// The dimension, axes and `N` offsets (innermost first, so the rows last),
    /// for the [output](geopolars_core::output) array of the kind built.
    pub(super) fn finish<const N: usize>(self) -> (GeoDimension, OwnedAxes, [Vec<i64>; N]) {
        let dim = self.dim();
        let offsets = self
            .offsets
            .try_into()
            .unwrap_or_else(|_| panic!("built with {N} list levels"));
        (dim, (self.x, self.y, self.z, None), offsets)
    }
}
