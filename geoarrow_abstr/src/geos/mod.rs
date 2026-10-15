//! Interfacing with the `geos` crate.
//!
//! GEOS, as a C++ library, owns its coordinate storage,
//! so arrow buffers can't be borrowed zero-copy.
//! The copy is one memcpy-like op, so still not very heavy.

use std::ops::Range;

use geos::{CoordSeq, CoordType, Geometry};

use crate::{
    GeoArrowCoordsView, GeoArrowListView, GeoArrowValidCoordsView, GeoDimension, LineString,
    LineStringArray, Point, PointArray,
};

impl From<GeoDimension> for CoordType {
    fn from(dim: GeoDimension) -> Self {
        match dim {
            GeoDimension::XY => CoordType::XY,
            GeoDimension::XYZ => CoordType::XYZ,
            GeoDimension::XYM => CoordType::XYM,
            GeoDimension::XYZM => CoordType::XYZM,
        }
    }
}

impl<C: GeoArrowCoordsView> Point<'_, C> {
    /// `None` if the point is empty (missing).
    pub fn to_geos(&self) -> geos::GResult<Option<Geometry>> {
        let dim = self.coords.dim();
        point_to_geos(self.coords, self.index, dim.field_names().len(), dim.into())
    }
}

impl<C: GeoArrowCoordsView> PointArray<C> {
    /// One geometry per row; `None` where the point is missing.
    pub fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        // GEOS has no batch constructor, so only the per-column work is shared.
        let dim = self.coords.dim();
        let len = dim.field_names().len();
        let coord_type: CoordType = dim.into();
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
fn point_to_geos<C: GeoArrowCoordsView>(
    coords: &C,
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

fn axes<C: GeoArrowValidCoordsView>(coords: &C) -> Axes<'_> {
    let (x, y) = (coords.axis(0), coords.axis(1));
    match coords.dim() {
        GeoDimension::XY => (x, y, None, None),
        GeoDimension::XYZ => (x, y, Some(coords.axis(2)), None),
        GeoDimension::XYM => (x, y, None, Some(coords.axis(2))),
        GeoDimension::XYZM => (x, y, Some(coords.axis(2)), Some(coords.axis(3))),
    }
}

impl<C: GeoArrowValidCoordsView> LineString<'_, C> {
    pub fn to_geos(&self) -> geos::GResult<Geometry> {
        line_string_to_geos(&axes(self.coords), self.vertices.clone())
    }
}

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArray<R, C> {
    /// One geometry per row; `None` where the row is missing.
    pub fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        let axes = axes(&self.coords);
        let mut geometries = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            geometries.push(if self.rows.is_valid(i) {
                Some(line_string_to_geos(&axes, self.rows.range(i))?)
            } else {
                None
            });
        }
        Ok(geometries)
    }
}

/// The line string through `vertices`, copied from `axes` in one go.
/// No vertices gives an empty line string.
#[inline(always)]
fn line_string_to_geos(axes: &Axes, vertices: Range<usize>) -> geos::GResult<Geometry> {
    let (x, y, z, m) = axes;
    let seq = CoordSeq::new_from_arrays(
        &x[vertices.clone()],
        &y[vertices.clone()],
        z.map(|z| &z[vertices.clone()]),
        m.map(|m| &m[vertices]),
    )?;
    Geometry::create_line_string(seq)
}
