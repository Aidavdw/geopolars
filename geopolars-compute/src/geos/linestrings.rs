//! Linestrings and multilinestrings to GEOS, and back.

use std::ops::Range;

use geopolars_core::{
    LineString, LineStringArray, MultiLineString, MultiLineStringArray, ValidList,
};
use geos::{Geom, Geometry, GeometryTypes};

use super::{axes, coord_seq, Axes, Buffers, ToGeos};

impl ToGeos for LineString<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> geos::GResult<Geometry> {
        line_string_to_geos(&axes(self.coords), self.vertices.clone())
    }
}

impl ToGeos for LineStringArray {
    type Geos = Vec<Option<Geometry>>;

    fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
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

impl ToGeos for MultiLineString<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> geos::GResult<Geometry> {
        multi_line_string_to_geos(&axes(self.coords), self.lines, self.parts.clone())
    }
}

impl ToGeos for MultiLineStringArray {
    type Geos = Vec<Option<Geometry>>;

    fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        let axes = axes(&self.coords);
        let mut geometries = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            geometries.push(if self.rows.is_valid(i) {
                Some(multi_line_string_to_geos(
                    &axes,
                    &self.lines,
                    self.rows.range(i),
                )?)
            } else {
                None
            });
        }
        Ok(geometries)
    }
}

/// Can copy from axes in one go.
/// No vertices gives an empty line string.
#[inline(always)]
pub(super) fn line_string_to_geos(axes: &Axes, vertices: Range<usize>) -> geos::GResult<Geometry> {
    Geometry::create_line_string(coord_seq(axes, vertices)?)
}

/// No parts gives an empty multilinestring.
#[inline(always)]
pub(super) fn multi_line_string_to_geos(
    axes: &Axes,
    lines: &ValidList,
    parts: Range<usize>,
) -> geos::GResult<Geometry> {
    // GEOS takes ownership of the parts, so they can't share a buffer across rows.
    let mut line_strings = Vec::with_capacity(parts.len());
    for j in parts {
        line_strings.push(line_string_to_geos(axes, lines.range(j))?);
    }
    Geometry::create_multiline_string(line_strings)
}

/// Appends the coordinates of a GEOS linestring.
/// Their list is left open for the caller to close.
/// Gives `false` for any other kind of geometry.
/// Nothing is appended then.
#[inline(always)]
pub(super) fn line_string_from_geos(
    buffers: &mut Buffers,
    geometry: &impl Geom,
) -> geos::GResult<bool> {
    if geometry.geometry_type()? != GeometryTypes::LineString {
        return Ok(false);
    }
    buffers.push_coords(geometry)?;
    Ok(true)
}

/// Appends the parts of a GEOS multilinestring, each closed at level 0.
/// The multilinestring itself is left open for the caller to close.
/// GEOS gives a multilinestring of one part as a plain linestring.
/// So a linestring is read as a single part.
/// An empty geometry has no parts.
/// Gives `false` for any other kind of geometry.
/// Nothing is appended then.
#[inline(always)]
pub(super) fn multi_line_string_from_geos(
    buffers: &mut Buffers,
    geometry: &impl Geom,
) -> geos::GResult<bool> {
    match geometry.geometry_type()? {
        GeometryTypes::LineString if geometry.is_empty()? => {}
        GeometryTypes::LineString => {
            buffers.push_coords(geometry)?;
            buffers.close(0);
        }
        GeometryTypes::MultiLineString => {
            for n in 0..geometry.get_num_geometries()? {
                buffers.push_coords(&geometry.get_geometry_n(n)?)?;
                buffers.close(0);
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}
