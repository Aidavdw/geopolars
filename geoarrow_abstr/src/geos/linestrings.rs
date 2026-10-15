//! Linestrings and multilinestrings to GEOS.

use std::ops::Range;

use geos::Geometry;

use super::{axes, coord_seq, Axes};
use crate::{
    GeoArrowListView, GeoArrowValidCoordsView, GeoArrowValidListView, LineStringArrayView,
    LineStringView, MultiLineStringArrayView, MultiLineStringView,
};

impl<C: GeoArrowValidCoordsView> LineStringView<'_, C> {
    pub fn to_geos(&self) -> geos::GResult<Geometry> {
        line_string_to_geos(&axes(self.coords), self.vertices.clone())
    }
}

impl<R: GeoArrowListView, C: GeoArrowValidCoordsView> LineStringArrayView<R, C> {
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

impl<L: GeoArrowValidListView, C: GeoArrowValidCoordsView> MultiLineStringView<'_, L, C> {
    pub fn to_geos(&self) -> geos::GResult<Geometry> {
        multi_line_string_to_geos(&axes(self.coords), self.lines, self.parts.clone())
    }
}

impl<R: GeoArrowListView, L: GeoArrowValidListView, C: GeoArrowValidCoordsView>
    MultiLineStringArrayView<R, L, C>
{
    /// One geometry per row; `None` where the row is missing.
    pub fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
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
fn line_string_to_geos(axes: &Axes, vertices: Range<usize>) -> geos::GResult<Geometry> {
    Geometry::create_line_string(coord_seq(axes, vertices)?)
}

/// No parts gives an empty multilinestring.
#[inline(always)]
fn multi_line_string_to_geos(
    axes: &Axes,
    lines: &impl GeoArrowValidListView,
    parts: Range<usize>,
) -> geos::GResult<Geometry> {
    // GEOS takes ownership of the parts, so they can't share a buffer across rows.
    let mut line_strings = Vec::with_capacity(parts.len());
    for j in parts {
        line_strings.push(line_string_to_geos(axes, lines.range(j))?);
    }
    Geometry::create_multiline_string(line_strings)
}
