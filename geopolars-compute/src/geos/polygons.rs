//! Polygons and multipolygons to GEOS, and back.

use std::ops::Range;

use geopolars_core::output::SizeHint;
use geopolars_core::{
    List, MultiPolygon, MultiPolygonArray, Polygon, PolygonArray, ValidCoords, ValidList,
};
use geos::{ConstGeometry, Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::Offsets;

use super::{append_coords, axes, coord_seq, Axes, Buffers, ToGeos};

impl ToGeos for Polygon<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> geos::GResult<Geometry> {
        polygon_to_geos(&axes(self.coords), self.rings, self.parts.clone())
    }
}

impl ToGeos for PolygonArray {
    type Geos = Vec<Option<Geometry>>;

    fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        let axes = axes(&self.coords);
        let mut geometries = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            geometries.push(if self.rows.is_valid(i) {
                Some(polygon_to_geos(&axes, &self.rings, self.rows.range(i))?)
            } else {
                None
            });
        }
        Ok(geometries)
    }
}

impl ToGeos for MultiPolygon<'_> {
    type Geos = Geometry;

    fn to_geos(&self) -> geos::GResult<Geometry> {
        multi_polygon_to_geos(
            &axes(self.coords),
            self.polygons,
            self.rings,
            self.parts.clone(),
        )
    }
}

impl ToGeos for MultiPolygonArray {
    type Geos = Vec<Option<Geometry>>;

    fn to_geos(&self) -> geos::GResult<Vec<Option<Geometry>>> {
        let axes = axes(&self.coords);
        let mut geometries = Vec::with_capacity(self.rows.len());
        for i in 0..self.rows.len() {
            // A missing row's offsets point anywhere, so it is not converted.
            geometries.push(if self.rows.is_valid(i) {
                Some(multi_polygon_to_geos(
                    &axes,
                    &self.polygons,
                    &self.rings,
                    self.rows.range(i),
                )?)
            } else {
                None
            });
        }
        Ok(geometries)
    }
}

/// The polygon of `rings` `parts`, the first of which is the exterior.
/// No parts gives an empty polygon.
#[inline(always)]
pub(super) fn polygon_to_geos(
    axes: &Axes,
    rings: &ValidList,
    parts: Range<usize>,
) -> geos::GResult<Geometry> {
    let mut parts = parts.map(|j| Geometry::create_linear_ring(coord_seq(axes, rings.range(j))?));
    // GEOS can't build a polygon without an exterior, so an empty one has its own constructor.
    let Some(exterior) = parts.next() else {
        return Geometry::create_empty_polygon();
    };
    Geometry::create_polygon(exterior?, parts.collect::<geos::GResult<_>>()?)
}

/// The multipolygon of `polygons` `parts`.
/// No parts gives an empty multipolygon.
#[inline(always)]
pub(super) fn multi_polygon_to_geos(
    axes: &Axes,
    polygons: &ValidList,
    rings: &ValidList,
    parts: Range<usize>,
) -> geos::GResult<Geometry> {
    // GEOS takes ownership of the parts, so they can't share a buffer across rows.
    let mut geometries = Vec::with_capacity(parts.len());
    for j in parts {
        geometries.push(polygon_to_geos(axes, rings, polygons.range(j))?);
    }
    Geometry::create_multipolygon(geometries)
}

/// GEOS polygons, one per row, for [`polygons_from_geos`].
pub struct GeosPolygons {
    /// `None` gives a missing row.
    pub geometries: Vec<Option<Geometry>>,
    pub size_hint: SizeHint<1>,
}

impl GeosPolygons {
    /// The exact size hint of `geometries`, counted through GEOS.
    /// A geometry that is not a polygon is skipped.
    pub fn size_hint(geometries: &[Option<Geometry>]) -> geos::GResult<SizeHint<1>> {
        let mut size_hint = SizeHint {
            coords: 0,
            parts: [0],
        };
        for geometry in geometries.iter().flatten() {
            // GEOS gives an empty polygon an empty exterior, which we don't store.
            if geometry.geometry_type()? == GeometryTypes::Polygon && !geometry.is_empty()? {
                size_hint.coords += geometry.get_num_coordinates()?;
                // rings
                size_hint.parts[0] += 1 + geometry.get_num_interior_rings()?;
            }
        }
        Ok(size_hint)
    }
}

/// Our polygon array of `polygons`, one row each.
/// Every GEOS polygon is freed once it is copied.
/// If `has_z=true`, the coordinates keep their z.
/// Errors for a geometry that is not a polygon.
pub fn polygons_from_geos(polygons: GeosPolygons, has_z: bool) -> geos::GResult<PolygonArray> {
    let GeosPolygons {
        geometries,
        size_hint,
    } = polygons;

    let len = geometries.len();
    let mut x = Vec::with_capacity(size_hint.coords);
    let mut y = Vec::with_capacity(size_hint.coords);
    let mut z = has_z.then(|| Vec::with_capacity(size_hint.coords));
    let mut rings = Vec::with_capacity(size_hint.parts[0] + 1);
    rings.push(0);
    let mut rows = Vec::with_capacity(len + 1);
    rows.push(0);
    // Only built once a row is missing.
    let mut validity: Option<MutableBitmap> = None;
    for (i, geometry) in geometries.into_iter().enumerate() {
        match geometry {
            Some(polygon) => {
                if polygon.geometry_type()? != GeometryTypes::Polygon {
                    return Err(geos::Error::ConversionError(format!(
                        "expected a polygon, got a {:?}",
                        polygon.geometry_type()?
                    )));
                }
                // GEOS gives an empty polygon an empty exterior, which we don't store.
                if !polygon.is_empty()? {
                    let mut push_ring = |ring: &ConstGeometry| {
                        append_coords(&mut x, &mut y, z.as_mut(), ring)?;
                        rings.push(x.len() as i64);
                        geos::GResult::Ok(())
                    };
                    push_ring(&polygon.get_exterior_ring()?)?;
                    for n in 0..polygon.get_num_interior_rings()? {
                        push_ring(&polygon.get_interior_ring_n(n)?)?;
                    }
                }
                if let Some(validity) = &mut validity {
                    validity.push(true);
                }
            }
            None => validity
                .get_or_insert_with(|| {
                    let mut validity = MutableBitmap::with_capacity(len);
                    validity.extend_constant(i, true);
                    validity
                })
                .push(false),
        }
        rows.push(rings.len() as i64 - 1);
    }

    // SAFETY: both start at 0 and only grow, ending at the length of the level below.
    let (rings, rows) = unsafe {
        (
            Offsets::new_unchecked(rings).into(),
            Offsets::new_unchecked(rows).into(),
        )
    };
    Ok(PolygonArray {
        rows: List::from_offsets(rows, validity.map(Bitmap::from)),
        rings: ValidList::from_offsets(rings),
        coords: ValidCoords::from_axes((x, y, z, None)),
    })
}

/// Appends the rings of a GEOS polygon, each closed at level 0.
/// The polygon itself is left open for the caller to close.
/// Gives `false` for any other kind of geometry, such as a multipolygon.
/// Nothing is appended then.
#[inline(always)]
pub(super) fn polygon_from_geos(
    buffers: &mut Buffers,
    geometry: &impl Geom,
) -> geos::GResult<bool> {
    if geometry.geometry_type()? != GeometryTypes::Polygon {
        return Ok(false);
    }
    push_rings(buffers, geometry)?;
    Ok(true)
}

/// Appends the parts of a GEOS multipolygon, each closed at level 1.
/// The multipolygon itself is left open for the caller to close.
/// GEOS gives a multipolygon of one part as a plain polygon.
/// So a polygon is read as a single part.
/// An empty geometry has no parts.
/// Gives `false` for any other kind of geometry.
/// Nothing is appended then.
#[inline(always)]
pub(super) fn multi_polygon_from_geos(
    buffers: &mut Buffers,
    geometry: &impl Geom,
) -> geos::GResult<bool> {
    match geometry.geometry_type()? {
        GeometryTypes::Polygon if geometry.is_empty()? => {}
        GeometryTypes::Polygon => {
            push_rings(buffers, geometry)?;
            buffers.close(1);
        }
        GeometryTypes::MultiPolygon => {
            for n in 0..geometry.get_num_geometries()? {
                push_rings(buffers, &geometry.get_geometry_n(n)?)?;
                buffers.close(1);
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Appends the rings of `polygon`, each closed at level 0.
/// The exterior comes first, then the interiors.
/// An empty polygon has no rings.
#[inline(always)]
fn push_rings(buffers: &mut Buffers, polygon: &impl Geom) -> geos::GResult<()> {
    // GEOS gives an empty polygon an empty exterior.
    // We would store that as a ring without vertices.
    if polygon.is_empty()? {
        return Ok(());
    }
    buffers.push_coords(&polygon.get_exterior_ring()?)?;
    buffers.close(0);
    for n in 0..polygon.get_num_interior_rings()? {
        buffers.push_coords(&polygon.get_interior_ring_n(n)?)?;
        buffers.close(0);
    }
    Ok(())
}
