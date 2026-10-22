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
pub use simplify::{
    simplify_line_strings, simplify_multi_line_strings, simplify_multi_polygons, simplify_polygons,
};

use std::ops::Range;

use geopolars_core::output::SizeHint;
use geopolars_core::{Coords, GeoDimension, List, PointArray, ScalarPoint, ValidCoords};
use geos::{CoordSeq, CoordType, Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::{Offsets, OffsetsBuffer};

use crate::{row_or_missing, GeoComputeError, GeoComputeResult};

/// Converts a single geometry into a GEOS geometry.
/// For an array, see [`ToGeosRows`].
pub trait ToGeos {
    type Geos;

    fn to_geos(&self) -> GeoComputeResult<Self::Geos>;
}

/// Converts an array into one [`GeosRow`] per row.
/// A missing row is `Missing`.
/// A row without parts is `Empty`, so it never goes to GEOS.
/// The size hint is the size of the array itself.
/// If `ignore_per_row_errors=true`, a row GEOS fails to build is missing too;
/// otherwise its error is returned.
/// See python for `early_finite_check`
pub trait ToGeosRows {
    fn to_geos_rows(
        &self,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries>;
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
impl ToGeos for ScalarPoint<'_> {
    type Geos = Option<Geometry>;

    fn to_geos(&self) -> GeoComputeResult<Option<Geometry>> {
        let dim = self.coords.dim();
        point_to_geos::<false>(
            self.coords,
            self.index,
            dim.field_names().len(),
            coord_type(dim),
        )
    }
}

impl ToGeosRows for PointArray {
    fn to_geos_rows(
        &self,
        ignore_per_row_errors: bool,
        early_finite_check: bool,
    ) -> GeoComputeResult<GeosGeometries> {
        let geometries = with_const_flags!(
            points_to_geos(&self.coords, ignore_per_row_errors);
            early_finite_check
        )?;
        Ok(GeosGeometries {
            geometries,
            size_hint: SizeHint {
                coords: self.coords.len(),
                parts: [0, 0],
            },
        })
    }
}

fn points_to_geos<const EARLY_FINITE_CHECK: bool>(
    coords: &Coords,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<Vec<GeosRow>> {
    // GEOS has no batch constructor, so only the per-column work is shared.
    let dim = coords.dim();
    let len = dim.field_names().len();
    let coord_type = coord_type(dim);
    let mut geometries = Vec::with_capacity(coords.len());
    for i in 0..coords.len() {
        let point = point_to_geos::<EARLY_FINITE_CHECK>(coords, i, len, coord_type);
        geometries.push(row_or_missing(point, ignore_per_row_errors)?.into());
    }
    Ok(geometries)
}

/// Point `i` of `coords`, which carry `len` axes of `coord_type`.
/// `None` if it is missing.
/// If `EARLY_FINITE_CHECK=true`, a NaN or infinite x or y is an error,
/// unless both are NaN (the empty point).
#[inline(always)]
fn point_to_geos<const EARLY_FINITE_CHECK: bool>(
    coords: &Coords,
    i: usize,
    len: usize,
    coord_type: CoordType,
) -> GeoComputeResult<Option<Geometry>> {
    let Some((x, y)) = coords.xy(i) else {
        return Ok(None);
    };
    // GeoArrow's empty point. GEOS only counts a point without coordinates as empty.
    if x.is_nan() && y.is_nan() {
        return Ok(Some(Geometry::create_empty_point()?));
    }
    if EARLY_FINITE_CHECK {
        xy_is_finite_or_error(&[x], &[y])?;
    }
    let mut buffer = [0.0; 4];
    for (n, value) in buffer[..len].iter_mut().enumerate() {
        *value = coords.nth(i, n);
    }
    let seq = CoordSeq::new_from_buffer(&buffer[..len], 1, coord_type)?;
    Ok(Some(Geometry::create_point(seq)?))
}

/// [`ToGeosRows`] for the rows of a list array, whose size is `size_hint`.
/// `to_geos` builds one geometry from the range of a row's parts.
/// `span` gives where the coordinates of those parts sit in `axes`.
fn rows_to_geos(
    rows: &List,
    axes: &Axes,
    size_hint: SizeHint<2>,
    ignore_per_row_errors: bool,
    early_finite_check: bool,
    span: impl Fn(Range<usize>) -> Range<usize>,
    to_geos: impl Fn(Range<usize>) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<GeosGeometries> {
    let geometries = with_const_flags!(
        rows_to_geos_with(rows, axes, ignore_per_row_errors, span, to_geos);
        rows.is_all_valid(), early_finite_check
    )?;
    Ok(GeosGeometries {
        geometries,
        size_hint,
    })
}

/// See [`rows_to_geos`].
/// If `ALL_VALID=true`, no row is checked for being missing first.
/// `EARLY_FINITE_CHECK` is the `early_finite_check` of [`ToGeosRows`].
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
fn rows_to_geos_with<const ALL_VALID: bool, const EARLY_FINITE_CHECK: bool>(
    rows: &List,
    axes: &Axes,
    ignore_per_row_errors: bool,
    span: impl Fn(Range<usize>) -> Range<usize>,
    to_geos: impl Fn(Range<usize>) -> GeoComputeResult<Geometry>,
) -> GeoComputeResult<Vec<GeosRow>> {
    let (x, y, _, _) = axes;
    let mut geometries = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not converted.
        if !ALL_VALID && !rows.is_valid(i) {
            geometries.push(GeosRow::Missing);
            continue;
        }
        let parts = rows.range(i);
        geometries.push(if parts.is_empty() {
            GeosRow::Empty
        } else {
            let geometry = if EARLY_FINITE_CHECK {
                let coords = span(parts.clone());
                xy_is_finite_or_error(&x[coords.clone()], &y[coords]).and_then(|()| to_geos(parts))
            } else {
                to_geos(parts)
            };
            row_or_missing(geometry.map(Some), ignore_per_row_errors)?.into()
        });
    }
    Ok(geometries)
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

/// An error for the first coordinate with a NaN or infinite `x` or `y`, if there is one.
/// GEOS takes neither, and has no empty point inside a geometry.
/// z is not checked: GEOS itself uses a NaN z for a coordinate without one.
#[inline(always)]
fn xy_is_finite_or_error(x: &[f64], y: &[f64]) -> GeoComputeResult<()> {
    // Does not stop at the first that fails, so that it vectorises.
    let finite = |axis: &[f64]| axis.iter().fold(true, |ok, value| ok & value.is_finite());
    if finite(x) && finite(y) {
        return Ok(());
    }
    let position = x
        .iter()
        .zip(y)
        .map(|(&x, &y)| (x, y))
        .find(|(x, y)| !(x.is_finite() && y.is_finite()))
        .expect("a non-finite coordinate, as the check above failed");
    Err(GeoComputeError::NotFinite { position })
}

/// The coordinates at `vertices`, copied from `axes` in one go.
#[inline(always)]
fn coord_seq(axes: &Axes, vertices: Range<usize>) -> GeoComputeResult<CoordSeq> {
    let (x, y, z, m) = axes;
    Ok(CoordSeq::new_from_arrays(
        &x[vertices.clone()],
        &y[vertices.clone()],
        z.map(|z| &z[vertices.clone()]),
        m.map(|m| &m[vertices]),
    )?)
}

/// GEOS geometries of any kind, one per row,
/// to copy into one of our arrays with an `into_*_array` conversion.
pub struct GeosGeometries {
    pub geometries: Vec<GeosRow>,
    /// An upper bound, sized for the deepest kind, a multipolygon:
    /// rings (or lines), then polygons.
    pub size_hint: SizeHint<2>,
}

/// One row of [`GeosGeometries`].
pub enum GeosRow {
    Missing,
    /// Empty, without a GEOS geometry. Valid for every kind.
    Empty,
    Geometry(Geometry),
}

/// `None` is a missing row.
impl From<Option<Geometry>> for GeosRow {
    fn from(geometry: Option<Geometry>) -> Self {
        geometry.map_or(GeosRow::Missing, GeosRow::Geometry)
    }
}

/// For a conversion getting a geometry of a kind it can't hold:
/// `false` (its row is missing) if `ignore_per_row_errors=true`, otherwise an error.
fn wrong_kind(
    expected: &str,
    found: GeometryTypes,
    ignore_per_row_errors: bool,
) -> GeoComputeResult<bool> {
    if ignore_per_row_errors {
        return Ok(false);
    }
    Err(geos::Error::ConversionError(format!("expected a {expected}, got a {found:?}")).into())
}

/// Which rows are there, pushed one after the other.
/// A bitmap is only built once a row is missing.
struct ValidityBuilder {
    /// How many rows were pushed before the first missing one.
    valid: usize,
    rows: usize,
    bitmap: Option<MutableBitmap>,
}

impl ValidityBuilder {
    fn with_capacity(rows: usize) -> Self {
        Self {
            valid: 0,
            rows,
            bitmap: None,
        }
    }

    #[inline(always)]
    fn push(&mut self, is_valid: bool) {
        match &mut self.bitmap {
            Some(bitmap) => bitmap.push(is_valid),
            None if is_valid => self.valid += 1,
            None => {
                let mut bitmap = MutableBitmap::with_capacity(self.rows);
                bitmap.extend_constant(self.valid, true);
                bitmap.push(false);
                self.bitmap = Some(bitmap);
            }
        }
    }

    /// `None` when every row is there.
    fn finish(self) -> Option<Bitmap> {
        self.bitmap.map(Bitmap::from)
    }
}

/// The coordinates the conversions copy out of GEOS.
/// GEOS never gives m values.
struct CoordsBuilder {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Empty (and unallocated) without z.
    z: Vec<f64>,
    has_z: bool,
}

impl CoordsBuilder {
    /// With `has_z=true` the coordinates keep their z.
    fn with_capacity(has_z: bool, coords: usize) -> Self {
        Self {
            x: Vec::with_capacity(coords),
            y: Vec::with_capacity(coords),
            z: Vec::with_capacity(if has_z { coords } else { 0 }),
            has_z,
        }
    }

    fn len(&self) -> usize {
        self.x.len()
    }

    /// Appends the coordinates of `geometry`, a linestring or a ring.
    /// `Z` says whether the coordinates have z, which must match the builder.
    /// If `REVERSED=true`, they are appended in reverse order.
    fn push<const Z: bool, const REVERSED: bool>(
        &mut self,
        geometry: &impl Geom,
    ) -> GeoComputeResult<()> {
        debug_assert_eq!(Z, self.has_z, "pushed with the wrong z");
        let dim = if Z {
            GeoDimension::XYZ
        } else {
            GeoDimension::XY
        };
        let buffer = geometry.get_coord_seq()?.as_buffer(Some(coord_type(dim)))?;
        let width = dim.field_names().len();
        let push = |coord: &[f64]| {
            self.x.push(coord[0]);
            self.y.push(coord[1]);
            if Z {
                self.z.push(coord[2]);
            }
        };
        if REVERSED {
            buffer.chunks_exact(width).rev().for_each(push);
        } else {
            buffer.chunks_exact(width).for_each(push);
        }
        Ok(())
    }

    fn finish(self) -> ValidCoords {
        ValidCoords::from_axes((self.x, self.y, self.has_z.then_some(self.z), None))
    }
}

/// The offsets of one list level, built by ending one list after the other.
struct OffsetsBuilder {
    offsets: Vec<i64>,
}

impl OffsetsBuilder {
    fn with_capacity(lists: usize) -> Self {
        let mut offsets = Vec::with_capacity(lists + 1);
        offsets.push(0);
        Self { offsets }
    }

    /// How many lists were ended.
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Ends the current list at `end`: the length of the level below, so far.
    #[inline(always)]
    fn push(&mut self, end: usize) {
        debug_assert!(end as i64 >= *self.offsets.last().unwrap());
        self.offsets.push(end as i64);
    }

    fn finish(self) -> OffsetsBuffer<i64> {
        // SAFETY: they start at 0, and every end is the length of a level below that only grows.
        unsafe { Offsets::new_unchecked(self.offsets) }.into()
    }
}
