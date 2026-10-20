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
use geopolars_core::{Coords, GeoDimension, List, Point, PointArray, ValidCoords};
use geos::{CoordSeq, CoordType, Geom, Geometry, GeometryTypes};
use polars_arrow::bitmap::{Bitmap, MutableBitmap};
use polars_arrow::offset::{Offsets, OffsetsBuffer};

/// Converts a single geometry into a GEOS geometry.
/// For an array, see [`ToGeosRows`].
pub trait ToGeos {
    type Geos;

    fn to_geos(&self) -> geos::GResult<Self::Geos>;
}

/// Converts an array into one GEOS geometry per row, with `None` where the row is missing.
/// The size hint is the size of the array itself.
/// If `ignore_per_row_errors=true`, a row GEOS fails to build is missing too;
/// otherwise its error is returned.
pub trait ToGeosRows {
    fn to_geos_rows(&self, ignore_per_row_errors: bool) -> geos::GResult<GeosGeometries>;
}

/// The geometry of one row, or `None` if building it failed and `ignore_per_row_errors=true`.
#[inline(always)]
fn row_or_missing<T>(
    built: geos::GResult<Option<T>>,
    ignore_per_row_errors: bool,
) -> geos::GResult<Option<T>> {
    match built {
        Err(_) if ignore_per_row_errors => Ok(None),
        built => built,
    }
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

impl ToGeosRows for PointArray {
    fn to_geos_rows(&self, ignore_per_row_errors: bool) -> geos::GResult<GeosGeometries> {
        // GEOS has no batch constructor, so only the per-column work is shared.
        let dim = self.coords.dim();
        let len = dim.field_names().len();
        let coord_type = coord_type(dim);
        let mut geometries = Vec::with_capacity(self.coords.len());
        for i in 0..self.coords.len() {
            let point = point_to_geos(&self.coords, i, len, coord_type);
            geometries.push(row_or_missing(point, ignore_per_row_errors)?);
        }
        Ok(GeosGeometries {
            geometries,
            size_hint: SizeHint {
                coords: self.coords.len(),
                parts: [0, 0],
            },
        })
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

/// [`ToGeosRows`] for the rows of a list array, whose size is `size_hint`.
/// `to_geos` builds one geometry from the range of a row's parts.
fn rows_to_geos(
    rows: &List,
    size_hint: SizeHint<2>,
    ignore_per_row_errors: bool,
    to_geos: impl Fn(Range<usize>) -> geos::GResult<Geometry>,
) -> geos::GResult<GeosGeometries> {
    let geometries = with_const_flags!(
        rows_to_geos_with(rows, ignore_per_row_errors, to_geos);
        rows.is_all_valid()
    )?;
    Ok(GeosGeometries {
        geometries,
        size_hint,
    })
}

/// See [`rows_to_geos`].
/// If `ALL_VALID=true`, no row is checked for being missing first.
/// Errors are the exception, so `ignore_per_row_errors` is checked per row rather than made const.
fn rows_to_geos_with<const ALL_VALID: bool>(
    rows: &List,
    ignore_per_row_errors: bool,
    to_geos: impl Fn(Range<usize>) -> geos::GResult<Geometry>,
) -> geos::GResult<Vec<Option<Geometry>>> {
    let mut geometries = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        // A missing row's offsets point anywhere, so it is not converted.
        geometries.push(if ALL_VALID || rows.is_valid(i) {
            row_or_missing(to_geos(rows.range(i)).map(Some), ignore_per_row_errors)?
        } else {
            None
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

/// GEOS geometries of any kind, one per row,
/// to copy into one of our arrays with an `into_*_array` conversion.
pub struct GeosGeometries {
    /// `None` gives a missing row.
    pub geometries: Vec<Option<Geometry>>,
    /// An upper bound, sized for the deepest kind, a multipolygon:
    /// rings (or lines), then polygons.
    pub size_hint: SizeHint<2>,
}

/// For a conversion getting a geometry of a kind it can't hold:
/// `false` (its row is missing) if `ignore_per_row_errors=true`, otherwise an error.
fn wrong_kind(
    expected: &str,
    found: GeometryTypes,
    ignore_per_row_errors: bool,
) -> geos::GResult<bool> {
    if ignore_per_row_errors {
        return Ok(false);
    }
    Err(geos::Error::ConversionError(format!(
        "expected a {expected}, got a {found:?}"
    )))
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
    z: Option<Vec<f64>>,
}

impl CoordsBuilder {
    /// With `has_z=true` the coordinates keep their z.
    fn with_capacity(has_z: bool, coords: usize) -> Self {
        Self {
            x: Vec::with_capacity(coords),
            y: Vec::with_capacity(coords),
            z: has_z.then(|| Vec::with_capacity(coords)),
        }
    }

    fn len(&self) -> usize {
        self.x.len()
    }

    /// Appends the coordinates of `geometry`, a linestring or a ring.
    fn push(&mut self, geometry: &impl Geom) -> geos::GResult<()> {
        let dim = match self.z {
            Some(_) => GeoDimension::XYZ,
            None => GeoDimension::XY,
        };
        let buffer = geometry.get_coord_seq()?.as_buffer(Some(coord_type(dim)))?;
        let width = dim.field_names().len();
        // Once per linestring or ring, rather than per coordinate.
        match &mut self.z {
            Some(z) => buffer.chunks_exact(width).for_each(|coord| {
                self.x.push(coord[0]);
                self.y.push(coord[1]);
                z.push(coord[2]);
            }),
            None => buffer.chunks_exact(width).for_each(|coord| {
                self.x.push(coord[0]);
                self.y.push(coord[1]);
            }),
        }
        Ok(())
    }

    fn finish(self) -> ValidCoords {
        ValidCoords::from_axes((self.x, self.y, self.z, None))
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
