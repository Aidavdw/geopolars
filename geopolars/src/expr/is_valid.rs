//! Whether a geometry is valid
//! See `is_valid` in Python for the rules.
//!
//! Every geometry has its own kernel.
//!
//! Because we allow so many flags, there are a lot of different paths that can be taken.
//! Switching in the hot-path adds up, so this is done using const generics.
//! Inlining all is ≈  1.5MB more in release build,
//! but with up to 2x perf increase for simple geometries.
//!
//! [are_valid] can also be used by other modules

use std::mem;
use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, BooleanArray, ListArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use pyo3_polars::derive::polars_expr;
use rsgeo::coordinate_position::CoordPos;
use rsgeo::dimensions::Dimensions;
use rsgeo::relate::IntersectionMatrix;
use rsgeo::{Coord, LineString, Polygon, PreparedGeometry, Relate};
use serde::Deserialize;

use super::coords::same_geometry;
use super::self_intersects::vertices_self_intersect;
use crate::geoarrow::storage::{downcast, same_position, CoordsView};
use crate::geoarrow::{describe, Kind};

/// Where `x` and `y` may lie, as `[xmin, ymin, xmax, ymax]`, ends included.
type Bounds = [f64; 4];

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IsValidKwargs {
    /// `None` when there are none, or when they are not to be checked.
    bounds: Option<Bounds>,
    /// Whether every coordinate has to be there, with every axis.
    check_missing_coordinates: bool,
    /// Whether every axis of a coordinate that is there has to be finite.
    check_finite_coordinates: bool,
    /// Whether no two vertices in a row of a line or ring may lie at the same place.
    check_line_string_zero_length_sections: bool,
    /// Whether a polygon's rings may not cross themselves.
    check_ring_self_intersect: bool,
    /// Whether a polygon's rings have to [fit](polygon_rings_fit) together.
    check_polygon_self_overlap: bool,
    /// Whether a multipolygon's parts have to [fit](multi_polygon_parts_fit) together.
    check_multi_polygon_self_overlap: bool,
}

impl Default for IsValidKwargs {
    fn default() -> Self {
        Self {
            bounds: None,
            check_missing_coordinates: true,
            check_finite_coordinates: true,
            check_line_string_zero_length_sections: true,
            check_ring_self_intersect: true,
            check_polygon_self_overlap: true,
            check_multi_polygon_self_overlap: true,
        }
    }
}

/// The coordinates of one chunk, and the bounds they have to lie in.
struct Coords<'a> {
    view: CoordsView<'a>,
    /// Only read with `BOUNDS`.
    bounds: Bounds,
}

/// Room for the rings of a polygon, kept between rows so that they are allocated once.
#[derive(Default)]
struct Scratch {
    /// The x and y of the ring being checked.
    xy: Vec<Coord<f64>>,
    /// The rings checked so far, each as the polygon it encloses,
    /// when they have to be related to each other.
    rings: Vec<Polygon<f64>>,
}

impl<'a> Coords<'a> {
    fn new(coords: &'a dyn Array, kwargs: &IsValidKwargs) -> PolarsResult<Self> {
        Ok(Self {
            view: CoordsView::new(coords)?,
            bounds: kwargs.bounds.unwrap_or_default(),
        })
    }

    /// The runtime flags, in order:
    /// `NULLS`, `MISSING`, `FINITE`, and `BOUNDS`.
    fn coordinate_check_flags(&self, kwargs: &IsValidKwargs) -> (bool, bool, bool, bool) {
        let nulls = self.view.has_missing_values();
        (
            nulls,
            nulls && kwargs.check_missing_coordinates,
            kwargs.check_finite_coordinates,
            kwargs.bounds.is_some(),
        )
    }

    /// Whether coordinate `i` passes the coordinate checks.
    /// A missing coordinate has no values to check.
    #[inline(always)]
    fn coordinate_is_valid<
        const NULLS: bool,
        const MISSING: bool,
        const FINITE: bool,
        const BOUNDS: bool,
    >(
        &self,
        i: usize,
    ) -> bool {
        if NULLS && !self.view.present(i) {
            return !MISSING;
        }
        (!FINITE || self.view.values(i).all(f64::is_finite))
            && (!BOUNDS || {
                let [xmin, ymin, xmax, ymax] = self.bounds;
                let (x, y) = self.view.xy_checking_nulls::<false>(i).expect("present");
                (xmin..=xmax).contains(&x) && (ymin..=ymax).contains(&y)
            })
    }

    fn is_valid_or_empty_point<
        const NULLS: bool,
        const MISSING: bool,
        const FINITE: bool,
        const BOUNDS: bool,
    >(
        &self,
        i: usize,
    ) -> bool {
        self.coordinate_is_valid::<NULLS, MISSING, FINITE, BOUNDS>(i)
            || ((!NULLS || self.view.present(i)) && self.view.values(i).all(f64::is_nan))
    }

    /// Whether every one of `vertices` is valid, in one pass that stops at the first that is not.
    /// Each has to pass `coordinate_is_valid`.
    /// If `ZERO=true`, none may lie at the same position as the one before;
    /// If `XY=true`, each needs a finite x and y (which are pushed onto `xy`).
    #[inline(always)]
    fn vertices_are_valid<
        const NULLS: bool,
        const MISSING: bool,
        const FINITE: bool,
        const BOUNDS: bool,
        const ZERO: bool,
        const XY: bool,
    >(
        &self,
        vertices: Range<usize>,
        xy: &mut Vec<Coord<f64>>,
    ) -> bool {
        // Where the vertex before lies, `None` if it is missing (which is close to nothing).
        let mut before = None;
        for i in vertices {
            if !self.coordinate_is_valid::<NULLS, MISSING, FINITE, BOUNDS>(i) {
                return false;
            }
            if ZERO {
                let here = self.view.position::<NULLS>(i);
                if let (Some(before), Some(here)) = (before, here) {
                    if same_position(before, here) {
                        return false;
                    }
                }
                before = here;
            }
            if XY {
                // `coordinate_is_valid` already found it finite, unless it let it through as missing.
                match self.view.xy_checking_nulls::<NULLS>(i) {
                    Some((x, y))
                        if (FINITE && (!NULLS || MISSING)) || (x.is_finite() && y.is_finite()) =>
                    {
                        xy.push(Coord { x, y });
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    /// Whether `vertices` make a valid linestring
    /// no vertices, or at least two valid ones.
    fn line_string_is_valid<
        const NULLS: bool,
        const MISSING: bool,
        const FINITE: bool,
        const BOUNDS: bool,
        const ZERO: bool,
    >(
        &self,
        vertices: Range<usize>,
    ) -> bool {
        vertices.len() != 1
            && self.vertices_are_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO, false>(
                vertices,
                &mut Vec::new(),
            )
    }

    /// Whether the rings `polygon` (indices into `rings`) make a valid polygon:
    /// all rings empty OR
    /// - all rings valid linestrings (if `RING`)
    /// - all rings don't overlap (if `OVERLAP`)
    /// With `KEEP`, its rings are left in `scratch.rings` (none for the empty polygon).
    fn polygon_is_valid<
        const NULLS: bool,
        const MISSING: bool,
        const FINITE: bool,
        const BOUNDS: bool,
        const ZERO: bool,
        const RING: bool,
        const OVERLAP: bool,
        const KEEP: bool,
    >(
        &self,
        rings: &ListArray<i64>,
        polygon: Range<usize>,
        scratch: &mut Scratch,
    ) -> bool {
        scratch.rings.clear();
        let mut of_polygon = polygon.map(|ring| list_item_range(rings, ring));
        // An empty exterior makes the empty polygon, so every other ring has to be empty too.
        if of_polygon
            .clone()
            .next()
            .is_some_and(|exterior| exterior.is_some_and(|exterior| exterior.is_empty()))
        {
            return of_polygon.all(|ring| ring.is_some_and(|ring| ring.is_empty()));
        }
        of_polygon.all(|ring| {
            ring.is_some_and(|ring| {
                self.ring_is_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO, RING, OVERLAP, KEEP>(
                    ring, scratch,
                )
            })
        }) && (!OVERLAP || polygon_rings_fit(&scratch.rings))
    }

    /// Whether `vertices` make a valid ring
    /// - must close (final vert = first vert)
    /// - min 4 points (triangle)
    #[inline(always)]
    fn ring_is_valid<
        const NULLS: bool,
        const MISSING: bool,
        const FINITE: bool,
        const BOUNDS: bool,
        const ZERO: bool,
        const RING: bool,
        const OVERLAP: bool,
        const KEEP: bool,
    >(
        &self,
        vertices: Range<usize>,
        scratch: &mut Scratch,
    ) -> bool {
        if !self.view.is_ring(vertices.clone()) {
            return false;
        }
        // Without topology, the ring need not be read into vertices at all.
        if !(RING || OVERLAP || KEEP) {
            return self.vertices_are_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO, false>(
                vertices,
                &mut scratch.xy,
            );
        }
        scratch.xy.clear();
        scratch.xy.reserve(vertices.len());
        if !self.vertices_are_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO, true>(
            vertices,
            &mut scratch.xy,
        ) {
            return false;
        }
        if RING && vertices_self_intersect(&scratch.xy) {
            return false;
        }
        // Handed to rsgeo, which needs them owned: the only copy made of the vertices.
        if OVERLAP || KEEP {
            let ring = LineString::new(mem::take(&mut scratch.xy));
            scratch.rings.push(Polygon::new(ring, vec![]));
        }
        true
    }
}

/// Whether the two geometries `matrix` relates touch only at points, never along a line.
fn touch_only_at_points(matrix: &IntersectionMatrix) -> bool {
    matrix.get(CoordPos::OnBoundary, CoordPos::OnBoundary) != Dimensions::OneDimensional
}

/// Whether a polygon's rings fit together, each given as the polygon it encloses.
/// First is exterior, others is holes.
/// See python for rules.
fn polygon_rings_fit(rings: &[Polygon<f64>]) -> bool {
    // Re-use matrix, as getting that is the expensive part.
    let [exterior, holes @ ..] = rings else {
        return true;
    };
    if holes.is_empty() {
        return true;
    }
    let exterior = PreparedGeometry::from(exterior);
    let holes: Vec<_> = holes.iter().map(PreparedGeometry::from).collect();
    holes.iter().enumerate().all(|(i, hole)| {
        let matrix = exterior.relate(hole);
        matrix.is_contains()
            && touch_only_at_points(&matrix)
            && holes[i + 1..].iter().all(|other| {
                let matrix = hole.relate(other);
                matrix.get(CoordPos::Inside, CoordPos::Inside) == Dimensions::Empty
                    && touch_only_at_points(&matrix)
            })
    })
}

/// Whether a multipolygon's parts fit together:
/// - each a valid polygon that is not empty
/// - no two parts share any area
/// - parts only allowed to touch at points
fn multi_polygon_parts_fit(parts: &[Polygon<f64>]) -> bool {
    let prepared: Vec<_> = parts.iter().map(PreparedGeometry::from).collect();
    prepared.iter().enumerate().all(|(i, part)| {
        prepared[i + 1..].iter().all(|other| {
            let matrix = part.relate(other);
            matrix.get(CoordPos::Inside, CoordPos::Inside) == Dimensions::Empty
                && touch_only_at_points(&matrix)
        })
    })
}

/// The rings a [`Coords::polygon_is_valid`] kept, as one polygon. `None` for the empty polygon.
fn rings_into_polygon(rings: &mut Vec<Polygon<f64>>) -> Option<Polygon<f64>> {
    let mut rings = rings.drain(..).map(|ring| ring.into_inner().0);
    let exterior = rings.next()?;
    Some(Polygon::new(exterior, rings.collect()))
}

fn list_item_range(list: &ListArray<i64>, i: usize) -> Option<Range<usize>> {
    list.is_valid(i).then(|| {
        let (start, end) = list.offsets().start_end(i);
        start..end
    })
}

/// Whether each of `len` rows is valid, as `check` says.
fn is_valid_per_row(
    len: usize,
    validity: Option<&Bitmap>,
    mut check: impl FnMut(usize) -> bool,
) -> BooleanArray {
    let values = (0..len).map(|i| validity.is_none_or(|valid| valid.get_bit(i)) && check(i));
    BooleanArray::new(
        ArrowDataType::Boolean,
        Bitmap::from_iter(values),
        validity.cloned(),
    )
}

/// [`is_valid_per_row`], with `check` given the range of each row's parts.
fn is_valid_per_list_row(
    rows: &ListArray<i64>,
    mut check: impl FnMut(Range<usize>) -> bool,
) -> BooleanArray {
    let offsets = rows.offsets();
    is_valid_per_row(rows.len(), rows.validity(), |i| {
        let (start, end) = offsets.start_end(i);
        check(start..end)
    })
}

fn points_are_valid<
    const NULLS: bool,
    const MISSING: bool,
    const FINITE: bool,
    const BOUNDS: bool,
>(
    rows: &dyn Array,
    coords: &Coords,
) -> BooleanArray {
    is_valid_per_row(rows.len(), rows.validity(), |i| {
        coords.is_valid_or_empty_point::<NULLS, MISSING, FINITE, BOUNDS>(i)
    })
}

fn multi_points_are_valid<
    const NULLS: bool,
    const MISSING: bool,
    const FINITE: bool,
    const BOUNDS: bool,
>(
    rows: &ListArray<i64>,
    coords: &Coords,
) -> BooleanArray {
    is_valid_per_list_row(rows, |mut points| {
        points.all(|i| coords.is_valid_or_empty_point::<NULLS, MISSING, FINITE, BOUNDS>(i))
    })
}

fn line_strings_are_valid<
    const NULLS: bool,
    const MISSING: bool,
    const FINITE: bool,
    const BOUNDS: bool,
    const ZERO: bool,
>(
    rows: &ListArray<i64>,
    coords: &Coords,
) -> BooleanArray {
    is_valid_per_list_row(rows, |line| {
        coords.line_string_is_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO>(line)
    })
}

fn multi_line_strings_are_valid<
    const NULLS: bool,
    const MISSING: bool,
    const FINITE: bool,
    const BOUNDS: bool,
    const ZERO: bool,
>(
    rows: &ListArray<i64>,
    lines: &ListArray<i64>,
    coords: &Coords,
) -> BooleanArray {
    is_valid_per_list_row(rows, |mut of_row| {
        of_row.all(|line| {
            list_item_range(lines, line).is_some_and(|line| {
                coords.line_string_is_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO>(line)
            })
        })
    })
}

fn polygons_are_valid<
    const NULLS: bool,
    const MISSING: bool,
    const FINITE: bool,
    const BOUNDS: bool,
    const ZERO: bool,
    const RING: bool,
    const OVERLAP: bool,
>(
    rows: &ListArray<i64>,
    rings: &ListArray<i64>,
    coords: &Coords,
) -> BooleanArray {
    let mut scratch = Scratch::default();
    is_valid_per_list_row(rows, |polygon| {
        coords.polygon_is_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO, RING, OVERLAP, false>(
            rings,
            polygon,
            &mut scratch,
        )
    })
}

/// Requires:
/// - Every part a valid polygon,
/// - [multi_polygon_parts_fit] for all parts if `MULTI=true`
#[allow(clippy::too_many_arguments)]
fn multi_polygons_are_valid<
    const NULLS: bool,
    const MISSING: bool,
    const FINITE: bool,
    const BOUNDS: bool,
    const ZERO: bool,
    const RING: bool,
    const OVERLAP: bool,
    const MULTI: bool,
>(
    rows: &ListArray<i64>,
    polygons: &ListArray<i64>,
    rings: &ListArray<i64>,
    coords: &Coords,
) -> BooleanArray {
    let mut scratch = Scratch::default();
    // The parts of the row being checked, kept between rows so that they are allocated once.
    let mut parts = Vec::new();
    is_valid_per_list_row(rows, |of_row| {
        parts.clear();
        for polygon in of_row {
            let valid = list_item_range(polygons, polygon).is_some_and(|polygon| {
                coords
                    .polygon_is_valid::<NULLS, MISSING, FINITE, BOUNDS, ZERO, RING, OVERLAP, MULTI>(
                        rings,
                        polygon,
                        &mut scratch,
                    )
            });
            if !valid {
                return false;
            }
            if MULTI {
                parts.extend(rings_into_polygon(&mut scratch.rings));
            }
        }
        !MULTI || multi_polygon_parts_fit(&parts)
    })
}

/// [`are_valid`] for one chunk of the storage of a `kind`.
///
/// The flags every kernel takes first (through [`with_const_flags!`]), in this order:
/// - `NULLS`: whether the chunk has a missing coordinate or axis at all ([`CoordsView::has_missing_values`]).
/// - `MISSING`: whether a missing coordinate is invalid (only read with `NULLS`).
/// - `FINITE`: whether every axis has to be finite.
/// - `BOUNDS`: whether x and y have to lie inside [`Coords::bounds`].
///
/// Then, where the geometry has them,
/// `ZERO` (no two vertices in a row at the same place), `RING` (no ring crosses itself),
/// `OVERLAP` (a polygon's rings fit) and `MULTI` (a multipolygon's parts fit).
fn chunk_is_valid(
    chunk: &dyn Array,
    kind: Kind,
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanArray> {
    let IsValidKwargs {
        check_line_string_zero_length_sections: zero,
        check_ring_self_intersect: ring,
        check_polygon_self_overlap: overlap,
        check_multi_polygon_self_overlap: multi,
        ..
    } = *kwargs;
    if kind == Kind::Point {
        let coords = Coords::new(chunk, kwargs)?;
        let (nulls, missing, finite, bounds) = coords.coordinate_check_flags(kwargs);
        return Ok(with_const_flags!(
            points_are_valid(chunk, &coords);
            nulls, missing, finite, bounds
        ));
    }
    let rows: &ListArray<i64> = downcast(chunk, "a list of parts")?;
    let parts = rows.values().as_ref();
    Ok(match kind {
        Kind::Point => unreachable!("handled above"),
        Kind::LineString => {
            let coords = Coords::new(parts, kwargs)?;
            let (nulls, missing, finite, bounds) = coords.coordinate_check_flags(kwargs);
            with_const_flags!(
                line_strings_are_valid(rows, &coords);
                nulls, missing, finite, bounds, zero
            )
        }
        Kind::MultiPoint => {
            let coords = Coords::new(parts, kwargs)?;
            let (nulls, missing, finite, bounds) = coords.coordinate_check_flags(kwargs);
            with_const_flags!(
                multi_points_are_valid(rows, &coords);
                nulls, missing, finite, bounds
            )
        }
        Kind::Polygon => {
            let rings: &ListArray<i64> = downcast(parts, "a list of rings")?;
            let coords = Coords::new(rings.values().as_ref(), kwargs)?;
            let (nulls, missing, finite, bounds) = coords.coordinate_check_flags(kwargs);
            with_const_flags!(
                polygons_are_valid(rows, rings, &coords);
                nulls, missing, finite, bounds, zero, ring, overlap
            )
        }
        Kind::MultiLineString => {
            let lines: &ListArray<i64> = downcast(parts, "a list of linestrings")?;
            let coords = Coords::new(lines.values().as_ref(), kwargs)?;
            let (nulls, missing, finite, bounds) = coords.coordinate_check_flags(kwargs);
            with_const_flags!(
                multi_line_strings_are_valid(rows, lines, &coords);
                nulls, missing, finite, bounds, zero
            )
        }
        Kind::MultiPolygon => {
            let polygons: &ListArray<i64> = downcast(parts, "a list of polygons")?;
            let rings: &ListArray<i64> = downcast(polygons.values().as_ref(), "a list of rings")?;
            let coords = Coords::new(rings.values().as_ref(), kwargs)?;
            let (nulls, missing, finite, bounds) = coords.coordinate_check_flags(kwargs);
            with_const_flags!(
                multi_polygons_are_valid(rows, polygons, rings, &coords);
                nulls, missing, finite, bounds, zero, ring, overlap, multi
            )
        }
    })
}

/// Whether each geometry in `geometries` is valid, missing where the geometry is.
/// One chunk out for every chunk in.
pub(crate) fn are_valid(
    geometries: &Series,
    kwargs: &IsValidKwargs,
) -> PolarsResult<BooleanChunked> {
    let kind = describe(geometries.dtype())?.kind;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_is_valid(chunk.as_ref(), kind, kwargs));
    BooleanChunked::try_from_chunk_iter(geometries.name().clone(), chunks)
}

/// `output_type_func` for [`is_valid`]: a `bool` per geometry.
fn is_valid_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(Field::new(field.name().clone(), DataType::Boolean))
}

/// See `is_valid`.
#[polars_expr(output_type_func=is_valid_type)]
fn is_valid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    Ok(are_valid(&inputs[0], &kwargs)?.into_series())
}

/// See `null_out_invalid`.
#[polars_expr(output_type_func=same_geometry)]
fn null_out_invalid(inputs: &[Series], kwargs: IsValidKwargs) -> PolarsResult<Series> {
    let geometries = &inputs[0];
    let valid = are_valid(geometries, &kwargs)?;
    // Missing rows do not count, and stay missing.
    if valid.all() {
        return Ok(geometries.clone());
    }

    let geo = describe(geometries.dtype())?;
    let storage = geometries.ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .zip(valid.downcast_iter())
        .map(|(chunk, valid)| {
            // A missing row is `false` in `valid`'s values, so they alone say what stays.
            chunk.with_validity(Some(valid.values().clone()))
        })
        .collect();
    // SAFETY: only the outer validity changed, so every chunk is still the storage.
    let storage = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };
    Ok(storage.into_extension(geo.typ.clone()))
}
