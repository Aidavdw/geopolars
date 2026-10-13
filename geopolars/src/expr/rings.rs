//! The rings of a polygon:
//! - `exterior` keeps its first ring, as a linestring,
//! - `interior` keeps the rest (its holes), as a multilinestring.
//!
//! A multipolygon gets a list of them, one per polygon.
//!
//! Arrow lists cannot skip entries, so the kept rings' coordinates are copied.
//! What one polygon keeps is contiguous in the coordinates,
//! so that is a single slice per field per polygon.
//! (Polars' own `take` on the rings goes ring by ring, at ~1.5 µs per ring.)

use std::ops::Range;

use polars::prelude::*;
use polars_arrow::array::{Array, ListArray, PrimitiveArray, StructArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::offset::Offsets;
use pyo3_polars::derive::polars_expr;

use crate::geoarrow::storage::downcast;
use crate::geoarrow::{describe, Geo, Kind};

/// The rings to keep of the polygons `rows`, as a part of the result per polygon.
type Part = fn(&ListArray<i64>, Range<usize>) -> PolarsResult<Box<dyn Array>>;

/// A `kind` per polygon, or a list of them per multipolygon,
/// with the input's dimension and metadata.
/// Also whether the input is a multipolygon.
fn rings_type(dtype: &DataType, name: &str, kind: Kind) -> PolarsResult<(DataType, bool)> {
    let geo = describe(dtype)?;
    let multi = match geo.kind {
        Kind::Polygon => false,
        Kind::MultiPolygon => true,
        _ => polars_bail!(
            SchemaMismatch: "{name} expects a polygon or multipolygon column, got: {}", dtype
        ),
    };
    let part = Geo::new(kind, geo.dim, geo.metadata.clone()).dtype();
    let dtype = if multi {
        DataType::List(Box::new(part))
    } else {
        part
    };
    Ok((dtype, multi))
}

/// For [`exterior`].
fn kernel_output_type_exterior(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = rings_type(input_fields[0].dtype(), "exterior", Kind::LineString)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

/// For [`interior`].
fn kernel_output_type_interior(input_fields: &[Field]) -> PolarsResult<Field> {
    let (dtype, _) = rings_type(input_fields[0].dtype(), "interior", Kind::MultiLineString)?;
    Ok(Field::new(input_fields[0].name().clone(), dtype))
}

fn validity(polygons: &ListArray<i64>, rows: &Range<usize>) -> Option<Bitmap> {
    polygons
        .validity()
        .map(|valid| valid.clone().sliced(rows.start, rows.len()))
}

/// The coordinates at `ranges`, in order, copied out of `coords` field by field.
/// Coordinates are never null in GeoArrow, so no validity is carried along.
pub(crate) fn copy_coords(
    coords: &StructArray,
    ranges: &[Range<usize>],
) -> PolarsResult<Box<dyn Array>> {
    let len = ranges.iter().map(Range::len).sum();
    let fields = coords.values().iter().map(|field| {
        let field: &PrimitiveArray<f64> = downcast(field.as_ref(), "f64 coordinates")?;
        let mut values = Vec::with_capacity(len);
        for range in ranges {
            values.extend_from_slice(&field.values()[range.clone()]);
        }
        Ok(PrimitiveArray::new(field.dtype().clone(), values.into(), None).boxed())
    });
    let fields = fields.collect::<PolarsResult<Vec<_>>>()?;
    Ok(StructArray::new(coords.dtype().clone(), len, fields, None).boxed())
}

fn rings_of(polygons: &ListArray<i64>) -> PolarsResult<(&ListArray<i64>, &StructArray)> {
    let rings = downcast::<ListArray<i64>>(polygons.values().as_ref(), "a list of rings")?;
    let coords = downcast::<StructArray>(rings.values().as_ref(), "a coordinate struct")?;
    Ok((rings, coords))
}

/// Where rings `rings` sit among the coordinates.
fn coordinates_of(rings: &ListArray<i64>, kept: Range<usize>) -> Range<usize> {
    let offsets = rings.offsets();
    offsets[kept.start] as usize..offsets[kept.end] as usize
}

/// the first ring of every polygon, as a linestring.
fn exterior_part(polygons: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<Box<dyn Array>> {
    let (rings, coords) = rings_of(polygons)?;
    let mut ranges = Vec::with_capacity(rows.len());
    for i in rows.clone() {
        let (start, end) = polygons.offsets().start_end(i);
        // A polygon without rings gets an empty linestring.
        let exterior = if polygons.is_valid(i) && end > start {
            start..start + 1
        } else {
            start..start
        };
        ranges.push(coordinates_of(rings, exterior));
    }
    Ok(ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(coords.dtype().clone()),
        Offsets::try_from_lengths(ranges.iter().map(Range::len))?.into(),
        copy_coords(coords, &ranges)?,
        validity(polygons, &rows),
    )
    .boxed())
}

/// every ring after the first, as a multilinestring.
fn interior_part(polygons: &ListArray<i64>, rows: Range<usize>) -> PolarsResult<Box<dyn Array>> {
    let (rings, coords) = rings_of(polygons)?;
    let (mut holes, mut counts) = (
        Vec::with_capacity(rows.len()),
        Vec::with_capacity(rows.len()),
    );
    for i in rows.clone() {
        let (start, end) = polygons.offsets().start_end(i);
        let kept = if polygons.is_valid(i) {
            (start + 1).min(end)..end
        } else {
            start..start
        };
        counts.push(kept.len());
        holes.push(kept);
    }
    let ring_lengths = holes
        .iter()
        .flat_map(|kept| kept.clone().map(|ring| rings.offsets().start_end(ring)))
        .map(|(start, end)| end - start);
    let ranges: Vec<_> = holes
        .iter()
        .map(|kept| coordinates_of(rings, kept.clone()))
        .collect();
    let kept_rings = ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(coords.dtype().clone()),
        Offsets::try_from_lengths(ring_lengths)?.into(),
        copy_coords(coords, &ranges)?,
        None,
    );
    Ok(ListArray::<i64>::new(
        ListArray::<i64>::default_datatype(kept_rings.dtype().clone()),
        Offsets::try_from_lengths(counts.into_iter())?.into(),
        kept_rings.boxed(),
        validity(polygons, &rows),
    )
    .boxed())
}

/// Apply `part` to every polygon. For a multipolygon (`multi`),
/// put the results back into one list per multipolygon.
fn per_polygon(
    inputs: &[Series],
    dtype: &DataType,
    multi: bool,
    part: Part,
) -> PolarsResult<Series> {
    let storage = inputs[0].ext()?.storage();
    let chunks = storage.list()?.downcast_iter().map(|chunk| {
        if !multi {
            return part(chunk, 0..chunk.len());
        }
        let polygons = downcast::<ListArray<i64>>(chunk.values().as_ref(), "a list of polygons")?;
        // Only the polygons this chunk points at: it may be a slice of a larger one.
        let (first, last) = (
            *chunk.offsets().first() as usize,
            *chunk.offsets().last() as usize,
        );
        // Only the multipolygon itself may be missing, not one of its parts.
        let parts = part(polygons, first..last)?.with_validity(None);
        Ok(ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(parts.dtype().clone()),
            Offsets::try_from_lengths(chunk.offsets().lengths())?.into(),
            parts,
            chunk.validity().cloned(),
        )
        .boxed())
    });
    let chunks = chunks.collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage `dtype` describes.
    Ok(unsafe { Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, dtype) })
}

/// See `exterior`.
#[polars_expr(output_type_func=kernel_output_type_exterior)]
fn exterior(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, multi) = rings_type(inputs[0].dtype(), "exterior", Kind::LineString)?;
    per_polygon(inputs, &dtype, multi, exterior_part)
}

/// See `interior`.
#[polars_expr(output_type_func=kernel_output_type_interior)]
fn interior(inputs: &[Series]) -> PolarsResult<Series> {
    let (dtype, multi) = rings_type(inputs[0].dtype(), "interior", Kind::MultiLineString)?;
    per_polygon(inputs, &dtype, multi, interior_part)
}
