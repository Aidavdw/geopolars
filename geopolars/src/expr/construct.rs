//! Building a geometry out of the parts it is made of.

use std::sync::Arc;

use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::close_rings::CloseRingsKwargs;
use super::coords::trimmed;
use crate::geoarrow::crs::{ExtensionMetadata, MetadataKwargs};
use crate::geoarrow::{coord, describe, Geo, GeoDimension, Kind};
use geopolars_compute::inner_nulls::{
    propagate_inner_nulls_of_line_strings, propagate_inner_nulls_of_multi_line_strings,
    propagate_inner_nulls_of_multi_points, propagate_inner_nulls_of_multi_polygons,
    propagate_inner_nulls_of_polygons,
};
use geopolars_compute::native::{
    close_open_rings, multi_polygons_have_only_rings, polygons_have_only_rings,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{UncheckedStorage, ValidCoords, ValidList};

/// The keyword arguments of the constructors that build polygons:
/// [`MetadataKwargs`], plus whether and how to close open rings.
/// These have to be mirrored on the python side.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolygonKwargs {
    /// The CRS the coordinates are in, in any form PROJ accepts.
    pub crs: Option<String>,
    /// How to close every open ring, as `close_rings` does. `None` closes nothing.
    pub close: Option<CloseRingsKwargs>,
}

impl PolygonKwargs {
    fn metadata(&self) -> MetadataKwargs {
        MetadataKwargs {
            crs: self.crs.clone(),
        }
    }
}

/// Used in error messages, don't care about allocation
fn shape_of(part: Kind) -> String {
    let lists = "lists of ".repeat(part.nesting() as usize);
    format!("{lists}coordinate structs")
}

/// Used in error messages, don't care about allocation
fn coordinate_shape_of(kind: Kind) -> String {
    let lists = "lists of ".repeat(kind.nesting() as usize);
    format!("{lists}f64")
}

/// The dimension and metadata of the parts a geometry can be gathered from:
/// a list of `part`s, each either a `part` geometry already or the bare storage
/// one of those wraps.
fn parts_of(dtype: &DataType, part: Kind) -> PolarsResult<(GeoDimension, Arc<ExtensionMetadata>)> {
    let expected = format!(
        "expected a list of `{}`s or of {}",
        part.name(),
        shape_of(part)
    );

    let DataType::List(inner) = dtype else {
        polars_bail!(SchemaMismatch: "{}, got: {}", expected, dtype);
    };

    if matches!(inner.as_ref(), DataType::Extension(..)) {
        match describe(inner) {
            Ok(geo) if geo.kind == part => {
                // metadata should be carried over
                return Ok((geo.dim, geo.metadata.clone()));
            }
            _ => polars_bail!(
                SchemaMismatch: "{} with separated x/y[/z][/m] coordinates, got: {}",
                expected, dtype
            ),
        }
    }

    let Some(dim) = coord::dimension_of_storage(inner, part.nesting()) else {
        polars_bail!(SchemaMismatch: "{}, got: {}", expected, dtype);
    };
    Ok((dim, Default::default()))
}

/// Small helper
fn gathered(
    dtype: &DataType,
    kwargs: &MetadataKwargs,
    kind: Kind,
    part: Kind,
) -> PolarsResult<Geo> {
    let (dim, metadata) = parts_of(dtype, part)?;
    Ok(Geo::new(kind, dim, kwargs.apply(metadata)?))
}

/// For gathering a list of `part`s into a `kind`.
fn kernel_output_type_wrapper_gathered(
    input_fields: &[Field],
    kwargs: &MetadataKwargs,
    kind: Kind,
    part: Kind,
) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = gathered(field.dtype(), kwargs, kind, part)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

fn kernel_output_type_linestring(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(input_fields, &kwargs, Kind::LineString, Kind::Point)
}

fn kernel_output_type_polygon(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(
        input_fields,
        &kwargs.metadata(),
        Kind::Polygon,
        Kind::LineString,
    )
}

fn kernel_output_type_multipoint(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(input_fields, &kwargs, Kind::MultiPoint, Kind::Point)
}

fn kernel_output_type_multilinestring(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(
        input_fields,
        &kwargs,
        Kind::MultiLineString,
        Kind::LineString,
    )
}

fn kernel_output_type_multipolygon(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_gathered(
        input_fields,
        &kwargs.metadata(),
        Kind::MultiPolygon,
        Kind::Polygon,
    )
}

/// See [`close_open_rings`].
/// The rings and their coordinates, or `None` if no ring changed.
pub(super) fn closed_rings(
    rings: &ValidList,
    coords: &ValidCoords,
    closing: CloseRingsKwargs,
) -> Option<(ValidList, ValidCoords)> {
    let CloseRingsKwargs { snap_z, snap_m } = closing;
    let z = coords.dim().has_z();
    let closed = with_const_flags!(close_open_rings(rings, coords, snap_z, snap_m); z)?;
    Some((closed.rows.ignoring_validity().clone(), closed.coords))
}

/// One chunk of the storage of a `kind`, with every missing value below a row moved out to it.
/// A polygon is also missing where it does not have only rings (see [polygons_have_only_rings]).
/// With a `close`, the open rings of a polygon are closed first.
fn validated_chunk(
    chunk: &dyn Array,
    kind: Kind,
    close: Option<CloseRingsKwargs>,
) -> PolarsResult<Box<dyn Array>> {
    let storage = UncheckedStorage::new(chunk)?;
    Ok(match kind {
        Kind::Point => unreachable!("constructors always gather into lists"),
        Kind::LineString => propagate_inner_nulls_of_line_strings(storage)
            .into_list_array()
            .boxed(),
        Kind::MultiPoint => propagate_inner_nulls_of_multi_points(storage)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => propagate_inner_nulls_of_multi_line_strings(storage)
            .into_list_array()
            .boxed(),
        Kind::Polygon => {
            let mut array = propagate_inner_nulls_of_polygons(storage);
            if let Some((rings, coords)) =
                close.and_then(|closing| closed_rings(&array.rings, &array.coords, closing))
            {
                (array.rings, array.coords) = (rings, coords);
            }
            // A missing row is `false` in the values too.
            let z = array.coords.dim().has_z();
            let valid = with_const_flags!(polygons_have_only_rings(&array); z)
                .values()
                .clone();
            array.rows = array.rows.with_validity(Some(valid));
            array.into_list_array().boxed()
        }
        Kind::MultiPolygon => {
            let mut array = propagate_inner_nulls_of_multi_polygons(storage);
            if let Some((rings, coords)) =
                close.and_then(|closing| closed_rings(&array.rings, &array.coords, closing))
            {
                (array.rings, array.coords) = (rings, coords);
            }
            // A missing row is `false` in the values too.
            let z = array.coords.dim().has_z();
            let valid = with_const_flags!(multi_polygons_have_only_rings(&array); z)
                .values()
                .clone();
            array.rows = array.rows.with_validity(Some(valid));
            array.into_list_array().boxed()
        }
    })
}

/// [validated_chunk] for every chunk of `storage`.
fn validated(storage: Series, kind: Kind, close: Option<CloseRingsKwargs>) -> PolarsResult<Series> {
    let storage = trimmed(&storage);
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| validated_chunk(chunk.as_ref(), kind, close))
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: the kind and dimension stay the same, so every chunk is still the storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    })
}

/// Gather a list of `part`s into one geometry of `kind` per row.
/// `close` says how to close open rings, and only means something for kinds that have them.
fn gather(
    parts: &Series,
    kwargs: &MetadataKwargs,
    close: Option<CloseRingsKwargs>,
    kind: Kind,
    part: Kind,
) -> PolarsResult<Series> {
    let geo = gathered(parts.dtype(), kwargs, kind, part)?;
    // This only changes dtype, nothing on the data.
    let storage = parts.list()?.apply_to_inner(&|part| {
        Ok(match part.dtype() {
            DataType::Extension(_, _) => part.ext()?.storage().clone(),
            _ => part,
        })
    })?;

    Ok(validated(storage.into_series(), kind, close)?.into_extension(geo.instance()))
}

/// See `linestring_from_vertices`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_linestring)]
fn linestring(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    gather(&inputs[0], &kwargs, None, Kind::LineString, Kind::Point)
}

/// See `polygon_from_rings`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon)]
fn polygon(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs.metadata(),
        kwargs.close,
        Kind::Polygon,
        Kind::LineString,
    )
}

/// See `multipoint_from_points`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipoint)]
fn multipoint(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    // Note that this is basically the same as linestring, but just a different [Kind].
    gather(&inputs[0], &kwargs, None, Kind::MultiPoint, Kind::Point)
}

/// See `multilinestring_from_linestrings`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multilinestring)]
fn multilinestring(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs,
        None,
        Kind::MultiLineString,
        Kind::LineString,
    )
}

/// See `multipolygon_from_polygons`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon)]
fn multipolygon(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    gather(
        &inputs[0],
        &kwargs.metadata(),
        kwargs.close,
        Kind::MultiPolygon,
        Kind::Polygon,
    )
}

/// The dimension a set of separate coordinate columns spells out.
fn coordinates_of(input_fields: &[Field], kind: Kind) -> PolarsResult<GeoDimension> {
    let mut coordinates = Vec::with_capacity(input_fields.len());
    for field in input_fields {
        let mut dtype = field.dtype();
        for _ in 0..kind.nesting() {
            let DataType::List(inner) = dtype else {
                polars_bail!(
                    SchemaMismatch: "expected `{}` to hold {}, one per {}, got: {}",
                    field.name(), coordinate_shape_of(kind), kind.display(), field.dtype()
                );
            };
            dtype = inner;
        }
        // The values themselves are cast, but only from something that is
        // already a number: parsing coordinates out of strings is not our job.
        polars_ensure!(
            dtype.is_primitive_numeric(),
            SchemaMismatch: "expected `{}` to hold {}, got: {}",
            field.name(), coordinate_shape_of(kind), field.dtype()
        );
        coordinates.push(Field::new(field.name().clone(), DataType::Float64));
    }

    coord::dimension_of(&DataType::Struct(coordinates)).ok_or_else(|| {
        polars_err!(
            SchemaMismatch: "expected coordinate columns named x, y[, z][, m], in that order"
        )
    })
}

/// Small helper
fn zipped(input_fields: &[Field], kwargs: &MetadataKwargs, kind: Kind) -> PolarsResult<Geo> {
    let dim = coordinates_of(input_fields, kind)?;
    Ok(Geo::new(kind, dim, kwargs.apply(Default::default())?))
}

/// For zipping coordinate columns into a `kind`.
fn kernel_output_type_wrapper_zipped(
    input_fields: &[Field],
    kwargs: &MetadataKwargs,
    kind: Kind,
) -> PolarsResult<Field> {
    let geo = zipped(input_fields, kwargs, kind)?;
    Ok(Field::new(input_fields[0].name().clone(), geo.dtype()))
}

fn kernel_output_type_linestring_coords(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs, Kind::LineString)
}

fn kernel_output_type_polygon_coords(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs.metadata(), Kind::Polygon)
}

fn kernel_output_type_multipoint_coords(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs, Kind::MultiPoint)
}

fn kernel_output_type_multilinestring_coords(
    input_fields: &[Field],
    kwargs: MetadataKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs, Kind::MultiLineString)
}

fn kernel_output_type_multipolygon_coords(
    input_fields: &[Field],
    kwargs: PolygonKwargs,
) -> PolarsResult<Field> {
    kernel_output_type_wrapper_zipped(input_fields, &kwargs.metadata(), Kind::MultiPolygon)
}

/// Interleave columns that each nest their coordinate `nesting` `List` layers
/// deep into one `List`-of-that-depth of coordinate structs.
///
/// Only the first column's offsets survive; the callers checks the rest carry the same ones.
fn interleave(columns: &[Series], nesting: u8) -> PolarsResult<Series> {
    if nesting == 0 {
        let coordinates = columns
            .iter()
            .map(|values| values.cast(&DataType::Float64))
            .collect::<PolarsResult<Vec<Series>>>()?;
        let name = columns[0].name().clone();
        let len = columns[0].len();
        return Ok(StructChunked::from_series(name, len, coordinates.iter())?.into_series());
    }

    let inner = columns
        .iter()
        .map(|column| Ok(column.list()?.get_inner()))
        .collect::<PolarsResult<Vec<Series>>>()?;
    Ok(columns[0]
        .list()?
        .with_inner_values(&interleave(&inner, nesting - 1)?)
        .into_series())
}

/// Zip one coordinate column per axis into one geometry of `kind` per row.
/// `close` says how to close open rings, and only means something for kinds that have them.
fn zip(
    inputs: &[Series],
    kwargs: &MetadataKwargs,
    close: Option<CloseRingsKwargs>,
    kind: Kind,
) -> PolarsResult<Series> {
    let input_fields: Vec<Field> = inputs
        .iter()
        .map(|s| Field::new(s.name().clone(), s.dtype().clone()))
        .collect();
    let geo = zipped(&input_fields, kwargs, kind)?;

    // `get_inner` reads the values array whole, so anything a slice left
    // outside the offsets has to go before the columns can be read in lockstep.
    let columns: Vec<Series> = inputs
        .iter()
        .map(|s| {
            s.trim_lists_to_normalized_offsets()
                .unwrap_or_else(|| s.clone())
                .rechunk()
        })
        .collect();

    // One set of offsets has to describe all of them, or the coordinates do not
    // line up into vertices at all.
    let structure = columns[0].list_offsets_and_validities_recursive();
    for column in &columns[1..] {
        polars_ensure!(
            column.list_offsets_and_validities_recursive() == structure,
            ComputeError: "`{}` and `{}` do not nest the same way: their lists \
            differ in length, or in which of them are missing",
            columns[0].name(), column.name()
        );
    }

    let storage = interleave(&columns, kind.nesting())?;
    Ok(validated(storage, kind, close)?.into_extension(geo.instance()))
}

/// See `linestring_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_linestring_coords)]
fn linestring_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, Kind::LineString)
}

/// See `polygon_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_polygon_coords)]
fn polygon_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs.metadata(), kwargs.close, Kind::Polygon)
}

/// See `multipoint_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipoint_coords)]
fn multipoint_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, Kind::MultiPoint)
}

/// See `multilinestring_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multilinestring_coords)]
fn multilinestring_coords(inputs: &[Series], kwargs: MetadataKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs, None, Kind::MultiLineString)
}

/// See `multipolygon_from_columns`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_multipolygon_coords)]
fn multipolygon_coords(inputs: &[Series], kwargs: PolygonKwargs) -> PolarsResult<Series> {
    zip(inputs, &kwargs.metadata(), kwargs.close, Kind::MultiPolygon)
}
