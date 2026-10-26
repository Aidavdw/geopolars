//! The bounding box of a geometry.

use geopolars_compute::native::{
    line_string_bounds, multi_line_string_bounds, multi_line_string_part_bounds,
    multi_point_bounds, multi_point_part_bounds, multi_polygon_bounds, multi_polygon_part_bounds,
    point_bounds, polygon_bounds, BoxOptions,
};
use geopolars_compute::with_const_flags;
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::geoarrow::bbox::GeoBox;
use crate::geoarrow::{describe, Kind};

/// See python `bounds`.
/// `turn` is the CRS's longitude turn, asked of PROJ at plan time.
/// If `per_part=true`, a multi-geometry gets a list of boxes, one per part.
/// If `per_part=false`, it gets one box around the whole row.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundsKwargs {
    margin_x: f64,
    margin_y: f64,
    turn: Option<f64>,
    per_part: bool,
}

fn is_multi(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::MultiPoint | Kind::MultiLineString | Kind::MultiPolygon
    )
}

/// The box of `dtype`'s dimension and metadata, in a list if each part gets one.
fn bounds_dtype(dtype: &DataType, per_part: bool) -> PolarsResult<DataType> {
    let geo = describe(dtype)?;
    let bbox = GeoBox::new(geo.dim, geo.metadata.clone()).dtype();
    Ok(if per_part && is_multi(geo.kind) {
        DataType::List(Box::new(bbox))
    } else {
        bbox
    })
}

/// For [`bounds`].
fn kernel_output_type_bounds(input_fields: &[Field], kwargs: BoundsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let dtype = bounds_dtype(field.dtype(), kwargs.per_part)?;
    Ok(Field::new(field.name().clone(), dtype))
}

/// The kernel of `kind` on one chunk of the storage.
fn chunk_bounds(
    chunk: &dyn Array,
    kind: Kind,
    (has_z, has_m): (bool, bool),
    per_part: bool,
    options: &BoxOptions,
) -> PolarsResult<Box<dyn Array>> {
    Ok(match kind {
        Kind::Point => {
            let array = PointArray::new(Coords::new(chunk)?);
            with_const_flags!(point_bounds(&array, options); has_z, has_m).boxed()
        }
        Kind::LineString => {
            let array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            with_const_flags!(line_string_bounds(&array, options); has_z, has_m).boxed()
        }
        Kind::Polygon => {
            let array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            with_const_flags!(polygon_bounds(&array, options); has_z, has_m).boxed()
        }
        Kind::MultiPoint => {
            let array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            if per_part {
                with_const_flags!(multi_point_part_bounds(&array, options); has_z, has_m).boxed()
            } else {
                with_const_flags!(multi_point_bounds(&array, options); has_z, has_m).boxed()
            }
        }
        Kind::MultiLineString => {
            let array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            if per_part {
                with_const_flags!(multi_line_string_part_bounds(&array, options); has_z, has_m)
                    .boxed()
            } else {
                with_const_flags!(multi_line_string_bounds(&array, options); has_z, has_m).boxed()
            }
        }
        Kind::MultiPolygon => {
            let array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            if per_part {
                with_const_flags!(multi_polygon_part_bounds(&array, options); has_z, has_m).boxed()
            } else {
                with_const_flags!(multi_polygon_bounds(&array, options); has_z, has_m).boxed()
            }
        }
    })
}

/// See python `bounds`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_bounds)]
fn bounds(inputs: &[Series], kwargs: BoundsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let (kind, dims) = (geo.kind, (geo.dim.has_z(), geo.dim.has_m()));
    let dtype = bounds_dtype(inputs[0].dtype(), kwargs.per_part)?;
    let options = BoxOptions {
        margin_x: kwargs.margin_x,
        margin_y: kwargs.margin_y,
        turn: kwargs.turn,
    };
    let storage = inputs[0].ext()?.storage();
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_bounds(chunk.as_ref(), kind, dims, kwargs.per_part, &options))
        .collect::<PolarsResult<Vec<_>>>()?;
    // SAFETY: every chunk is the storage of `dtype`.
    Ok(
        unsafe {
            Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, &dtype)
        },
    )
}
