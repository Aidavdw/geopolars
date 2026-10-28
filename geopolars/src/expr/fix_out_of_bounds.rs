//! Fixing the coordinates outside the CRS's bounds,
//! through `geopolars_compute::native::*_with_fixed_out_of_bounds`.
//! See `fix_out_of_bounds` in Python for the rules.

use geopolars_compute::native::{
    line_strings_with_fixed_out_of_bounds, multi_line_strings_with_fixed_out_of_bounds,
    multi_points_with_fixed_out_of_bounds, multi_polygons_with_fixed_out_of_bounds,
    points_with_fixed_out_of_bounds, polygons_with_fixed_out_of_bounds, Bounds, OutOfBounds,
    OutOfBoundsAction, TooShort,
};
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::{Array, ListArray};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::{kernel_output_type_same_geometry, trimmed};
use super::remove_repeated_points::TooShortDef;
use crate::geoarrow::{describe, Kind};

#[derive(Deserialize)]
#[serde(remote = "OutOfBoundsAction", rename_all = "snake_case")]
enum OutOfBoundsActionDef {
    SetEmpty,
    Drop,
    SetInvalid,
    Clamp,
}

/// See `OutOfBounds` in python, with its `crs_bounds` looked up as `bounds`.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixOutOfBoundsKwargs {
    bounds: Bounds,
    #[serde(with = "OutOfBoundsActionDef")]
    action: OutOfBoundsAction,
    #[serde(with = "TooShortDef")]
    too_short: TooShort,
}

/// [`fix_out_of_bounds`] for one chunk of the storage of a `kind`.
fn chunk_with_fixed_out_of_bounds(
    chunk: &dyn Array,
    kind: Kind,
    fix: &OutOfBounds,
) -> PolarsResult<Box<dyn Array>> {
    let rows = || downcast::<ListArray<i64>>(chunk, "a list of parts");
    Ok(match kind {
        Kind::Point => points_with_fixed_out_of_bounds(&PointArray::new(Coords::new(chunk)?), fix)
            .into_struct_array()
            .boxed(),
        Kind::LineString => {
            line_strings_with_fixed_out_of_bounds(&LineStringArray::new(rows()?)?, fix)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPoint => {
            multi_points_with_fixed_out_of_bounds(&MultiPointArray::new(rows()?)?, fix)
                .into_list_array()
                .boxed()
        }
        Kind::Polygon => polygons_with_fixed_out_of_bounds(&PolygonArray::new(rows()?)?, fix)
            .into_list_array()
            .boxed(),
        Kind::MultiLineString => {
            multi_line_strings_with_fixed_out_of_bounds(&MultiLineStringArray::new(rows()?)?, fix)
                .into_list_array()
                .boxed()
        }
        Kind::MultiPolygon => {
            multi_polygons_with_fixed_out_of_bounds(&MultiPolygonArray::new(rows()?)?, fix)
                .into_list_array()
                .boxed()
        }
    })
}

/// See `fix_out_of_bounds`.
#[polars_expr(output_type_func=kernel_output_type_same_geometry)]
fn fix_out_of_bounds(inputs: &[Series], kwargs: FixOutOfBoundsKwargs) -> PolarsResult<Series> {
    let kind = describe(inputs[0].dtype())?.kind;
    let FixOutOfBoundsKwargs {
        bounds,
        action,
        too_short,
    } = kwargs;
    let fix = OutOfBounds {
        bounds,
        action,
        too_short,
    };
    let storage = trimmed(inputs[0].ext()?.storage());
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_with_fixed_out_of_bounds(chunk.as_ref(), kind, &fix))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is the input's storage.
    Ok(unsafe {
        Series::from_chunks_and_dtype_unchecked(inputs[0].name().clone(), chunks, inputs[0].dtype())
    })
}
