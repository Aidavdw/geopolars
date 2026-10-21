//! Moving a geometry from one coordinate reference system to another.

use std::sync::Arc;

use geopolars_compute::proj::{reproject_coords, reproject_valid_coords};
use geopolars_compute::GeoComputeError;
use geopolars_core::{
    downcast, Coords, LineStringArray, MultiLineStringArray, MultiPointArray, MultiPolygonArray,
    PointArray, PolygonArray,
};
use polars::prelude::*;
use polars_arrow::array::Array;
use pyo3_polars::derive::polars_expr;
use rsgeo::proj::Proj;
use serde::Deserialize;

use super::coords::trimmed;
use crate::geoarrow::{describe, Geo, Kind};

#[derive(Deserialize)]
struct ToCrsKwargs {
    /// The CRS to reproject to, in any form PROJ accepts.
    to: String,
}

/// The geometry we were handed, with its metadata naming the target CRS.
/// Type operation only.
fn reprojected_geometry_type(dtype: &DataType, to: &str) -> PolarsResult<Geo> {
    let geo = describe(dtype)?;
    let metadata = Arc::unwrap_or_clone(geo.metadata.clone()).with_crs(to);
    Ok(Geo::new(geo.kind, geo.dim, Arc::new(metadata)))
}

/// For [`to_crs`]: the same geometry, but the
/// metadata now names the target CRS, so the dtype is a different one.
fn kernel_output_type_to_crs(input_fields: &[Field], kwargs: ToCrsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    // Built here as well as in the kernel, so a CRS PROJ cannot use fails while the plan is built.
    transformation(field.dtype(), &kwargs.to)?;
    let geo = reprojected_geometry_type(field.dtype(), &kwargs.to)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// The PROJ transformation from the CRS of `dtype` to `to`.
fn transformation(dtype: &DataType, to: &str) -> PolarsResult<Proj> {
    let from = describe(dtype)?.metadata.crs()?;
    Proj::new_known_crs(from, to, None)
        .map_err(|e| polars_err!(ComputeError: "cannot reproject from {from} to {to}: {e}"))
}

/// One chunk of `kind` geometries, reprojected by `proj`.
fn chunk_reprojected(chunk: &dyn Array, kind: Kind, proj: &Proj) -> PolarsResult<Box<dyn Array>> {
    let err = |e: GeoComputeError| polars_err!(ComputeError: "{e}");
    Ok(match kind {
        Kind::Point => {
            let coords = reproject_coords(proj, &Coords::new(chunk)?).map_err(err)?;
            PointArray::new(coords).into_struct_array().boxed()
        }
        Kind::LineString => {
            let mut array = LineStringArray::new(downcast(chunk, "a list of coordinates")?)?;
            array.coords = reproject_valid_coords(proj, &array.coords).map_err(err)?;
            array.into_list_array().boxed()
        }
        Kind::MultiPoint => {
            let mut array = MultiPointArray::new(downcast(chunk, "a list of coordinates")?)?;
            array.coords = reproject_valid_coords(proj, &array.coords).map_err(err)?;
            array.into_list_array().boxed()
        }
        Kind::Polygon => {
            let mut array = PolygonArray::new(downcast(chunk, "a list of rings")?)?;
            array.coords = reproject_valid_coords(proj, &array.coords).map_err(err)?;
            array.into_list_array().boxed()
        }
        Kind::MultiLineString => {
            let mut array = MultiLineStringArray::new(downcast(chunk, "a list of linestrings")?)?;
            array.coords = reproject_valid_coords(proj, &array.coords).map_err(err)?;
            array.into_list_array().boxed()
        }
        Kind::MultiPolygon => {
            let mut array = MultiPolygonArray::new(downcast(chunk, "a list of polygons")?)?;
            array.coords = reproject_valid_coords(proj, &array.coords).map_err(err)?;
            array.into_list_array().boxed()
        }
    })
}

/// See `to_crs`.
#[polars_expr(output_type_func_with_kwargs=kernel_output_type_to_crs)]
fn to_crs(inputs: &[Series], kwargs: ToCrsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let proj = transformation(inputs[0].dtype(), &kwargs.to)?;

    let storage = trimmed(inputs[0].ext()?.storage());
    let chunks = storage
        .chunks()
        .iter()
        .map(|chunk| chunk_reprojected(chunk.as_ref(), geo.kind, &proj))
        .collect::<PolarsResult<_>>()?;
    // SAFETY: the geometries keep their kind and dimension, so this is still the storage.
    let out = unsafe {
        Series::from_chunks_and_dtype_unchecked(storage.name().clone(), chunks, storage.dtype())
    };

    let typ = reprojected_geometry_type(inputs[0].dtype(), &kwargs.to)?.instance();
    Ok(out.into_extension(typ))
}
