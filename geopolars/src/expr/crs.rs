//! Moving a geometry from one coordinate reference system to another.

use std::sync::Arc;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rsgeo::proj::Proj;
use serde::Deserialize;

use super::coords::map_coords;
use crate::geoarrow::{describe, Geo, GeoDimension};

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

/// `output_type_func_with_kwargs` for [`to_crs`]: the same geometry, but the
/// metadata now names the target CRS, so the dtype is a different one.
fn reprojected(input_fields: &[Field], kwargs: ToCrsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    // Built here as well as in the kernel, so a CRS PROJ cannot use fails while the plan is built.
    transformation(field.dtype(), &kwargs.to)?;
    let geo = reprojected_geometry_type(field.dtype(), &kwargs.to)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// The PROJ transformation from the CRS of `dtype` to `to`.
fn transformation(dtype: &DataType, to: &str) -> PolarsResult<Proj> {
    let from = describe(dtype)?.metadata.crs()?;
    Proj::new_known_crs(&from, to, None)
        .map_err(|e| polars_err!(ComputeError: "cannot reproject from {from} to {to}: {e}"))
}

/// Reproject one flat array of coordinates.
///
/// Only `x` and `y` go through PROJ. `z` and `m` are carried through untouched.
fn reproject(coords: &Series, dim: GeoDimension, proj: &Proj) -> PolarsResult<Series> {
    // TODO: Make this function take a flag 'must_check_validity'.
    // This flag would only have to be set to true if you
    // hand a column of Points, because invalidity can only be on the outer level.
    // Rings/polygons will have them be valid.
    let fields = coords.struct_()?;
    let position = |name: &str| -> PolarsResult<Float64Chunked> {
        Ok(fields.field_by_name(name)?.f64()?.rechunk().into_owned())
    };
    let (x, y) = (position("x")?, position("y")?);
    let (x, y) = (
        x.downcast_as_array().values().as_slice(),
        y.downcast_as_array().values().as_slice(),
    );

    let mut xs = Vec::with_capacity(coords.len());
    let mut ys = Vec::with_capacity(coords.len());
    // A missing point can hold anything in its fields. Never hand that to PROJ.
    let present = fields.rechunk_validity();
    for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
        let (x, y) = if present.as_ref().is_none_or(|present| present.get_bit(i)) {
            proj.convert((x, y))
                .map_err(|e| polars_err!(ComputeError: "failed to reproject ({x}, {y}): {e}"))?
        } else {
            (x, y)
        };
        xs.push(x);
        ys.push(y);
    }

    let xs = Float64Chunked::from_vec("x".into(), xs).into_series();
    let ys = Float64Chunked::from_vec("y".into(), ys).into_series();
    let reprojected = dim
        .field_names()
        .iter()
        .map(|name| {
            let out = match *name {
                "x" => xs.clone(),
                "y" => ys.clone(),
                _ => fields.field_by_name(name)?,
            };
            Ok(out)
        })
        .collect::<PolarsResult<Vec<Series>>>()?;

    let mut out =
        StructChunked::from_series(coords.name().clone(), coords.len(), reprojected.iter())?;
    // For points the coordinate struct is the geometry, and its validity says the point is missing.
    // Building from fields alone drops that.
    out.zip_outer_validity(fields);

    Ok(out.into_series())
}

/// See `to_crs`.
#[polars_expr(output_type_func_with_kwargs=reprojected)]
fn to_crs(inputs: &[Series], kwargs: ToCrsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let proj = transformation(inputs[0].dtype(), &kwargs.to)?;

    let out = map_coords(inputs[0].ext()?.storage(), geo.kind.nesting(), &|coords| {
        reproject(coords, geo.dim, &proj)
    })?;

    let typ = reprojected_geometry_type(inputs[0].dtype(), &kwargs.to)?.instance();
    Ok(out.into_extension(typ))
}
