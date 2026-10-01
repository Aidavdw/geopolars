//! Moving a geometry from one coordinate reference system to another.

use geo::proj::Proj;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::map_coords;
use crate::geoarrow::crs::{crs_of, with_crs};
use crate::geoarrow::{describe, Geo, GeoDimension};

#[derive(Deserialize)]
struct ToCrsKwargs {
    /// The CRS to reproject to, in any form PROJ accepts.
    to: String,
}

/// The geometry we were handed, with its metadata naming the target CRS.
fn reprojected_geometry(dtype: &DataType, to: &str) -> PolarsResult<Geo> {
    let geo = describe(dtype)?;
    let metadata = with_crs(geo.typ.serialize_metadata().as_deref(), to)?;
    Ok(Geo::new(geo.kind, geo.dim, Some(metadata)))
}

/// `output_type_func_with_kwargs` for [`to_crs`]: the same geometry, but the
/// metadata now names the target CRS, so the dtype is a different one.
fn reprojected(input_fields: &[Field], kwargs: ToCrsKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = reprojected_geometry(field.dtype(), &kwargs.to)?;
    Ok(Field::new(field.name().clone(), geo.dtype()))
}

/// Reproject one flat array of coordinates.
///
/// Only `x` and `y` go through PROJ. `z` and `m` are carried through untouched.
fn reproject(coords: &Series, dim: GeoDimension, proj: &Proj) -> PolarsResult<Series> {
    let fields = coords.struct_()?;
    let x = fields.field_by_name("x")?;
    let y = fields.field_by_name("y")?;

    let mut xs = Vec::with_capacity(coords.len());
    let mut ys = Vec::with_capacity(coords.len());
    // A missing coordinate can hold anything in its fields. Never hand that to PROJ.
    let present = coords.is_not_null();
    for ((present, x), y) in present.iter().zip(x.f64()?.iter()).zip(y.f64()?.iter()) {
        let (x, y) = match (present, x, y) {
            (Some(true), Some(x), Some(y)) => {
                let (x, y) = proj.convert((x, y)).map_err(
                    |e| polars_err!(ComputeError: "failed to reproject ({x}, {y}): {e}"),
                )?;
                (Some(x), Some(y))
            }
            _ => (None, None),
        };
        xs.push(x);
        ys.push(y);
    }

    let reprojected =
        dim.field_names()
            .iter()
            .map(|name| {
                let out = match *name {
                    "x" => Float64Chunked::from_iter_options("x".into(), xs.iter().copied())
                        .into_series(),
                    "y" => Float64Chunked::from_iter_options("y".into(), ys.iter().copied())
                        .into_series(),
                    _ => fields.field_by_name(name)?,
                };
                Ok(out)
            })
            .collect::<PolarsResult<Vec<Series>>>()?;

    let mut out =
        StructChunked::from_series(coords.name().clone(), coords.len(), reprojected.iter())?;
    // Same as in `shift`: rebuilding from the fields drops which coordinates
    // were missing entirely.
    out.zip_outer_validity(fields);

    Ok(out.into_series())
}

/// Reproject every coordinate from the CRS the column declares to `to`.
#[polars_expr(output_type_func_with_kwargs=reprojected)]
fn to_crs(inputs: &[Series], kwargs: ToCrsKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let from = crs_of(geo.typ.serialize_metadata().as_deref())?;
    let proj = Proj::new_known_crs(&from, &kwargs.to, None).map_err(
        |e| polars_err!(ComputeError: "cannot reproject from {from} to {}: {e}", kwargs.to),
    )?;

    let out = map_coords(inputs[0].ext()?.storage(), geo.kind.nesting(), &|coords| {
        reproject(coords, geo.dim, &proj)
    })?;

    let typ = reprojected_geometry(inputs[0].dtype(), &kwargs.to)?.instance();
    Ok(out.into_extension(typ))
}
