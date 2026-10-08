//! Geometric transformations that preserves lines and parallelism,
//! but not necessarily Euclidean distances and angles.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use super::coords::map_coords;
use crate::geoarrow::{describe, GeoDimension};

#[derive(Deserialize)]
struct TranslateKwargs {
    dx: f64,
    dy: f64,
    dz: f64,
}

/// Shift one flat array of coordinates by a constant offset.
fn shift(coords: &Series, dim: GeoDimension, kwargs: &TranslateKwargs) -> PolarsResult<Series> {
    let fields = coords.struct_()?;
    // We know the names of the parts are right, but we don't know the exact order they will
    // be in.
    let shifted = dim
        .field_names()
        .iter()
        .map(|name| {
            let coord = fields.field_by_name(name)?;
            let out = match *name {
                "x" => coord + kwargs.dx,
                "y" => coord + kwargs.dy,
                "z" => coord + kwargs.dz,
                // `m` is a measure, not a position. Moving the geometry through
                // space must leave it untouched.
                _ => coord,
            };
            Ok(out.with_name((*name).into()))
        })
        .collect::<PolarsResult<Vec<Series>>>()?;

    let mut out = StructChunked::from_series(coords.name().clone(), coords.len(), shifted.iter())?;
    // Arithmetic per coordinate operates on fields alone,
    // which drops the validity that says a coordinate is missing entirely.
    // We need to re-apply the validity.
    out.zip_outer_validity(fields);

    Ok(out.into_series())
}

/// `output_type_func_with_kwargs` for `translate`:
/// the input geometry, as long as there is a z for `dz` to shift.
fn translated_type(input_fields: &[Field], kwargs: TranslateKwargs) -> PolarsResult<Field> {
    let field = &input_fields[0];
    let geo = describe(field.dtype())?;

    // Without this, translating an XY column by dz would quietly do nothing.
    // Checked here rather than in the kernel, so it fails while the plan is built.
    polars_ensure!(
        kwargs.dz == 0.0 || geo.dim.has_z(),
        SchemaMismatch: "cannot translate by dz: {} has no z coordinate", field.dtype()
    );
    Ok(field.clone())
}

/// See `translate`.
#[polars_expr(output_type_func_with_kwargs=translated_type)]
fn translate(inputs: &[Series], kwargs: TranslateKwargs) -> PolarsResult<Series> {
    let geo = describe(inputs[0].dtype())?;
    let out = map_coords(inputs[0].ext()?.storage(), geo.kind.nesting(), &|coords| {
        shift(coords, geo.dim, &kwargs)
    })?;

    Ok(out.into_extension(geo.typ.clone()))
}
