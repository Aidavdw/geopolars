//! Shared functionality for expressions is often implemented over coordinates,
//! the shared building block of geometries.

use polars::prelude::*;

use crate::geoarrow::describe;

/// `output_type_func` for an operation that hands back the geometry it got.
pub fn same_geometry(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(field.clone())
}

/// Apply a coordinate-wise kernel through however many `List` layers the
/// geometry nests its coordinates under.
///
/// This works on any geometry, because it works on the coordinates directly.
/// Geometry-specific data (e.g. group association) is never touched.
pub fn map_coords(
    storage: &Series,
    nesting: u8,
    f: &dyn Fn(&Series) -> PolarsResult<Series>,
) -> PolarsResult<Series> {
    match nesting {
        0 => f(storage),
        n => Ok(storage
            .list()?
            .apply_to_inner(&|inner| map_coords(&inner, n - 1, f))?
            .into_series()),
    }
}
