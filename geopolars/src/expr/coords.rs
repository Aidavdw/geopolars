//! Shared functionality for expressions is often implemented over coordinates,
//! the shared building block of geometries.

use polars::prelude::*;

use crate::geoarrow::describe;

/// For an operation that hands back the geometry it got.
pub fn kernel_output_type_same_geometry(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    describe(field.dtype())?;
    Ok(field.clone())
}

/// `storage`, holding only the coordinates its rows use.
/// A streaming morsel still holds every coordinate of the column it was sliced from,
/// which the kernels would otherwise all process. Trimming is zero-copy.
pub fn trimmed(storage: &Series) -> Series {
    storage
        .trim_lists_to_normalized_offsets()
        .unwrap_or_else(|| storage.clone())
}
