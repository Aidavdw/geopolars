//! Glue for the GeoArrow data types, and dispatching to it.
//!
//! Expressions never name a concrete geometry:
//! They ask [`describe`] what a column holds and dispatch on the [`Kind`] and [`Dimension`] it reports,
//! so adding a geometry does not touch `crate::expr`.

pub mod coord;
pub mod crs;
mod dimension;
mod geo;
mod kind;

pub use dimension::GeoDimension;
pub use geo::Geo;
pub use kind::Kind;

use std::sync::Arc;

use polars::prelude::*;
use polars_core::datatypes::extension::{register_extension_type, ExtensionTypeInstance};

use geo::GeoFactory;

/// Populate this library's extension-type registry.
/// This exposes the types from this plugin's library.
/// This cannot run lazily.
///
/// Registering also needs to happen in the host `polars` wheel,
/// which is done with `pl.register_extension_type`.
/// Both are needed, and they must agree on names.
pub fn register() -> PolarsResult<()> {
    for kind in Kind::ALL {
        register_extension_type(kind.name(), Some(Arc::new(GeoFactory(kind))))?;
    }
    Ok(())
}

/// A view of a geometry column, containing only what an expression over it needs.
pub struct GeoColumn<'a> {
    /// The [`ExtensionTypeInstance`] the column came in, so the result can be
    /// put back under it without losing metadata.
    pub typ: &'a ExtensionTypeInstance,
    pub kind: Kind,
    pub dim: GeoDimension,
}

/// Parses a geometry column's dtype (as a polars [DataType]) into a [GeoColumn].
pub fn describe(dtype: &DataType) -> PolarsResult<GeoColumn<'_>> {
    let DataType::Extension(typ, storage) = dtype else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    let Some(kind) = Kind::from_name(&typ.name()) else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    let Some(dim) = coord::dimension_of_storage(storage, kind.nesting()) else {
        // One of ours, over storage we do not implement -- an interleaved
        // encoding, most likely. See `Unsupported`.
        polars_bail!(
            SchemaMismatch: "expected a `{}` column with separated \
            x/y[/z][/m] coordinates, got: {}", kind.name(), dtype
        );
    };
    Ok(GeoColumn { typ, kind, dim })
}
