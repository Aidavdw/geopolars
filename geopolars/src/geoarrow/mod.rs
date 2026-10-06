//! Glue for the GeoArrow data types, and dispatching to it.
//!
//! Expressions never name a concrete geometry:
//! They ask [`describe`] what a column holds and dispatch on the [`Kind`] and [`Dimension`] it reports,
//! so adding a geometry does not touch `crate::expr`.

pub mod coord;
pub mod crs;
mod dimension;
mod geo;
pub mod geodetic;
mod kind;
pub mod storage;

pub use dimension::GeoDimension;
pub use geo::Geo;
pub use kind::Kind;

use std::any::Any;
use std::sync::Arc;

use polars::prelude::*;
use polars_core::datatypes::extension::{register_extension_type, ExtensionTypeInstance};

use crs::ExtensionMetadata;
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
    pub metadata: &'a Arc<ExtensionMetadata>,
}

/// Parses a geometry column's dtype (as a polars [DataType]) into a [GeoColumn].
pub fn describe(dtype: &DataType) -> PolarsResult<GeoColumn<'_>> {
    let DataType::Extension(typ, storage) = dtype else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    let Some(kind) = Kind::from_name(&typ.name()) else {
        polars_bail!(SchemaMismatch: "expected a {} column, got: {}", Kind::names(), dtype);
    };
    // `GeoFactory` already parsed everything, unless it had to give up.
    if let Some(geo) = (typ.0.as_ref() as &dyn Any).downcast_ref::<Geo>() {
        return Ok(GeoColumn {
            typ,
            kind: geo.kind(),
            dim: geo.dim(),
            metadata: geo.metadata(),
        });
    }

    polars_ensure!(
        coord::dimension_of_storage(storage, kind.nesting()).is_some(),
        // An interleaved encoding, most likely.
        SchemaMismatch: "expected a `{}` column with separated \
        x/y[/z][/m] coordinates, got: {}", kind.name(), dtype
    );
    ExtensionMetadata::parse(typ.serialize_metadata().as_deref())?;
    polars_bail!(
        SchemaMismatch: "`{}` is not registered with this library, got: {}", kind.name(), dtype
    )
}
