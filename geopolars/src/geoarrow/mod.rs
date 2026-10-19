//! Glue for the GeoArrow data types, and dispatching to it.
//!
//! Expressions never name a concrete geometry:
//! They ask [`describe`] what a column holds and dispatch on the [`Kind`] and [`Dimension`] it reports,
//! so adding a geometry does not touch `crate::expr`.

pub mod bbox;
pub mod coord;
pub mod crs;
pub mod decode;
pub mod encoded;
mod geo;
pub mod geodetic;
pub mod geotraits;
mod kind;

pub use geo::Geo;
pub use geopolars_core::GeoDimension;
pub use kind::Kind;

use std::any::Any;
use std::sync::Arc;

use polars::prelude::*;
use polars_core::datatypes::extension::{register_extension_type, ExtensionTypeInstance};

use bbox::BoxFactory;
use crs::ExtensionMetadata;
use encoded::{EncodedFactory, Encoding};
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
    for encoding in Encoding::ALL {
        register_extension_type(encoding.name(), Some(Arc::new(EncodedFactory(encoding))))?;
    }
    register_extension_type(bbox::NAME, Some(Arc::new(BoxFactory)))?;
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

/// Errors unless `a` and `b` declare the same CRS, or neither declares one.
/// Should be used as a backup, as this is runtime, not plan time.
pub fn ensure_same_crs(a: &GeoColumn, b: &GeoColumn) -> PolarsResult<()> {
    match (a.metadata.declares_crs(), b.metadata.declares_crs()) {
        (false, false) => Ok(()),
        (true, true) => {
            let (crs_a, crs_b) = (a.metadata.crs()?, b.metadata.crs()?);
            polars_ensure!(
                crs_a == crs_b,
                SchemaMismatch: "the geometries declare different CRSs, {crs_a} and {crs_b}; \
                use `to_crs` to bring one onto the other first"
            );
            Ok(())
        }
        (true, false) | (false, true) => polars_bail!(
            SchemaMismatch: "only one of the geometries declares a CRS; declare the same one on both"
        ),
    }
}
