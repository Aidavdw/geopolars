//! The `geoarrow.wkb` extension type

use std::any::Any;
use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use polars::prelude::*;
use polars_core::datatypes::extension::{
    ExtensionTypeFactory, ExtensionTypeImpl, ExtensionTypeInstance,
};

use super::crs::ExtensionMetadata;

/// A `geoarrow.wkb` column's type.
///
/// WKB is for exchange only (e.g. GeoParquet).
/// It is deliberately not a [`Kind`](super::Kind): no expression operates on it,
/// other than converting it to and from the native geometries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Wkb {
    // Behind an [`Arc`], since dtypes get cloned a lot during planning.
    metadata: Arc<ExtensionMetadata>,
}

impl Wkb {
    /// The `ARROW:extension:name` WKB is registered under.
    /// The Python side must spell the same name.
    pub const NAME: &'static str = "geoarrow.wkb";

    pub fn new(metadata: Arc<ExtensionMetadata>) -> Self {
        Self { metadata }
    }

    pub fn instance(self) -> ExtensionTypeInstance {
        ExtensionTypeInstance(Box::new(self))
    }

    pub fn dtype(self) -> DataType {
        DataType::Extension(self.instance(), Box::new(DataType::Binary))
    }
}

/// Polars arrow extension type trait impl
impl ExtensionTypeImpl for Wkb {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(Self::NAME)
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.serialize().map(Cow::Owned)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        (other as &dyn Any)
            .downcast_ref::<Self>()
            .is_some_and(|o| self == o)
    }

    fn dyn_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed("wkb")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Borrowed("WKB")
    }
}

/// A `geoarrow.wkb` whose storage is not `Binary`, or whose metadata is not a JSON object.
/// The WKB counterpart of [`Unsupported`](super::geo::Unsupported).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnsupportedWkb {
    storage: DataType,
    /// Kept verbatim, as it might not parse.
    metadata: Option<String>,
}

impl ExtensionTypeImpl for UnsupportedWkb {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(Wkb::NAME)
    }

    fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.metadata.as_deref().map(Cow::Borrowed)
    }

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl> {
        Box::new(self.clone())
    }

    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool {
        (other as &dyn Any)
            .downcast_ref::<Self>()
            .is_some_and(|o| self == o)
    }

    fn dyn_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    fn dyn_display(&self) -> Cow<'_, str> {
        Cow::Borrowed("wkb[?]")
    }

    fn dyn_debug(&self) -> Cow<'_, str> {
        Cow::Owned(format!("UnsupportedWkb({:?})", self.storage))
    }
}

pub struct WkbFactory;

impl ExtensionTypeFactory for WkbFactory {
    fn create_type_instance(
        &self,
        _name: &str,
        storage: &DataType,
        metadata: Option<&str>,
    ) -> Box<dyn ExtensionTypeImpl> {
        match (storage, ExtensionMetadata::parse(metadata)) {
            (DataType::Binary, Ok(parsed)) => Box::new(Wkb::new(Arc::new(parsed))),
            _ => Box::new(UnsupportedWkb {
                storage: storage.clone(),
                metadata: metadata.map(str::to_owned),
            }),
        }
    }
}

/// The metadata of a column holding WKB:
/// a `geoarrow.wkb` column's own, or none for plain `Binary`
pub fn describe_wkb(dtype: &DataType) -> PolarsResult<Arc<ExtensionMetadata>> {
    match dtype {
        DataType::Binary => Ok(Default::default()),
        DataType::Extension(typ, _) if typ.name() == Wkb::NAME => {
            match (typ.0.as_ref() as &dyn Any).downcast_ref::<Wkb>() {
                Some(wkb) => Ok(wkb.metadata.clone()),
                None => polars_bail!(
                    SchemaMismatch: "expected a `{}` column over binary storage, got: {}",
                    Wkb::NAME, dtype
                ),
            }
        }
        _ => polars_bail!(
            SchemaMismatch: "expected a `{}` or binary column, got: {}", Wkb::NAME, dtype
        ),
    }
}
