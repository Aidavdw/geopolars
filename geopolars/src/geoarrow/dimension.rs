//! The polars dtypes a [GeoDimension] is stored as.
//! [GeoDimension] itself lives in `geoarrow_abstr`, which does not know about polars.

use geoarrow_abstr::GeoDimension;
use polars_core::prelude::{DataType, Field};

/// The polars dtypes of coordinates, geometries and boxes of a [GeoDimension].
pub trait GeoDimensionDtype {
    /// The separated-coordinate struct an array of these coordinates is stored as.
    fn coordinates(self) -> DataType;

    /// The storage of a geometry that nests its coordinates `nesting` `List`
    /// layers deep. See [`Kind::nesting`](super::Kind::nesting).
    fn storage(self, nesting: u8) -> DataType;

    /// The struct a `geoarrow.box` of this dimension is stored as.
    fn box_storage(self) -> DataType;
}

impl GeoDimensionDtype for GeoDimension {
    fn coordinates(self) -> DataType {
        DataType::Struct(
            self.field_names()
                .iter()
                .map(|name| Field::new((*name).into(), DataType::Float64))
                .collect(),
        )
    }

    fn storage(self, nesting: u8) -> DataType {
        (0..nesting).fold(self.coordinates(), |inner, _| {
            DataType::List(Box::new(inner))
        })
    }

    fn box_storage(self) -> DataType {
        DataType::Struct(
            self.box_field_names()
                .iter()
                .map(|name| Field::new((*name).into(), DataType::Float64))
                .collect(),
        )
    }
}
