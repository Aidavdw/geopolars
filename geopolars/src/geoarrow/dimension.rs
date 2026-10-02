//! In GeoArrow, a coordinate can be 2D or 3D,
//! and optionally carry a 'measurement'.
//!
//! `Geometry` can be built up from any of these.
//! Dimensional information is carried by the coordinates, and forwarded to the geometries.
//!
//! All variations are defined in [GeoDimension].
//! Note this is one enum rather than four separate rust types.
//! In polars we already work with an abstraction over the type system,
//! and this extends that framework.

use polars_core::prelude::{DataType, Field};

/// Which coordinate dimensions a coordinate/geometry carries.
///
/// Because the spec fixes the order of the coordinate data,
/// this also determines its dimension.
///
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeoDimension {
    // This is named 'GeoDimension' and not 'Dimension' because for some reason
    // rust-analyzer really messes up by confusing it with [polars::datatype::Dimension].
    XY,
    XYZ,
    XYM,
    XYZM,
}

/// Const impl
impl GeoDimension {
    pub const ALL: [GeoDimension; 4] = [
        GeoDimension::XY,
        GeoDimension::XYZ,
        GeoDimension::XYM,
        GeoDimension::XYZM,
    ];

    /// The storage struct's field names, in the order the spec requires.
    pub const fn field_names(self) -> &'static [&'static str] {
        match self {
            GeoDimension::XY => &["x", "y"],
            GeoDimension::XYZ => &["x", "y", "z"],
            GeoDimension::XYM => &["x", "y", "m"],
            GeoDimension::XYZM => &["x", "y", "z", "m"],
        }
    }

    /// How the dimension is spelled in a type name: `point[xy]`, `LineStringXY`.
    /// The same thing [`Dimension::field_names`] says, but as one `const` string.
    pub const fn tag(self) -> &'static str {
        match self {
            GeoDimension::XY => "xy",
            GeoDimension::XYZ => "xyz",
            GeoDimension::XYM => "xym",
            GeoDimension::XYZM => "xyzm",
        }
    }

    /// `true` if this dimension has a `z` (elevation) coordinate.
    pub const fn has_z(self) -> bool {
        matches!(self, GeoDimension::XYZ | GeoDimension::XYZM)
    }

    /// `true` if this dimension has an `m` (measure) value.
    #[allow(dead_code)]
    pub const fn has_m(self) -> bool {
        matches!(self, GeoDimension::XYM | GeoDimension::XYZM)
    }

    /// The separated-coordinate struct an array of these coordinates is stored as.
    pub fn coordinates(self) -> DataType {
        DataType::Struct(
            self.field_names()
                .iter()
                .map(|name| Field::new((*name).into(), DataType::Float64))
                .collect(),
        )
    }

    /// The storage of a geometry that nests its coordinates `nesting` `List`
    /// layers deep. See [`Kind::nesting`](super::Kind::nesting).
    pub fn storage(self, nesting: u8) -> DataType {
        (0..nesting).fold(self.coordinates(), |inner, _| {
            DataType::List(Box::new(inner))
        })
    }
}

#[cfg(test)]
mod tests {
    //! Only what the Python suite cannot reach.

    use super::*;

    /// Consistency check
    #[test]
    fn the_tag_follows_from_the_field_names() {
        for dim in GeoDimension::ALL {
            assert_eq!(dim.tag(), dim.field_names().concat());
        }
    }
}
