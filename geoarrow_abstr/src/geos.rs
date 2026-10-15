//! Interfacing with the `geos` crate.

use crate::{GeoArrowValidCoordsView, LineString};

fn line_string_to_geos<C: GeoArrowValidCoordsView>(line_string: LineString<C>) -> geos::Geometry {
    todo!()
}
