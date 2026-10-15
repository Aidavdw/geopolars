//! Interfacing with the `geos` crate.
//!
//! GEOS, as a C++ library, owns its coordinate storage,
//! so arrow buffers can't be borrowed zero-copy.
//! The copy is one memcpy-like op, so still not very heavy.

use geos::{CoordSeq, CoordType, Geometry};

use crate::{GeoArrowCoordsView, GeoArrowValidCoordsView, GeoDimension, LineString, Point};

impl From<GeoDimension> for CoordType {
    fn from(dim: GeoDimension) -> Self {
        match dim {
            GeoDimension::XY => CoordType::XY,
            GeoDimension::XYZ => CoordType::XYZ,
            GeoDimension::XYM => CoordType::XYM,
            GeoDimension::XYZM => CoordType::XYZM,
        }
    }
}

impl<C: GeoArrowCoordsView> Point<'_, C> {
    /// `None` if the point is empty (missing).
    pub fn to_geos(&self) -> geos::GResult<Option<Geometry>> {
        let (coords, i) = (self.coords, self.index);
        let Some((x, y)) = coords.xy(i) else {
            return Ok(None);
        };
        // GeoArrow's empty point. GEOS only counts a point without coordinates as empty.
        if x.is_nan() && y.is_nan() {
            return Geometry::create_empty_point().map(Some);
        }
        let dim = coords.dim();
        let len = dim.field_names().len();
        let mut buffer = [0.0; 4];
        for (n, value) in buffer[..len].iter_mut().enumerate() {
            *value = coords.nth(i, n);
        }
        let seq = CoordSeq::new_from_buffer(&buffer[..len], 1, dim.into())?;
        Geometry::create_point(seq).map(Some)
    }
}
