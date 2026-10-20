use std::fmt;

use proj::ProjError;

/// An error from one of the kernels of this crate.
#[derive(Debug)]
pub enum GeoComputeError {
    /// A coordinate PROJ could not bring to longitude/latitude.
    Reproject {
        position: (f64, f64),
        source: ProjError,
    },
    #[cfg(feature = "geos")]
    Geos(geos::Error),
}

pub type GeoComputeResult<T> = Result<T, GeoComputeError>;

impl fmt::Display for GeoComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reproject {
                position: (x, y),
                source,
            } => write!(f, "failed to reproject ({x}, {y}): {source}"),
            #[cfg(feature = "geos")]
            Self::Geos(source) => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for GeoComputeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reproject { source, .. } => Some(source),
            #[cfg(feature = "geos")]
            Self::Geos(source) => Some(source),
        }
    }
}

#[cfg(feature = "geos")]
impl From<geos::Error> for GeoComputeError {
    fn from(source: geos::Error) -> Self {
        Self::Geos(source)
    }
}
