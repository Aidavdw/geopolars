//! The coordinate reference system a geometry column declares.
//!
//! GeoArrow keeps the CRS in `ARROW:extension:metadata`, a JSON object along
//! the lines of `{"crs": ..., "crs_type": ..., "edges": ...}`. These functions
//! take and give back that raw string, so nothing has to downcast to
//! [`Geo`](super::Geo) to get at it.

use polars::prelude::*;

/// The CRS a column's `ARROW:extension:metadata` declares, in a form PROJ
/// accepts (PROJJSON, WKT or `AUTH:CODE` all work with `proj_create_crs_to_crs`).
///
/// Errors if there is none: we cannot reproject from an unknown CRS.
pub fn crs_of(_metadata: Option<&str>) -> PolarsResult<String> {
    todo!("parse the crs out of the GeoArrow extension metadata")
}

/// `metadata` with its CRS replaced by `crs`. Every other key (`edges`) is kept.
pub fn with_crs(_metadata: Option<&str>, _crs: &str) -> PolarsResult<String> {
    todo!("write the crs back into the GeoArrow extension metadata")
}
