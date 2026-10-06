use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expr;
mod geoarrow;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// See `GeoArrowType.__init__`.
#[pyfunction]
#[pyo3(signature = (*, crs = None))]
fn extension_metadata(crs: Option<String>) -> PyResult<Option<String>> {
    geoarrow::crs::MetadataKwargs { crs }
        .apply(Default::default())
        .map(|metadata| metadata.serialize())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// See `GeoArrowType._declares_crs`.
#[pyfunction]
#[pyo3(signature = (metadata))]
fn declares_crs(metadata: Option<&str>) -> PyResult<bool> {
    geoarrow::crs::ExtensionMetadata::parse(metadata)
        .map(|metadata| metadata.declares_crs())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The plugin's Python module.
/// Here so that extension types are explicitly registered with Polars with `PyInit_geopolars`.
#[pymodule]
fn geopolars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expressions don't have to be registered,
    // that hook is already by calling the plugin functions.
    m.add_function(wrap_pyfunction!(extension_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(declares_crs, m)?)?;
    geoarrow::register().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register geoarrow extension types: {e}"
        ))
    })
}
