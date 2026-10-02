use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expr;
mod geoarrow;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// The `ARROW:extension:metadata` a geometry dtype built with these fields carries.
///
/// Used for constructing dtype, keeping stuff on the python and rust side in agreement.
#[pyfunction]
#[pyo3(signature = (*, crs = None))]
fn extension_metadata(crs: Option<String>) -> PyResult<Option<String>> {
    geoarrow::crs::MetadataKwargs { crs }
        .apply(None)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The plugin's Python module.
/// Unlike custom expressions, extension types must be explicitly
/// registered with Polars.
/// `PyInit_geopolars` is that hook.
#[pymodule]
fn geopolars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extension_metadata, m)?)?;
    geoarrow::register().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to register geoarrow extension types: {e}"
        ))
    })
}
