mod api;
pub mod bson_arrow;
mod conn;
pub mod errors;
mod expressions;
pub mod objectid;
pub mod read;
mod write;

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// The compiled extension module.
///
/// Polars loads the cdylib directly through `register_plugin_function` for the
/// `#[polars_expr]` entry points in `expressions.rs`; this `#[pymodule]` exists
/// for the exception classes and the connection type, which Python imports by
/// name.
#[pymodule]
fn _internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    conn::register(module)?;
    api::register(module)?;
    read::register(module)?;
    write::register(module)?;
    Ok(())
}
