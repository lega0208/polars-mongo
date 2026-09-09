//! The fixed five-class exception hierarchy (M3).
//!
//! Every class carries typed attributes; library code and programmatic recovery
//! depend only on the class and these attributes, never on message text.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;

create_exception!(
    _internal,
    MongoError,
    PyException,
    "Base class for all polars-mongo errors."
);
create_exception!(
    _internal,
    MongoConnectionClosedError,
    MongoError,
    "The connection was closed before an operation that requires it open."
);
create_exception!(
    _internal,
    MongoSchemaError,
    MongoError,
    "The caller-supplied schema or projection pairing is invalid."
);
create_exception!(
    _internal,
    MongoConversionError,
    MongoError,
    "A BSON value could not be converted to the declared Arrow type."
);
create_exception!(
    _internal,
    MongoWriteError,
    MongoError,
    "One or more individual rows failed to insert."
);
create_exception!(
    _internal,
    MongoBatchError,
    MongoError,
    "A whole batch failed for a non-row reason (for example a write concern error)."
);

/// Build an exception instance with typed attributes attached.
fn raise_with_attrs<'py>(
    py: Python<'py>,
    exc_type: &Bound<'py, PyAny>,
    message: &str,
    attrs: &[(&str, Py<PyAny>)],
) -> PyErr {
    match exc_type.call1((message,)) {
        Ok(instance) => {
            for (name, value) in attrs {
                if let Err(err) = instance.setattr(*name, value.bind(py)) {
                    return err;
                }
            }
            PyErr::from_value(instance)
        }
        Err(err) => err,
    }
}

/// `MongoConnectionClosedError` with its `uri` attribute set.
pub fn connection_closed(py: Python<'_>, uri: &str) -> PyErr {
    let message =
        format!("connection to {uri} is closed; this operation requires it open at collect time");
    let exc_type = py.get_type::<MongoConnectionClosedError>();
    raise_with_attrs(
        py,
        exc_type.as_any(),
        &message,
        &[("uri", uri.into_pyobject(py).unwrap().unbind().into_any())],
    )
}

/// Base `MongoError` for an operational driver failure in a namespace.
pub fn operational_error(py: Python<'_>, namespace: &str, cause: &str) -> PyErr {
    let message = format!("MongoDB operation on {namespace} failed: {cause}");
    let exc_type = py.get_type::<MongoError>();
    raise_with_attrs(py, exc_type.as_any(), &message, &[])
}

/// `MongoSchemaError` with `field_path` and `reason` attributes set.
pub fn schema_error(py: Python<'_>, field_path: &str, reason: &str) -> PyErr {
    let message = format!("schema error at {field_path}: {reason}");
    let exc_type = py.get_type::<MongoSchemaError>();
    raise_with_attrs(
        py,
        exc_type.as_any(),
        &message,
        &[
            (
                "field_path",
                field_path.into_pyobject(py).unwrap().unbind().into_any(),
            ),
            (
                "reason",
                reason.into_pyobject(py).unwrap().unbind().into_any(),
            ),
        ],
    )
}

/// `MongoConversionError` with its full typed attribute set.
pub fn conversion_error(py: Python<'_>, err: &crate::bson_arrow::decode::ConversionError) -> PyErr {
    let exc_type = py.get_type::<MongoConversionError>();
    let attrs: Vec<(&str, Py<PyAny>)> = vec![
        ("document_id", into_py(py, &err.document_id)),
        ("field_path", into_py(py, &err.field_path)),
        ("declared_type", into_py(py, &err.declared_type)),
        ("bson_type", into_py(py, &err.bson_type)),
        ("collection", into_py(py, &err.collection)),
        ("reason", into_py(py, &err.reason)),
    ];
    raise_with_attrs(py, exc_type.as_any(), &err.message(), &attrs)
}

/// `MongoBatchError` with the failed stream row range and driver cause.
pub fn batch_error(py: Python<'_>, row_start: usize, row_stop: usize, cause: &str) -> PyErr {
    let message = format!("batch rows {row_start}..{row_stop} failed: {cause}");
    let exc_type = py.get_type::<MongoBatchError>();
    raise_with_attrs(
        py,
        exc_type.as_any(),
        &message,
        &[
            (
                "row_start",
                row_start.into_pyobject(py).unwrap().unbind().into_any(),
            ),
            (
                "row_stop",
                row_stop.into_pyobject(py).unwrap().unbind().into_any(),
            ),
            (
                "cause",
                cause.into_pyobject(py).unwrap().unbind().into_any(),
            ),
        ],
    )
}

fn into_py(py: Python<'_>, value: &str) -> Py<PyAny> {
    value
        .into_pyobject(py)
        .expect("str always converts")
        .unbind()
        .into_any()
}

/// Register every exception class on the extension module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("MongoError", py.get_type::<MongoError>())?;
    module.add(
        "MongoConnectionClosedError",
        py.get_type::<MongoConnectionClosedError>(),
    )?;
    module.add("MongoSchemaError", py.get_type::<MongoSchemaError>())?;
    module.add(
        "MongoConversionError",
        py.get_type::<MongoConversionError>(),
    )?;
    module.add("MongoWriteError", py.get_type::<MongoWriteError>())?;
    module.add("MongoBatchError", py.get_type::<MongoBatchError>())?;
    Ok(())
}
