//! The `MongoConnection` handle: the single chokepoint through which every
//! driver call in this crate obtains a client.

use std::sync::Arc;

use mongodb::bson::doc;
use mongodb::sync::Client;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::errors::connection_closed;

/// A MongoDB connection handle usable as a context manager.
///
/// The driver connects lazily, so construction never contacts a server.
#[pyclass(module = "polars_mongo._internal")]
pub struct MongoConnection {
    uri: String,
    client: Option<Arc<Client>>,
    closed: bool,
}

impl MongoConnection {
    /// The single accessor for the underlying client.
    ///
    /// Every driver call in this crate goes through here, so "closed" is
    /// enforced in exactly one place.
    pub fn client(&self, py: Python<'_>) -> PyResult<Arc<Client>> {
        if self.closed {
            return Err(connection_closed(py, &self.uri));
        }
        match &self.client {
            Some(client) => Ok(Arc::clone(client)),
            None => Err(connection_closed(py, &self.uri)),
        }
    }

    /// The URI this connection was constructed with.
    pub fn uri(&self) -> &str {
        &self.uri
    }
}

#[pymethods]
impl MongoConnection {
    #[new]
    fn new(py: Python<'_>, uri: String) -> PyResult<Self> {
        let built = py.detach(|| Client::with_uri_str(&uri));
        let client = built.map_err(|err| {
            crate::errors::schema_error(py, "uri", &format!("invalid MongoDB URI: {err}"))
        })?;
        Ok(Self {
            uri,
            client: Some(Arc::new(client)),
            closed: false,
        })
    }

    #[getter]
    fn uri_(&self) -> &str {
        &self.uri
    }

    #[getter]
    fn closed(&self) -> bool {
        self.closed
    }

    /// Release the client. Idempotent.
    fn close(&mut self, py: Python<'_>) {
        if let Some(client) = self.client.take() {
            py.detach(move || {
                if let Ok(client) = Arc::try_unwrap(client) {
                    client.shutdown().run();
                }
            });
        }
        self.closed = true;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, py: Python<'_>, _args: &Bound<'_, PyAny>) -> bool {
        self.close(py);
        false
    }

    /// Test-only: issue a single trivial, read-only server command through
    /// `client()`.
    ///
    /// Private by name, absent from `__all__`, never exported from
    /// `polars_mongo`, and never broadened to carry insert or arbitrary
    /// command traffic. It exists so the connection lifecycle can be proven
    /// against *this* client before any IO phase exists.
    ///
    /// The command is fixed as `dbStats`, not `ping`: the database profiler
    /// does not record `ping`, and it records nothing at all against `admin`,
    /// so a `ping` would be invisible to the `appName`-scoping harness
    /// validation. Only the target database is a parameter.
    #[pyo3(signature = (database = "admin"))]
    fn _test_ping(&self, py: Python<'_>, database: &str) -> PyResult<()> {
        let client = self.client(py)?;
        let database = database.to_owned();
        let result = py.detach(move || {
            client
                .database(&database)
                .run_command(doc! { "dbStats": 1 })
                .run()
        });
        result.map(|_| ()).map_err(|err| {
            crate::errors::schema_error(py, "_test_ping", &format!("dbStats failed: {err}"))
        })
    }
}

/// Register the connection class on the extension module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MongoConnection>()
}
