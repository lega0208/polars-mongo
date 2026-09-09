//! The read path: a cursor-backed batch reader exposed to `register_io_source`.
//!
//! One reader is built per IO-source invocation. The cursor lives behind a
//! `Mutex` because the sync `Cursor` is `Send` but not `Sync` (F2), and the GIL
//! is released around the driver call only - every PyO3 value is built after
//! reattaching.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mongodb::bson::{Document, RawDocumentBuf};
use mongodb::options::FindOptions;
use mongodb::sync::Cursor;
use polars_core::frame::DataFrame;
use polars_core::frame::column::Column;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;
use pyo3_polars::PyDataFrame;

use crate::bson_arrow::decode::decode_documents;
use crate::bson_arrow::filter::dict_to_document;
use crate::bson_arrow::schema::SchemaPlan;
use crate::conn::MongoConnection;
use crate::errors::{conversion_error, operational_error, schema_error};

/// Test-only observation of the GIL-release window around the driver call.
///
/// AC-3 needs a *synchronized* experiment: a probe thread must measure its own
/// progress strictly between these two signals, not around incidental work.
static DETACH_ENTERED: AtomicU64 = AtomicU64::new(0);
static DETACH_EXITED: AtomicU64 = AtomicU64::new(0);

pub fn detach_counters() -> (u64, u64) {
    (
        DETACH_ENTERED.load(Ordering::SeqCst),
        DETACH_EXITED.load(Ordering::SeqCst),
    )
}

struct CursorState {
    /// `None` once the scan is finished or has failed: the cursor is dropped at
    /// that moment so a partially consumed scan does not hold a server cursor
    /// open for the lifetime of the reader object.
    cursor: Option<Cursor<RawDocumentBuf>>,
}

impl CursorState {
    fn close(&mut self) {
        self.cursor = None;
    }
}

/// Compile-time proof that the reader is safe to hand to Python.
const fn assert_send_sync<T: Send + Sync>() {}
const _: () = assert_send_sync::<MongoBatchReader>();

/// A batching reader over one `find` cursor.
#[pyclass(module = "polars_mongo._internal")]
pub struct MongoBatchReader {
    state: Arc<Mutex<CursorState>>,
    plan: Arc<SchemaPlan>,
    namespace: String,
    batch_size: usize,
}

impl MongoBatchReader {
    fn next_batch(&self) -> PyResult<Option<DataFrame>> {
        Python::attach(|py| {
            let state = Arc::clone(&self.state);
            let batch_size = self.batch_size;

            let pulled: Result<Vec<RawDocumentBuf>, mongodb::error::Error> = py.detach(move || {
                let mut guard = state.lock().expect("cursor mutex poisoned");
                let Some(cursor) = guard.cursor.as_mut() else {
                    return Ok(Vec::new());
                };
                let mut documents = Vec::with_capacity(batch_size);
                let mut finished = false;
                let mut failure = None;
                DETACH_ENTERED.fetch_add(1, Ordering::SeqCst);
                while documents.len() < batch_size {
                    match cursor.advance() {
                        Ok(true) => match cursor.deserialize_current() {
                            Ok(document) => documents.push(document),
                            Err(err) => {
                                failure = Some(err);
                                break;
                            }
                        },
                        Ok(false) => {
                            finished = true;
                            break;
                        }
                        Err(err) => {
                            failure = Some(err);
                            break;
                        }
                    }
                }
                DETACH_EXITED.fetch_add(1, Ordering::SeqCst);
                // Release the server cursor as soon as the scan ends, so a
                // partially consumed reader cannot hold one open.
                if finished || failure.is_some() {
                    guard.close();
                }
                match failure {
                    Some(err) => Err(err),
                    None => Ok(documents),
                }
            });

            let documents = pulled.map_err(|err| {
                operational_error(py, &self.namespace, &format!("cursor failed: {err}"))
            })?;
            if documents.is_empty() {
                return Ok(None);
            }

            let refs: Vec<&mongodb::bson::RawDocument> =
                documents.iter().map(|document| document.as_ref()).collect();
            let series = decode_documents(&self.plan, &refs, &self.namespace).map_err(|err| {
                // A failed conversion ends the scan: release the cursor rather
                // than leaving it open behind a reader nobody will drain.
                self.state.lock().expect("cursor mutex poisoned").close();
                conversion_error(py, &err)
            })?;
            let height = documents.len();
            let columns: Vec<Column> = series.into_iter().map(Column::from).collect();
            let frame = DataFrame::new(height, columns).map_err(|err| {
                schema_error(
                    py,
                    &self.namespace,
                    &format!("could not build a frame: {err}"),
                )
            })?;
            Ok(Some(frame))
        })
    }
}

#[pymethods]
impl MongoBatchReader {
    #[new]
    #[pyo3(signature = (
        connection,
        database,
        collection,
        schema,
        filter = None,
        projection = None,
        limit = None,
        batch_size = 1024,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        connection: PyRef<'_, MongoConnection>,
        database: &str,
        collection: &str,
        schema: &Bound<'_, PyAny>,
        filter: Option<&Bound<'_, PyDict>>,
        projection: Option<&Bound<'_, PyDict>>,
        limit: Option<i64>,
        batch_size: usize,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(schema_error(
                py,
                "batch_size",
                "must be a positive integer; got 0",
            ));
        }
        let plan = crate::api::import_schema(py, schema)?;
        let filter = match filter {
            Some(dict) => dict_to_document(dict, "filter")
                .map_err(|err| schema_error(py, &err.field_path, &err.reason))?,
            None => Document::new(),
        };
        let projection = match projection {
            Some(dict) => Some(
                dict_to_document(dict, "projection")
                    .map_err(|err| schema_error(py, &err.field_path, &err.reason))?,
            ),
            None => None,
        };

        let client = connection.client(py)?;
        let namespace = format!("{database}.{collection}");
        let options = FindOptions::builder()
            .projection(projection)
            .limit(limit)
            .batch_size(u32::try_from(batch_size).ok())
            .build();

        let database = database.to_owned();
        let collection_name = collection.to_owned();
        let cursor = py
            .detach(move || {
                client
                    .database(&database)
                    .collection::<RawDocumentBuf>(&collection_name)
                    .find(filter)
                    .with_options(options)
                    .run()
            })
            .map_err(|err| operational_error(py, &namespace, &format!("find failed: {err}")))?;

        Ok(Self {
            state: Arc::new(Mutex::new(CursorState {
                cursor: Some(cursor),
            })),
            plan: Arc::new(plan),
            namespace,
            batch_size,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyDataFrame>> {
        let _ = py;
        Ok(self.next_batch()?.map(PyDataFrame))
    }
}

#[cfg(test)]
mod tests {
    use mongodb::sync::Client;

    /// The supported execution mode, pinned rather than detected.
    ///
    /// The documented policy is *document, do not detect*: an IO entrypoint is
    /// supported from a plain thread with no active Tokio runtime context, and
    /// unsupported from a thread that has one. The sync driver builds its own
    /// runtime, so constructing and using a client from a plain
    /// `std::thread` - which is exactly how Polars drives the IO source - must
    /// work, and nothing in this crate may add a detection branch for the
    /// unsupported case.
    #[test]
    fn the_sync_client_is_usable_from_a_plain_non_async_thread() {
        let handle = std::thread::spawn(|| {
            // Connection is lazy, so this exercises the runtime construction
            // that an active Tokio context would reject, without a server.
            let client = Client::with_uri_str("mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=1")
                .expect("the sync client must build off a plain thread");
            client.database("polars_mongo_test").name().to_owned()
        });

        assert_eq!(
            handle.join().expect("the plain thread must not panic"),
            "polars_mongo_test"
        );
    }

    /// The reader is handed to Python and driven from Polars' own threads.
    #[test]
    fn the_reader_is_send_and_sync() {
        super::assert_send_sync::<super::MongoBatchReader>();
    }

    /// Even an exhausted reader enters the actual Python detach/reattach path.
    #[test]
    fn exhausted_reader_is_driven_from_a_plain_non_async_thread() {
        pyo3::Python::initialize();
        let handle = std::thread::spawn(|| {
            let reader = super::MongoBatchReader {
                state: std::sync::Arc::new(std::sync::Mutex::new(super::CursorState {
                    cursor: None,
                })),
                plan: std::sync::Arc::new(crate::bson_arrow::schema::SchemaPlan {
                    fields: Vec::new(),
                }),
                namespace: "test.exhausted".to_owned(),
                batch_size: 1,
            };
            assert!(reader.next_batch().expect("reader must run").is_none());
        });
        handle.join().expect("reader thread must not panic");
    }
}

/// Test-only signal for AC-3: `(entered, exited)` detach counts.
///
/// A probe thread reads this to bound its measurement strictly to a window in
/// which the reader is observably inside the blocking driver call, rather than
/// timing incidental surrounding Python work.
#[pyfunction]
#[pyo3(name = "_detach_counters")]
fn py_detach_counters() -> (u64, u64) {
    detach_counters()
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MongoBatchReader>()?;
    module.add_function(wrap_pyfunction!(py_detach_counters, module)?)
}
