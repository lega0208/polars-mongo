//! The write path: a reusable schema-bound MongoDB batch writer.

use std::sync::Arc;

use mongodb::error::ErrorKind;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_polars::PyDataFrame;

use crate::bson_arrow::encode::encode_documents;
use crate::bson_arrow::schema::SchemaPlan;
use crate::conn::MongoConnection;
use crate::errors::{batch_error, conversion_error};

/// The result of one unordered insert batch.
#[pyclass(module = "polars_mongo._internal", get_all)]
pub struct BatchOutcome {
    /// Number of source rows submitted to this batch.
    attempted: usize,
    /// Number of rows inserted successfully.
    inserted: usize,
    /// `(stream_row_index, server_error_message)` for indexed write failures.
    failures: Vec<(usize, String)>,
}

/// A MongoDB destination with a caller-declared Arrow schema compiled once.
#[pyclass(module = "polars_mongo._internal")]
pub struct MongoBatchWriter {
    connection: Py<MongoConnection>,
    database: String,
    collection: String,
    plan: Arc<SchemaPlan>,
}

#[pymethods]
impl MongoBatchWriter {
    #[new]
    fn new(
        py: Python<'_>,
        connection: Py<MongoConnection>,
        database: &str,
        collection: &str,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            connection,
            database: database.to_owned(),
            collection: collection.to_owned(),
            plan: Arc::new(crate::api::import_schema(py, schema)?),
        })
    }

    /// Insert one frame as an unordered batch.
    ///
    /// `row_offset` is the zero-based ordinal of this batch's first row in the
    /// ordered stream being inserted, not necessarily the source-frame row
    /// after an order-changing lazy operation.
    fn insert_batch(
        &self,
        py: Python<'_>,
        frame: PyDataFrame,
        row_offset: usize,
    ) -> PyResult<BatchOutcome> {
        let documents = encode_documents(&self.plan, &frame.0, &self.collection)
            .map_err(|err| conversion_error(py, &err))?;
        let attempted = documents.len();
        if attempted == 0 {
            return Ok(BatchOutcome {
                attempted: 0,
                inserted: 0,
                failures: Vec::new(),
            });
        }

        // Obtain the client for every operation rather than retaining an Arc:
        // close() must remain the lifecycle authority.
        let client = self.connection.bind(py).borrow().client(py)?;
        let database = self.database.clone();
        let collection = self.collection.clone();
        let result = py.detach(move || {
            client
                .database(&database)
                .collection::<mongodb::bson::Document>(&collection)
                .insert_many(documents)
                .ordered(false)
                .run()
        });

        match result {
            Ok(inserted) => Ok(BatchOutcome {
                attempted,
                inserted: inserted.inserted_ids.len(),
                failures: Vec::new(),
            }),
            Err(error) => match error.kind.as_ref() {
                ErrorKind::InsertMany(insert_many) if insert_many.write_concern_error.is_none() => {
                    let Some(write_errors) = insert_many.write_errors.as_ref() else {
                        return Err(batch_error(
                            py,
                            row_offset,
                            row_offset + attempted,
                            &error.to_string(),
                        ));
                    };
                    let failures = write_errors
                        .iter()
                        .map(|failure| (row_offset + failure.index, failure.message.clone()))
                        .collect::<Vec<_>>();
                    Ok(BatchOutcome {
                        attempted,
                        // `InsertManyError::inserted_ids` is crate-private in
                        // mongodb 3.9, so the acknowledged success count is the
                        // full batch minus the indexed failures: an unordered
                        // insert attempts every document, and this arm is only
                        // reached when no non-row error is present.
                        inserted: attempted - failures.len(),
                        failures,
                    })
                }
                _ => Err(batch_error(
                    py,
                    row_offset,
                    row_offset + attempted,
                    &error.to_string(),
                )),
            },
        }
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BatchOutcome>()?;
    module.add_class::<MongoBatchWriter>()
}
