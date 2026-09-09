//! Python-facing entry points for the schema foundation and the ObjectId
//! transport.

use polars_arrow::ffi::{ArrowSchema, import_field_from_c};
use polars_core::series::Series;
use polars_utils::pl_str::PlSmallStr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use pyo3_polars::PySeries;

use crate::bson_arrow::filter::dict_to_document;
use crate::bson_arrow::schema::{FieldPlan, PlanError, SchemaPlan};
use crate::errors::schema_error;
use crate::objectid::{
    extension_names, large_list_of, null_object_id_array, object_id_array, object_id_from_hex,
    object_id_to_hex, struct_of_object_id,
};

fn to_py_err(py: Python<'_>, err: PlanError) -> PyErr {
    schema_error(py, &err.field_path, &err.reason)
}

/// Import a `pyarrow.Schema` across the Arrow C schema boundary, exactly once.
pub fn import_schema(py: Python<'_>, schema: &Bound<'_, PyAny>) -> PyResult<SchemaPlan> {
    let mut ffi_schema = Box::new(ArrowSchema::empty());
    let address = (&mut *ffi_schema as *mut ArrowSchema) as usize;
    schema.call_method1("_export_to_c", (address,))?;
    let field = unsafe { import_field_from_c(&ffi_schema) }
        .map_err(|err| schema_error(py, "<root>", &format!("{err}")))?;
    SchemaPlan::from_arrow_struct(&field).map_err(|err| to_py_err(py, err))
}

fn describe(plan: &FieldPlan) -> (String, String, bool, Option<String>) {
    (
        plan.path.clone(),
        format!("{:?}", plan.plan_type),
        plan.nullable,
        plan.bson_target.map(|target| format!("{target:?}")),
    )
}

/// Compile the caller's pyarrow schema and return the resolved plan.
///
/// Raises `MongoSchemaError` for an unsupported or excluded declaration at any
/// depth, before any connection is touched.
#[pyfunction]
pub fn compile_schema_plan<'py>(
    py: Python<'py>,
    schema: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let plan = import_schema(py, schema)?;
    let mut described = Vec::new();
    plan.walk(|field| described.push(describe(field)));
    let mut rows = Vec::with_capacity(described.len());
    for (path, plan_type, nullable, target) in described {
        rows.push(PyTuple::new(
            py,
            [
                path,
                plan_type,
                nullable.to_string(),
                target.unwrap_or_default(),
            ],
        )?);
    }
    PyList::new(py, rows)
}

/// Convert a caller dictionary into a BSON document, returned as its extended
/// JSON rendering so tests can assert on it without a second BSON encoder.
#[pyfunction]
pub fn filter_to_extended_json(py: Python<'_>, filter: &Bound<'_, PyDict>) -> PyResult<String> {
    let document = dict_to_document(filter, "").map_err(|err| to_py_err(py, err))?;
    Ok(document.to_string())
}

/// Hex rendering of a 12-byte ObjectId.
#[pyfunction]
pub fn object_id_bytes_to_hex(py: Python<'_>, bytes: [u8; 12]) -> PyResult<String> {
    let _ = py;
    Ok(object_id_to_hex(bytes))
}

/// Parse a 24-character hex string into its 12 bytes.
#[pyfunction]
pub fn object_id_hex_to_bytes(py: Python<'_>, hex: &str) -> PyResult<[u8; 12]> {
    object_id_from_hex(hex)
        .ok_or_else(|| schema_error(py, "object_id", &format!("'{hex}' is not an ObjectId")))
}

const OID_A: &str = "507f1f77bcf86cd799439011";
const OID_B: &str = "5f2b4c8e1a2b3c4d5e6f7a8b";

/// Build a Series in Rust whose ObjectId nodes carry the extension identity.
///
/// This is the Rust half of the Phase 5 transport probe: the Python half
/// re-imports the Series and asks for its identity back, closing the
/// builder -> Series -> Python -> Rust loop that F3 does not cover.
#[pyfunction]
pub fn object_id_probe_series(py: Python<'_>, kind: &str) -> PyResult<PySeries> {
    let a = object_id_from_hex(OID_A).unwrap();
    let b = object_id_from_hex(OID_B).unwrap();

    let array = match kind {
        "flat" => object_id_array(&[Some(a), Some(b)]).boxed(),
        "flat_with_null" => object_id_array(&[Some(a), None]).boxed(),
        "struct" => struct_of_object_id(&[Some(a), Some(b)], None),
        "struct_null_parent" => {
            let validity = [true, false].into_iter().collect();
            struct_of_object_id(&[Some(a), None], Some(validity))
        }
        "struct_null_child" => struct_of_object_id(&[Some(a), None], None),
        "list" => large_list_of(
            object_id_array(&[Some(a), Some(b)]).boxed(),
            vec![0, 1, 2],
            None,
            "element",
        ),
        "list_empty" => large_list_of(object_id_array(&[]).boxed(), vec![0, 0, 0], None, "element"),
        "list_null_parent" => {
            let validity = [true, false].into_iter().collect();
            large_list_of(
                object_id_array(&[Some(a)]).boxed(),
                vec![0, 1, 1],
                Some(validity),
                "element",
            )
        }
        "list_struct" => large_list_of(
            struct_of_object_id(&[Some(a), Some(b)], None),
            vec![0, 2],
            None,
            "element",
        ),
        "all_null" => null_object_id_array(3),
        other => {
            return Err(schema_error(
                py,
                "object_id_probe_series",
                &format!("unknown probe kind '{other}'"),
            ));
        }
    };

    let series = Series::from_arrow(PlSmallStr::from_static("probe"), array)
        .map_err(|err| schema_error(py, "object_id_probe_series", &format!("{err}")))?;
    Ok(PySeries(series))
}

/// Report the extension names carried by a Series' dtype, outermost first.
///
/// Called with a Series that made the full round trip, this proves identity
/// survived at every level rather than only at the top.
#[pyfunction]
pub fn series_extension_names(series: PySeries) -> Vec<String> {
    extension_names(series.0.dtype())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_schema_plan, module)?)?;
    module.add_function(wrap_pyfunction!(filter_to_extended_json, module)?)?;
    module.add_function(wrap_pyfunction!(object_id_bytes_to_hex, module)?)?;
    module.add_function(wrap_pyfunction!(object_id_hex_to_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(object_id_probe_series, module)?)?;
    module.add_function(wrap_pyfunction!(series_extension_names, module)?)?;
    Ok(())
}
