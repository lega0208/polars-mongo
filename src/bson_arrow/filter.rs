//! Caller dictionaries -> BSON documents, for `find` filters and projections.
//!
//! Values are converted structurally; nothing here inspects a schema. Anything
//! this package cannot represent in BSON is rejected with the located key.

use mongodb::bson::{Bson, Document};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::bson_arrow::schema::{PlanError, PlanResult};
use crate::objectid::{OBJECT_ID_EXT_NAME, object_id_from_hex};

/// Convert a Python mapping into a BSON document.
pub fn dict_to_document(dict: &Bound<'_, PyDict>, path: &str) -> PlanResult<Document> {
    let mut document = Document::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract().map_err(|_| PlanError {
            field_path: path.to_owned(),
            reason: "document keys must be strings".to_owned(),
        })?;
        let child_path = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };
        document.insert(key, py_to_bson(&value, &child_path)?);
    }
    Ok(document)
}

fn unsupported(path: &str, value: &Bound<'_, PyAny>) -> PlanError {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "<unknown>".to_owned());
    PlanError {
        field_path: path.to_owned(),
        reason: format!("value of type '{type_name}' cannot be converted to BSON"),
    }
}

fn py_to_bson(value: &Bound<'_, PyAny>, path: &str) -> PlanResult<Bson> {
    if value.is_none() {
        return Ok(Bson::Null);
    }
    // `bool` is checked before `int`: in Python it *is* an int.
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Bson::Boolean(flag.is_true()));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Bson::String(text.to_string_lossy().into_owned()));
    }
    if let Ok(number) = value.cast::<PyInt>() {
        let as_i64: i64 = number.extract().map_err(|_| PlanError {
            field_path: path.to_owned(),
            reason: "integer does not fit in a BSON Int64".to_owned(),
        })?;
        return Ok(match i32::try_from(as_i64) {
            Ok(small) => Bson::Int32(small),
            Err(_) => Bson::Int64(as_i64),
        });
    }
    if let Ok(number) = value.cast::<PyFloat>() {
        let as_f64: f64 = number.extract().map_err(|_| PlanError {
            field_path: path.to_owned(),
            reason: "float value could not be read".to_owned(),
        })?;
        return Ok(Bson::Double(as_f64));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Bson::Binary(mongodb::bson::Binary {
            subtype: mongodb::bson::spec::BinarySubtype::Generic,
            bytes: bytes.as_bytes().to_vec(),
        }));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(Bson::Document(dict_to_document(dict, path)?));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return sequence_to_bson(list.iter(), path);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return sequence_to_bson(tuple.iter(), path);
    }
    if let Some(oid) = object_id_like(value)? {
        return Ok(Bson::ObjectId(oid));
    }
    Err(unsupported(path, value))
}

fn sequence_to_bson<'py>(
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    path: &str,
) -> PlanResult<Bson> {
    let mut out = Vec::new();
    for (index, item) in items.enumerate() {
        out.push(py_to_bson(&item, &format!("{path}.{index}"))?);
    }
    Ok(Bson::Array(out))
}

/// Recognize an ObjectId written as `bson.ObjectId` or as this package's
/// extension scalar, both of which render as a 24-character hex string.
fn object_id_like(value: &Bound<'_, PyAny>) -> PlanResult<Option<mongodb::bson::oid::ObjectId>> {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    if type_name != "ObjectId" && type_name != OBJECT_ID_EXT_NAME {
        return Ok(None);
    }
    let rendered = value.str().map(|s| s.to_string_lossy().into_owned());
    Ok(rendered
        .ok()
        .and_then(|hex| object_id_from_hex(&hex))
        .map(crate::objectid::object_id_from_bytes))
}

#[cfg(test)]
mod tests {
    use mongodb::bson::doc;

    use super::*;

    fn convert(py: Python<'_>, source: &str) -> PlanResult<Document> {
        let dict = py
            .eval(&std::ffi::CString::new(source).unwrap(), None, None)
            .unwrap();
        let dict = dict.cast::<PyDict>().unwrap();
        dict_to_document(dict, "")
    }

    #[test]
    fn dict_filter_to_bson_document() {
        Python::initialize();
        Python::attach(|py| {
            let document = convert(
                py,
                "{'a': 1, 'b': 'two', 'c': 1.5, 'd': True, 'e': None, \
                 'f': {'$gt': 3}, 'g': [1, 'x'], 'h': b'\\x00\\x01'}",
            )
            .unwrap();

            assert_eq!(document.get("a"), Some(&Bson::Int32(1)));
            assert_eq!(document.get("b"), Some(&Bson::String("two".into())));
            assert_eq!(document.get("c"), Some(&Bson::Double(1.5)));
            assert_eq!(document.get("d"), Some(&Bson::Boolean(true)));
            assert_eq!(document.get("e"), Some(&Bson::Null));
            assert_eq!(document.get("f"), Some(&Bson::Document(doc! {"$gt": 3})));
            assert_eq!(
                document.get("g"),
                Some(&Bson::Array(vec![Bson::Int32(1), Bson::String("x".into())]))
            );
            assert!(matches!(document.get("h"), Some(Bson::Binary(_))));
        });
    }

    #[test]
    fn projection_dict_keeps_inclusion_flags() {
        Python::initialize();
        Python::attach(|py| {
            let document = convert(py, "{'_id': 0, 'name': 1, 'nested.field': 1}").unwrap();
            assert_eq!(document.get("_id"), Some(&Bson::Int32(0)));
            assert_eq!(document.get("nested.field"), Some(&Bson::Int32(1)));
        });
    }

    #[test]
    fn large_integers_widen_to_int64() {
        Python::initialize();
        Python::attach(|py| {
            let document = convert(py, "{'big': 2**40}").unwrap();
            assert_eq!(document.get("big"), Some(&Bson::Int64(1 << 40)));
        });
    }

    #[test]
    fn unsupported_values_are_rejected_with_their_path() {
        Python::initialize();
        Python::attach(|py| {
            let err = convert(py, "{'outer': {'inner': set()}}").unwrap_err();
            assert_eq!(err.field_path, "outer.inner");
            assert!(err.reason.contains("cannot be converted to BSON"));
        });
    }
}
