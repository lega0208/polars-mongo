use polars_core::{
    error::{PolarsResult, polars_bail},
    prelude::{
        BooleanChunked, DataType, Field, Int64Chunked, NewChunkedArray, ReshapeDimension,
        StringChunked, TimeUnit, UInt8Chunked,
    },
    series::{IntoSeries, Series},
};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// A MongoDB ObjectId is a 12-byte value, rendered as 24 hex characters.
/// The leading 4 bytes are a big-endian Unix timestamp in seconds.
const OBJECT_ID_HEX_LEN: usize = 24;

fn is_valid_object_id(s: &str) -> bool {
    s.len() == OBJECT_ID_HEX_LEN && s.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Deserialize)]
pub struct TimestampKwargs {
    time_unit: String,
}

fn parse_time_unit(time_unit: &str) -> PolarsResult<TimeUnit> {
    match time_unit {
        "ms" => Ok(TimeUnit::Milliseconds),
        "us" => Ok(TimeUnit::Microseconds),
        "ns" => Ok(TimeUnit::Nanoseconds),
        other => {
            polars_bail!(InvalidOperation: "invalid time_unit '{}', expected one of 'ms', 'us', 'ns'", other)
        }
    }
}

fn seconds_scale(time_unit: TimeUnit) -> i64 {
    match time_unit {
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Nanoseconds => 1_000_000_000,
    }
}

fn timestamp_output(input_fields: &[Field], kwargs: TimestampKwargs) -> PolarsResult<Field> {
    let time_unit = parse_time_unit(&kwargs.time_unit)?;
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Datetime(time_unit, None),
    ))
}

/// Return the ObjectId storage, accepting only the registered extension or its
/// `Array(UInt8, 12)` storage representation.
fn object_id_storage(series: &Series) -> PolarsResult<Series> {
    match series.dtype() {
        DataType::Extension(instance, _)
            if instance.name() == crate::objectid::OBJECT_ID_EXT_NAME =>
        {
            Ok(series.ext()?.storage().clone())
        }
        DataType::Extension(_, _) => {
            polars_bail!(InvalidOperation: "expected ObjectId extension dtype")
        }
        DataType::Array(inner, 12) if inner.as_ref() == &DataType::UInt8 => Ok(series.clone()),
        _ => polars_bail!(
            InvalidOperation:
            "expected ObjectId dtype or Array(UInt8, 12) storage, got {}",
            series.dtype()
        ),
    }
}

/// `True` when the string is a well-formed 24-character hex ObjectId.
#[polars_expr(output_type=Boolean)]
fn is_object_id(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: BooleanChunked = ca.iter().map(|opt| opt.map(is_valid_object_id)).collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Creation timestamp encoded in a binary ObjectId. Null input stays null.
#[polars_expr(output_type_func_with_kwargs=timestamp_output)]
fn object_id_timestamp(inputs: &[Series], kwargs: TimestampKwargs) -> PolarsResult<Series> {
    let series = &inputs[0];
    let storage = object_id_storage(series)?;
    let ca = storage.array()?;
    let time_unit = parse_time_unit(&kwargs.time_unit)?;
    let scale = seconds_scale(time_unit);

    let out: Int64Chunked = ca
        .amortized_iter()
        .map(|opt| {
            let values = opt?;
            let bytes = values.as_ref().u8().ok()?;
            if bytes.null_count() > 0 || bytes.len() != 12 {
                return None;
            }
            let seconds = bytes
                .into_no_null_iter()
                .take(4)
                .fold(0i64, |seconds, byte| (seconds << 8) | i64::from(byte));
            Some(seconds * scale)
        })
        .collect();

    Ok(out
        .with_name(series.name().clone())
        .into_datetime(time_unit, None)
        .into_series())
}

/// Hex rendering of a binary ObjectId column.
///
/// Accepts the ObjectId extension dtype and its bare `Array(UInt8, 12)`
/// storage; both carry the same 12 bytes.
#[polars_expr(output_type=String)]
fn object_id_to_hex(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let storage = object_id_storage(series)?;
    let ca = storage.array()?;
    let mut out: Vec<Option<String>> = Vec::with_capacity(ca.len());
    for opt in ca.amortized_iter() {
        match opt {
            None => out.push(None),
            Some(values) => {
                let bytes = values.as_ref().u8()?;
                if bytes.null_count() > 0 || bytes.len() != 12 {
                    out.push(None);
                    continue;
                }
                let mut raw = [0u8; 12];
                for (slot, value) in raw.iter_mut().zip(bytes.into_no_null_iter()) {
                    *slot = value;
                }
                out.push(Some(crate::objectid::object_id_to_hex(raw)));
            }
        }
    }
    Ok(StringChunked::from_iter_options(series.name().clone(), out.into_iter()).into_series())
}

/// Parse a 24-character hex ObjectId column into its binary storage.
/// Invalid input becomes null.
#[polars_expr(output_type_func=object_id_storage_output)]
fn object_id_from_hex(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let mut builder: Vec<Option<Vec<u8>>> = Vec::with_capacity(ca.len());
    for opt in ca.iter() {
        builder.push(
            opt.and_then(crate::objectid::object_id_from_hex)
                .map(|raw| raw.to_vec()),
        );
    }
    let mut values: Vec<u8> = Vec::with_capacity(builder.len() * 12);
    let mut validity: Vec<bool> = Vec::with_capacity(builder.len());
    for slot in &builder {
        match slot {
            Some(raw) => {
                values.extend_from_slice(raw);
                validity.push(true);
            }
            None => {
                values.extend(std::iter::repeat_n(0u8, 12));
                validity.push(false);
            }
        }
    }
    let flat = UInt8Chunked::from_vec(ca.name().clone(), values).into_series();
    let array =
        flat.reshape_array(&[ReshapeDimension::Infer, ReshapeDimension::new_dimension(12)])?;
    let validity: BooleanChunked =
        BooleanChunked::from_iter_values(ca.name().clone(), validity.into_iter());
    array.zip_with(
        &validity,
        &Series::full_null(ca.name().clone(), ca.len(), array.dtype()),
    )
}

fn object_id_storage_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name().clone(),
        DataType::Array(Box::new(DataType::UInt8), 12),
    ))
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::NamedFrom;

    use super::*;

    #[test]
    fn validates_object_ids() {
        assert!(is_valid_object_id("507f1f77bcf86cd799439011"));
        assert!(!is_valid_object_id("507f1f77bcf86cd79943901"));
        assert!(!is_valid_object_id("zzzf1f77bcf86cd799439011"));
    }

    #[test]
    fn object_id_storage_rejects_utf8_and_accepts_array_storage() {
        let utf8 = Series::new("oid".into(), ["507f1f77bcf86cd799439011"]);
        assert!(object_id_storage(&utf8).is_err());

        let flat = Series::new("oid".into(), vec![0u8; 12]);
        let array = flat
            .reshape_array(&[ReshapeDimension::Infer, ReshapeDimension::new_dimension(12)])
            .expect("12 u8 values reshape into one Array(UInt8, 12) row");
        assert_eq!(
            object_id_storage(&array)
                .expect("storage dtype is accepted")
                .dtype(),
            &DataType::Array(Box::new(DataType::UInt8), 12)
        );
    }
}
