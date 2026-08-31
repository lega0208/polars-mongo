use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// A MongoDB ObjectId is a 12-byte value, rendered as 24 hex characters.
/// The leading 4 bytes are a big-endian Unix timestamp in seconds.
const OBJECT_ID_HEX_LEN: usize = 24;
const OBJECT_ID_TIMESTAMP_HEX_LEN: usize = 8;

fn is_valid_object_id(s: &str) -> bool {
    s.len() == OBJECT_ID_HEX_LEN && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn object_id_epoch_seconds(s: &str) -> Option<i64> {
    if !is_valid_object_id(s) {
        return None;
    }
    i64::from_str_radix(&s[..OBJECT_ID_TIMESTAMP_HEX_LEN], 16).ok()
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

/// `True` when the string is a well-formed 24-character hex ObjectId.
#[polars_expr(output_type=Boolean)]
fn is_object_id(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: BooleanChunked = ca.iter().map(|opt| opt.map(is_valid_object_id)).collect();
    Ok(out.with_name(ca.name().clone()).into_series())
}

/// Creation timestamp encoded in the ObjectId. Invalid input becomes null.
#[polars_expr(output_type_func_with_kwargs=timestamp_output)]
fn object_id_timestamp(inputs: &[Series], kwargs: TimestampKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let time_unit = parse_time_unit(&kwargs.time_unit)?;
    let scale = seconds_scale(time_unit);

    let out: Int64Chunked = ca
        .iter()
        .map(|opt| {
            opt.and_then(object_id_epoch_seconds)
                .map(|secs| secs * scale)
        })
        .collect();

    Ok(out
        .with_name(ca.name().clone())
        .into_datetime(time_unit, None)
        .into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_object_ids() {
        assert!(is_valid_object_id("507f1f77bcf86cd799439011"));
        assert!(!is_valid_object_id("507f1f77bcf86cd79943901"));
        assert!(!is_valid_object_id("zzzf1f77bcf86cd799439011"));
    }

    #[test]
    fn extracts_epoch_seconds() {
        assert_eq!(
            object_id_epoch_seconds("507f1f77bcf86cd799439011"),
            Some(1_350_508_407)
        );
        assert_eq!(object_id_epoch_seconds("nope"), None);
    }
}
