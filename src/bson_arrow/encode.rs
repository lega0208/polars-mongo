//! Arrow -> BSON, driven entirely by the `FieldPlan`.
//!
//! The declared type picks the BSON tag; values never do. `float64` writes a
//! Double and `BsonDecimal128Type` writes a Decimal128 - the same value written
//! under two declarations lands on two different tags, by design.

use std::str::FromStr;

use mongodb::bson::spec::BinarySubtype;
use mongodb::bson::{Binary, Bson, DateTime, Decimal128, Document, Timestamp};
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{DataType, TimeUnit};
use polars_core::series::{IntoSeries, Series};

use crate::bson_arrow::decode::{ConversionError, DecodeResult};
use crate::bson_arrow::schema::{FieldPlan, PlanType, SchemaPlan};
use crate::objectid::object_id_from_bytes;

fn error(
    plan: &FieldPlan,
    collection: &str,
    row: usize,
    reason: impl Into<String>,
) -> Box<ConversionError> {
    Box::new(ConversionError {
        document_id: format!("<row {row}>"),
        field_path: plan.path.clone(),
        declared_type: format!("{:?}", plan.plan_type),
        bson_type: "<encoding>".to_owned(),
        collection: collection.to_owned(),
        reason: reason.into(),
    })
}

/// Encode a DataFrame into one BSON document per row, following the plan.
pub fn encode_documents(
    plan: &SchemaPlan,
    frame: &DataFrame,
    collection: &str,
) -> DecodeResult<Vec<Document>> {
    let mut columns: Vec<Vec<Bson>> = Vec::with_capacity(plan.fields.len());
    for field in &plan.fields {
        let series = frame
            .column(&field.name)
            .map_err(|err| {
                error(
                    field,
                    collection,
                    0,
                    format!("the frame has no column for this field: {err}"),
                )
            })?
            .as_materialized_series();
        columns.push(encode_column(field, series, collection)?);
    }

    let mut documents = Vec::with_capacity(frame.height());
    for row in 0..frame.height() {
        let mut document = Document::new();
        for (field, column) in plan.fields.iter().zip(columns.iter()) {
            document.insert(field.name.clone(), column[row].clone());
        }
        documents.push(document);
    }
    Ok(documents)
}

/// Strip the identity wrapper: extension dtypes are storage plus a name.
fn storage_of(series: &Series) -> PolarsResult<Series> {
    if series.dtype().is_extension() {
        Ok(series.ext()?.storage().clone())
    } else {
        Ok(series.clone())
    }
}

fn null_or_error(plan: &FieldPlan, collection: &str, row: usize) -> DecodeResult<Bson> {
    if plan.nullable {
        Ok(Bson::Null)
    } else {
        Err(error(
            plan,
            collection,
            row,
            "the value is null and the declared type is not nullable",
        ))
    }
}

/// Encode one column into per-row BSON values.
pub fn encode_column(
    plan: &FieldPlan,
    series: &Series,
    collection: &str,
) -> DecodeResult<Vec<Bson>> {
    encode_column_where(plan, series, collection, None)
}

/// Rows outside `active` belong to a null ancestor: their physical slots hold
/// filler, so they are neither encoded nor checked for nullability.
fn is_active(active: Option<&[bool]>, row: usize) -> bool {
    active.is_none_or(|mask| mask[row])
}

fn encode_column_where(
    plan: &FieldPlan,
    series: &Series,
    collection: &str,
    active: Option<&[bool]>,
) -> DecodeResult<Vec<Bson>> {
    let series = storage_of(series)
        .map_err(|err| error(plan, collection, 0, format!("unreadable column: {err}")))?;
    let length = series.len();
    let mut out = Vec::with_capacity(length);

    match plan.plan_type {
        PlanType::Int32 => {
            let series = cast(&series, &DataType::Int32, plan, collection)?;
            let values = series.i32().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(number) => Bson::Int32(number),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::Int64 => {
            let series = cast(&series, &DataType::Int64, plan, collection)?;
            let values = series.i64().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(number) => Bson::Int64(number),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::Float64 | PlanType::Decimal128 => {
            let series = cast(&series, &DataType::Float64, plan, collection)?;
            let values = series.f64().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(number) if plan.plan_type == PlanType::Decimal128 => {
                        Bson::Decimal128(decimal_from_f64(number, plan, collection, row)?)
                    }
                    Some(number) => Bson::Double(number),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::Boolean => {
            let series = cast(&series, &DataType::Boolean, plan, collection)?;
            let values = series.bool().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(flag) => Bson::Boolean(flag),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::String => {
            let series = cast(&series, &DataType::String, plan, collection)?;
            let values = series.str().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(text) => Bson::String(text.to_owned()),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::Binary => {
            let series = cast(&series, &DataType::Binary, plan, collection)?;
            let values = series.binary().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(bytes) => Bson::Binary(Binary {
                        subtype: BinarySubtype::Generic,
                        bytes: bytes.to_vec(),
                    }),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::DateTimeMs | PlanType::Timestamp => {
            let series = cast(
                &series,
                &DataType::Datetime(TimeUnit::Milliseconds, None),
                plan,
                collection,
            )?;
            let physical = series.to_physical_repr();
            let values = physical.i64().unwrap();
            for (row, value) in values.iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(millis) if plan.plan_type == PlanType::Timestamp => {
                        // A BSON Timestamp is seconds plus an increment; the
                        // increment is synthesized as 0 because a DateTime-like
                        // value has none to preserve.
                        Bson::Timestamp(Timestamp {
                            time: u32::try_from(millis.div_euclid(1_000)).map_err(|_| {
                                error(
                                    plan,
                                    collection,
                                    row,
                                    "value is out of range for a BSON Timestamp",
                                )
                            })?,
                            increment: 0,
                        })
                    }
                    Some(millis) => Bson::DateTime(DateTime::from_millis(millis)),
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::ObjectId => {
            let values = series.array().map_err(|err| {
                error(plan, collection, 0, format!("not ObjectId storage: {err}"))
            })?;
            for (row, value) in values.amortized_iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(bytes) => {
                        let bytes = bytes.as_ref().u8().map_err(|err| {
                            error(plan, collection, row, format!("not 12 bytes: {err}"))
                        })?;
                        if bytes.len() != 12 || bytes.null_count() > 0 {
                            return Err(error(
                                plan,
                                collection,
                                row,
                                "an ObjectId must be exactly 12 non-null bytes",
                            ));
                        }
                        let mut raw = [0u8; 12];
                        for (slot, byte) in raw.iter_mut().zip(bytes.into_no_null_iter()) {
                            *slot = byte;
                        }
                        Bson::ObjectId(object_id_from_bytes(raw))
                    }
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
        PlanType::Struct => {
            let values = series
                .struct_()
                .map_err(|err| error(plan, collection, 0, format!("not a struct: {err}")))?;
            let fields = values.fields_as_series();
            let is_null = values.clone().into_series().is_null();
            // A null parent carries no values: its children's physical slots are
            // filler, so they must not be encoded or nullability-checked.
            let child_active: Vec<bool> = (0..length)
                .map(|row| is_active(active, row) && !is_null.get(row).unwrap_or(false))
                .collect();
            let mut child_columns = Vec::with_capacity(plan.children.len());
            for child_plan in &plan.children {
                let child = fields
                    .iter()
                    .find(|series| series.name().as_str() == child_plan.name)
                    .ok_or_else(|| {
                        error(child_plan, collection, 0, "the struct has no such field")
                    })?;
                child_columns.push(encode_column_where(
                    child_plan,
                    child,
                    collection,
                    Some(&child_active),
                )?);
            }
            for row in 0..length {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                if is_null.get(row).unwrap_or(false) {
                    out.push(null_or_error(plan, collection, row)?);
                    continue;
                }
                let mut document = Document::new();
                for (child_plan, column) in plan.children.iter().zip(child_columns.iter()) {
                    document.insert(child_plan.name.clone(), column[row].clone());
                }
                out.push(Bson::Document(document));
            }
        }
        PlanType::List => {
            let child_plan = &plan.children[0];
            let values = series
                .list()
                .map_err(|err| error(plan, collection, 0, format!("not a list: {err}")))?;
            for (row, value) in values.amortized_iter().enumerate() {
                if !is_active(active, row) {
                    out.push(Bson::Null);
                    continue;
                }
                out.push(match value {
                    Some(inner) => {
                        let inner = inner.as_ref().clone();
                        Bson::Array(encode_column(child_plan, &inner, collection)?)
                    }
                    None => null_or_error(plan, collection, row)?,
                });
            }
        }
    }

    Ok(out)
}

fn cast(
    series: &Series,
    dtype: &DataType,
    plan: &FieldPlan,
    collection: &str,
) -> DecodeResult<Series> {
    if series.dtype() == dtype {
        return Ok(series.clone());
    }
    series.strict_cast(dtype).map_err(|err| {
        error(
            plan,
            collection,
            0,
            format!(
                "column dtype {:?} does not match the declared type: {err}",
                series.dtype()
            ),
        )
    })
}

fn decimal_from_f64(
    value: f64,
    plan: &FieldPlan,
    collection: &str,
    row: usize,
) -> DecodeResult<Decimal128> {
    // The decimal rendering of the f64 is exactly what the caller can observe
    // from the column; precision beyond f64 was never present.
    Decimal128::from_str(&format!("{value}")).map_err(|err| {
        error(
            plan,
            collection,
            row,
            format!("value {value} is not representable as a BSON Decimal128: {err}"),
        )
    })
}
