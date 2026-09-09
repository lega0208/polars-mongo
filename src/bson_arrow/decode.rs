//! BSON -> Arrow, driven entirely by the `FieldPlan`.
//!
//! Nothing here infers a target type from data: the plan already chose the
//! converter. The converter inspects each observed BSON tag only to coerce it
//! onto the declared type or to reject it with a located error.

use mongodb::bson::spec::{BinarySubtype, ElementType};
use mongodb::bson::{RawBsonRef, RawDocument};
use polars_arrow::array::{
    Array, BinaryArray, BooleanArray, PrimitiveArray, StructArray, Utf8Array,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType, ExtensionType, Field as ArrowField, TimeUnit};
use polars_arrow::offset::OffsetsBuffer;
use polars_core::series::Series;
use polars_utils::pl_str::PlSmallStr;

use crate::bson_arrow::schema::{
    BSON_DECIMAL128_EXT_NAME, BSON_TIMESTAMP_EXT_NAME, FieldPlan, PlanType, SchemaPlan,
};
use crate::objectid::object_id_arrow_dtype;

/// A located conversion failure, carrying the full `MongoConversionError`
/// attribute set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionError {
    pub document_id: String,
    pub field_path: String,
    pub declared_type: String,
    pub bson_type: String,
    pub collection: String,
    pub reason: String,
}

impl ConversionError {
    pub fn message(&self) -> String {
        format!(
            "document {} field '{}' declared {} but found BSON {} in collection '{}': {}",
            self.document_id,
            self.field_path,
            self.declared_type,
            self.bson_type,
            self.collection,
            self.reason,
        )
    }
}

/// Boxed so the `Ok` path of every conversion result stays small.
pub type DecodeResult<T> = Result<T, Box<ConversionError>>;

/// Identity of the document currently being decoded, for located errors.
#[derive(Debug, Clone)]
pub struct DocumentContext {
    pub document_id: String,
    pub collection: String,
}

impl DocumentContext {
    pub fn new(document_id: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            document_id: document_id.into(),
            collection: collection.into(),
        }
    }
}

/// The `_id` of a document, or `<no _id>` when absent, plus the batch ordinal.
pub fn document_identity(document: &RawDocument, ordinal: usize) -> String {
    match document.get("_id") {
        Ok(Some(RawBsonRef::ObjectId(oid))) => {
            format!("_id={} (batch row {ordinal})", oid.to_hex())
        }
        Ok(Some(other)) => format!("_id={} (batch row {ordinal})", render_value(other)),
        _ => format!("<no _id> (batch row {ordinal})"),
    }
}

fn render_value(value: RawBsonRef<'_>) -> String {
    match value {
        RawBsonRef::String(text) => text.to_owned(),
        RawBsonRef::Int32(number) => number.to_string(),
        RawBsonRef::Int64(number) => number.to_string(),
        RawBsonRef::Double(number) => number.to_string(),
        RawBsonRef::Boolean(flag) => flag.to_string(),
        RawBsonRef::ObjectId(oid) => oid.to_hex(),
        other => format!("{:?}", other.element_type()),
    }
}

fn tag_name(element: ElementType) -> &'static str {
    match element {
        ElementType::Double => "Double",
        ElementType::String => "String",
        ElementType::EmbeddedDocument => "EmbeddedDocument",
        ElementType::Array => "Array",
        ElementType::Binary => "Binary",
        ElementType::Undefined => "Undefined",
        ElementType::ObjectId => "ObjectId",
        ElementType::Boolean => "Boolean",
        ElementType::DateTime => "DateTime",
        ElementType::Null => "Null",
        ElementType::RegularExpression => "RegularExpression",
        ElementType::DbPointer => "DbPointer",
        ElementType::JavaScriptCode => "JavaScriptCode",
        ElementType::Symbol => "Symbol",
        ElementType::JavaScriptCodeWithScope => "JavaScriptCodeWithScope",
        ElementType::Int32 => "Int32",
        ElementType::Timestamp => "Timestamp",
        ElementType::Int64 => "Int64",
        ElementType::Decimal128 => "Decimal128",
        ElementType::MinKey => "MinKey",
        ElementType::MaxKey => "MaxKey",
    }
}

/// BSON types this package refuses in either direction (AC-19, value half).
fn is_excluded(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::RegularExpression
            | ElementType::JavaScriptCode
            | ElementType::JavaScriptCodeWithScope
            | ElementType::Symbol
            | ElementType::MinKey
            | ElementType::MaxKey
            | ElementType::DbPointer
    )
}

fn declared_name(plan: &FieldPlan) -> String {
    format!("{:?}", plan.plan_type)
}

fn error(
    context: &DocumentContext,
    plan: &FieldPlan,
    element: Option<ElementType>,
    reason: impl Into<String>,
) -> Box<ConversionError> {
    Box::new(ConversionError {
        document_id: context.document_id.clone(),
        field_path: plan.path.clone(),
        declared_type: declared_name(plan),
        bson_type: element.map(tag_name).unwrap_or("Missing").to_owned(),
        collection: context.collection.clone(),
        reason: reason.into(),
    })
}

fn extension_dtype(name: &'static str, inner: ArrowDataType) -> ArrowDataType {
    ArrowDataType::Extension(Box::new(ExtensionType {
        name: PlSmallStr::from_static(name),
        inner,
        metadata: Some(PlSmallStr::EMPTY),
    }))
}

/// A column under construction, mirroring one `FieldPlan` node.
enum Builder {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    String(Vec<Option<String>>),
    Binary(Vec<Option<Vec<u8>>>),
    /// Milliseconds since the epoch, for both DateTime and Timestamp targets.
    Millis(Vec<Option<i64>>),
    ObjectId(Vec<Option<[u8; 12]>>),
    Struct {
        children: Vec<Builder>,
        validity: Vec<bool>,
    },
    List {
        child: Box<Builder>,
        offsets: Vec<i64>,
        validity: Vec<bool>,
    },
}

impl Builder {
    fn new(plan: &FieldPlan) -> Self {
        match plan.plan_type {
            PlanType::Int32 => Builder::Int32(Vec::new()),
            PlanType::Int64 => Builder::Int64(Vec::new()),
            PlanType::Float64 | PlanType::Decimal128 => Builder::Float64(Vec::new()),
            PlanType::Boolean => Builder::Boolean(Vec::new()),
            PlanType::String => Builder::String(Vec::new()),
            PlanType::Binary => Builder::Binary(Vec::new()),
            PlanType::DateTimeMs | PlanType::Timestamp => Builder::Millis(Vec::new()),
            PlanType::ObjectId => Builder::ObjectId(Vec::new()),
            PlanType::Struct => Builder::Struct {
                children: plan.children.iter().map(Builder::new).collect(),
                validity: Vec::new(),
            },
            PlanType::List => Builder::List {
                child: Box::new(Builder::new(&plan.children[0])),
                offsets: vec![0],
                validity: Vec::new(),
            },
        }
    }

    fn push_null(&mut self) {
        match self {
            Builder::Int32(values) => values.push(None),
            Builder::Int64(values) => values.push(None),
            Builder::Float64(values) => values.push(None),
            Builder::Boolean(values) => values.push(None),
            Builder::String(values) => values.push(None),
            Builder::Binary(values) => values.push(None),
            Builder::Millis(values) => values.push(None),
            Builder::ObjectId(values) => values.push(None),
            Builder::Struct { children, validity } => {
                for child in children.iter_mut() {
                    child.push_null();
                }
                validity.push(false);
            }
            Builder::List {
                offsets, validity, ..
            } => {
                offsets.push(*offsets.last().expect("offsets start at zero"));
                validity.push(false);
            }
        }
    }
}

/// Decode a batch of documents into one Series per top-level field.
pub fn decode_documents(
    plan: &SchemaPlan,
    documents: &[&RawDocument],
    collection: &str,
) -> DecodeResult<Vec<Series>> {
    let mut builders: Vec<Builder> = plan.fields.iter().map(Builder::new).collect();

    for (ordinal, document) in documents.iter().enumerate() {
        let context = DocumentContext::new(document_identity(document, ordinal), collection);
        for (field, builder) in plan.fields.iter().zip(builders.iter_mut()) {
            let value = document.get(field.name.as_str()).map_err(|err| {
                error(&context, field, None, format!("malformed document: {err}"))
            })?;
            push_value(builder, field, value, &context)?;
        }
    }

    plan.fields
        .iter()
        .zip(builders)
        .map(|(field, builder)| {
            let array = finish(builder, field);
            Series::from_arrow(PlSmallStr::from_str(&field.name), array).map_err(|err| {
                Box::new(ConversionError {
                    document_id: "<batch>".to_owned(),
                    field_path: field.path.clone(),
                    declared_type: declared_name(field),
                    bson_type: "<batch>".to_owned(),
                    collection: collection.to_owned(),
                    reason: format!("could not build a Series: {err}"),
                })
            })
        })
        .collect()
}

fn push_value(
    builder: &mut Builder,
    plan: &FieldPlan,
    value: Option<RawBsonRef<'_>>,
    context: &DocumentContext,
) -> DecodeResult<()> {
    let Some(value) = value else {
        // A missing field is a null; whether that is legal is the plan's call.
        if plan.nullable {
            builder.push_null();
            return Ok(());
        }
        return Err(error(
            context,
            plan,
            None,
            "the field is missing and the declared type is not nullable",
        ));
    };

    let element = value.element_type();
    if element == ElementType::Null || element == ElementType::Undefined {
        if plan.nullable {
            builder.push_null();
            return Ok(());
        }
        return Err(error(
            context,
            plan,
            Some(element),
            "the value is null and the declared type is not nullable",
        ));
    }

    if is_excluded(element) {
        return Err(error(
            context,
            plan,
            Some(element),
            format!(
                "BSON {} is not supported by polars-mongo",
                tag_name(element)
            ),
        ));
    }

    match (builder, &plan.plan_type) {
        (Builder::Int32(values), PlanType::Int32) => match value {
            RawBsonRef::Int32(number) => values.push(Some(number)),
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::Int64(values), PlanType::Int64) => match value {
            RawBsonRef::Int32(number) => values.push(Some(i64::from(number))),
            RawBsonRef::Int64(number) => values.push(Some(number)),
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::Float64(values), PlanType::Float64 | PlanType::Decimal128) => match value {
            RawBsonRef::Double(number) => values.push(Some(number)),
            RawBsonRef::Int32(number) => values.push(Some(f64::from(number))),
            RawBsonRef::Int64(number) => values.push(Some(number as f64)),
            RawBsonRef::Decimal128(decimal) => {
                let text = decimal.to_string();
                let parsed = text.parse::<f64>().map_err(|_| {
                    error(
                        context,
                        plan,
                        Some(element),
                        format!("Decimal128 value '{text}' is not representable as float64"),
                    )
                })?;
                values.push(Some(parsed));
            }
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::Boolean(values), PlanType::Boolean) => match value {
            RawBsonRef::Boolean(flag) => values.push(Some(flag)),
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::String(values), PlanType::String) => match value {
            RawBsonRef::String(text) => values.push(Some(text.to_owned())),
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::Binary(values), PlanType::Binary) => match value {
            RawBsonRef::Binary(binary) if binary.subtype == BinarySubtype::Generic => {
                values.push(Some(binary.bytes.to_vec()))
            }
            RawBsonRef::Binary(binary) => {
                return Err(error(
                    context,
                    plan,
                    Some(element),
                    format!(
                        "only BSON Binary subtype 0 is supported, found subtype {:?}",
                        binary.subtype
                    ),
                ));
            }
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::Millis(values), PlanType::DateTimeMs | PlanType::Timestamp) => match value {
            RawBsonRef::DateTime(datetime) => values.push(Some(datetime.timestamp_millis())),
            RawBsonRef::Timestamp(timestamp) => {
                // The increment is discarded: neither target can carry it back.
                values.push(Some(i64::from(timestamp.time) * 1_000))
            }
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::ObjectId(values), PlanType::ObjectId) => match value {
            RawBsonRef::ObjectId(oid) => values.push(Some(oid.bytes())),
            _ => return Err(mismatch(context, plan, element)),
        },
        (Builder::Struct { children, validity }, PlanType::Struct) => match value {
            RawBsonRef::Document(document) => {
                for (child_plan, child_builder) in plan.children.iter().zip(children.iter_mut()) {
                    let child_value = document.get(child_plan.name.as_str()).map_err(|err| {
                        error(
                            context,
                            child_plan,
                            None,
                            format!("malformed embedded document: {err}"),
                        )
                    })?;
                    push_value(child_builder, child_plan, child_value, context)?;
                }
                validity.push(true);
            }
            _ => return Err(mismatch(context, plan, element)),
        },
        (
            Builder::List {
                child,
                offsets,
                validity,
            },
            PlanType::List,
        ) => match value {
            RawBsonRef::Array(array) => {
                let child_plan = &plan.children[0];
                let mut length = *offsets.last().expect("offsets start at zero");
                for item in array.into_iter() {
                    let item = item.map_err(|err| {
                        error(context, child_plan, None, format!("malformed array: {err}"))
                    })?;
                    push_value(child, child_plan, Some(item), context)?;
                    length += 1;
                }
                offsets.push(length);
                validity.push(true);
            }
            _ => return Err(mismatch(context, plan, element)),
        },
        _ => return Err(mismatch(context, plan, element)),
    }

    Ok(())
}

fn mismatch(
    context: &DocumentContext,
    plan: &FieldPlan,
    element: ElementType,
) -> Box<ConversionError> {
    error(
        context,
        plan,
        Some(element),
        format!(
            "BSON {} cannot be converted to the declared type",
            tag_name(element)
        ),
    )
}

fn validity_bitmap(validity: Vec<bool>) -> Option<Bitmap> {
    if validity.iter().all(|flag| *flag) {
        None
    } else {
        Some(validity.into_iter().collect())
    }
}

fn finish(builder: Builder, plan: &FieldPlan) -> Box<dyn Array> {
    match builder {
        Builder::Int32(values) => PrimitiveArray::<i32>::from(values).boxed(),
        Builder::Int64(values) => PrimitiveArray::<i64>::from(values).boxed(),
        Builder::Float64(values) => {
            let array = PrimitiveArray::<f64>::from(values);
            if plan.plan_type == PlanType::Decimal128 {
                array
                    .to(extension_dtype(
                        BSON_DECIMAL128_EXT_NAME,
                        ArrowDataType::Float64,
                    ))
                    .boxed()
            } else {
                array.boxed()
            }
        }
        Builder::Boolean(values) => BooleanArray::from(values).boxed(),
        Builder::String(values) => Utf8Array::<i64>::from(values).boxed(),
        Builder::Binary(values) => BinaryArray::<i64>::from(values).boxed(),
        Builder::Millis(values) => {
            let array = PrimitiveArray::<i64>::from(values)
                .to(ArrowDataType::Timestamp(TimeUnit::Millisecond, None));
            if plan.plan_type == PlanType::Timestamp {
                array
                    .to(extension_dtype(
                        BSON_TIMESTAMP_EXT_NAME,
                        ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                    ))
                    .boxed()
            } else {
                array.boxed()
            }
        }
        Builder::ObjectId(values) => {
            let mut bytes = Vec::with_capacity(values.len() * 12);
            let mut validity = Vec::with_capacity(values.len());
            for value in &values {
                match value {
                    Some(raw) => {
                        bytes.extend_from_slice(raw);
                        validity.push(true);
                    }
                    None => {
                        bytes.extend(std::iter::repeat_n(0u8, 12));
                        validity.push(false);
                    }
                }
            }
            let inner = PrimitiveArray::<u8>::from_vec(bytes).boxed();
            polars_arrow::array::FixedSizeListArray::new(
                object_id_arrow_dtype(),
                values.len(),
                inner,
                validity_bitmap(validity),
            )
            .boxed()
        }
        Builder::Struct { children, validity } => {
            let arrays: Vec<Box<dyn Array>> = children
                .into_iter()
                .zip(plan.children.iter())
                .map(|(child, child_plan)| finish(child, child_plan))
                .collect();
            let fields: Vec<ArrowField> = arrays
                .iter()
                .zip(plan.children.iter())
                .map(|(array, child_plan)| {
                    ArrowField::new(
                        PlSmallStr::from_str(&child_plan.name),
                        array.dtype().clone(),
                        child_plan.nullable,
                    )
                })
                .collect();
            let length = validity.len();
            StructArray::new(
                ArrowDataType::Struct(fields),
                length,
                arrays,
                validity_bitmap(validity),
            )
            .boxed()
        }
        Builder::List {
            child,
            offsets,
            validity,
        } => {
            let child_plan = &plan.children[0];
            let values = finish(*child, child_plan);
            let dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(
                PlSmallStr::from_str(&child_plan.name),
                values.dtype().clone(),
                child_plan.nullable,
            )));
            let offsets = OffsetsBuffer::try_from(offsets).expect("monotonic offsets");
            polars_arrow::array::ListArray::<i64>::new(
                dtype,
                offsets,
                values,
                validity_bitmap(validity),
            )
            .boxed()
        }
    }
}
