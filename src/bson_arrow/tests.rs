//! Both directions of the section 5.1 mapping, the located-error payload, and
//! the recursive composition rules.
//!
//! Expected tags and values are hand-written; the lossy rows carry their lossy
//! result literally. Byte identity is asserted only for ObjectId.

use mongodb::bson::spec::ElementType;
use mongodb::bson::{Bson, Document, RawDocumentBuf, doc};
use polars_arrow::datatypes::{ArrowDataType, ExtensionType, Field, TimeUnit};
use polars_core::df;
use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType, NamedFrom, StructChunked};
use polars_core::series::{IntoSeries, Series};

use crate::bson_arrow::decode::decode_documents;
use crate::bson_arrow::encode::encode_documents;
use crate::bson_arrow::schema::{BSON_DECIMAL128_EXT_NAME, BSON_TIMESTAMP_EXT_NAME, SchemaPlan};
use crate::objectid::{OBJECT_ID_EXT_NAME, object_id_from_hex};

const COLLECTION: &str = "testdb.things";
const HEX: &str = "507f1f77bcf86cd799439011";

fn extension(name: &str, inner: ArrowDataType) -> ArrowDataType {
    ArrowDataType::Extension(Box::new(ExtensionType {
        name: name.into(),
        inner,
        metadata: None,
    }))
}

fn object_id_storage() -> ArrowDataType {
    ArrowDataType::FixedSizeList(
        Box::new(Field::new("item".into(), ArrowDataType::UInt8, false)),
        12,
    )
}

fn plan(fields: Vec<Field>) -> SchemaPlan {
    SchemaPlan::from_fields(&fields).expect("plan compiles")
}

fn decode(fields: Vec<Field>, documents: Vec<Document>) -> PolarsResult<Vec<Series>> {
    let plan = plan(fields);
    let raw: Vec<RawDocumentBuf> = documents
        .into_iter()
        .map(|document| RawDocumentBuf::from_document(&document).expect("valid document"))
        .collect();
    let refs: Vec<&mongodb::bson::RawDocument> = raw.iter().map(|doc| doc.as_ref()).collect();
    Ok(decode_documents(&plan, &refs, COLLECTION).expect("decode succeeds"))
}

fn decode_err(
    fields: Vec<Field>,
    documents: Vec<Document>,
) -> Box<crate::bson_arrow::decode::ConversionError> {
    let plan = plan(fields);
    let raw: Vec<RawDocumentBuf> = documents
        .into_iter()
        .map(|document| RawDocumentBuf::from_document(&document).expect("valid document"))
        .collect();
    let refs: Vec<&mongodb::bson::RawDocument> = raw.iter().map(|doc| doc.as_ref()).collect();
    decode_documents(&plan, &refs, COLLECTION).expect_err("decode fails")
}

fn encode_one(fields: Vec<Field>, frame: DataFrame) -> Document {
    let plan = plan(fields);
    encode_documents(&plan, &frame, COLLECTION)
        .expect("encode succeeds")
        .remove(0)
}

fn tag_of(document: &Document, key: &str) -> ElementType {
    document.get(key).expect("key present").element_type()
}

// ---------------------------------------------------------------- write side

#[test]
fn writes_float64_as_double_and_decimal128_target_as_decimal128() {
    let frame = df!("value" => [1.5f64]).unwrap();

    let as_double = encode_one(
        vec![Field::new("value".into(), ArrowDataType::Float64, true)],
        frame.clone(),
    );
    assert_eq!(tag_of(&as_double, "value"), ElementType::Double);
    assert_eq!(as_double.get("value"), Some(&Bson::Double(1.5)));

    let as_decimal = encode_one(
        vec![Field::new(
            "value".into(),
            extension(BSON_DECIMAL128_EXT_NAME, ArrowDataType::Float64),
            true,
        )],
        frame,
    );
    assert_eq!(tag_of(&as_decimal, "value"), ElementType::Decimal128);
    let Some(Bson::Decimal128(decimal)) = as_decimal.get("value") else {
        panic!("expected a Decimal128");
    };
    assert_eq!(decimal.to_string(), "1.5");
}

#[test]
fn writes_datetime_and_timestamp_targets_from_the_same_instant() {
    // 1970-01-01T00:00:01Z
    let millis = Series::new("t".into(), [1_000i64])
        .cast(&DataType::Datetime(
            polars_core::datatypes::TimeUnit::Milliseconds,
            None,
        ))
        .unwrap();
    let frame = DataFrame::new(1, vec![millis.into()]).unwrap();

    let as_datetime = encode_one(
        vec![Field::new(
            "t".into(),
            ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        )],
        frame.clone(),
    );
    assert_eq!(tag_of(&as_datetime, "t"), ElementType::DateTime);
    let Some(Bson::DateTime(value)) = as_datetime.get("t") else {
        panic!("expected a DateTime");
    };
    assert_eq!(value.timestamp_millis(), 1_000);

    let as_timestamp = encode_one(
        vec![Field::new(
            "t".into(),
            extension(
                BSON_TIMESTAMP_EXT_NAME,
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            true,
        )],
        frame,
    );
    assert_eq!(tag_of(&as_timestamp, "t"), ElementType::Timestamp);
    let Some(Bson::Timestamp(value)) = as_timestamp.get("t") else {
        panic!("expected a Timestamp");
    };
    assert_eq!(value.time, 1);
    // A BSON DateTime has no increment to preserve, so it is synthesized as 0.
    assert_eq!(value.increment, 0);
}

#[test]
fn writes_the_remaining_scalar_rows_with_their_declared_tags() {
    let frame = df!(
        "i32" => [7i32],
        "i64" => [8i64],
        "flag" => [true],
        "text" => ["hello"],
        "blob" => [&b"\x00\x01"[..]],
    )
    .unwrap();
    let document = encode_one(
        vec![
            Field::new("i32".into(), ArrowDataType::Int32, true),
            Field::new("i64".into(), ArrowDataType::Int64, true),
            Field::new("flag".into(), ArrowDataType::Boolean, true),
            Field::new("text".into(), ArrowDataType::Utf8, true),
            Field::new("blob".into(), ArrowDataType::Binary, true),
        ],
        frame,
    );

    assert_eq!(tag_of(&document, "i32"), ElementType::Int32);
    assert_eq!(tag_of(&document, "i64"), ElementType::Int64);
    assert_eq!(tag_of(&document, "flag"), ElementType::Boolean);
    assert_eq!(tag_of(&document, "text"), ElementType::String);
    assert_eq!(tag_of(&document, "blob"), ElementType::Binary);
    assert_eq!(document.get("i32"), Some(&Bson::Int32(7)));
    assert_eq!(document.get("i64"), Some(&Bson::Int64(8)));
}

#[test]
fn writes_object_id_bytes_exactly() {
    let bytes = object_id_from_hex(HEX).unwrap();
    let series = Series::from_arrow(
        "oid".into(),
        crate::objectid::object_id_array(&[Some(bytes)]).boxed(),
    )
    .unwrap();
    let height = series.len();
    let frame = DataFrame::new(height, vec![series.into()]).unwrap();
    let document = encode_one(
        vec![Field::new(
            "oid".into(),
            extension(OBJECT_ID_EXT_NAME, object_id_storage()),
            true,
        )],
        frame,
    );
    assert_eq!(tag_of(&document, "oid"), ElementType::ObjectId);
    let Some(Bson::ObjectId(oid)) = document.get("oid") else {
        panic!("expected an ObjectId");
    };
    assert_eq!(oid.bytes(), bytes);
    assert_eq!(oid.to_hex(), HEX);
}

#[test]
fn rejects_a_null_in_a_non_nullable_declaration_on_write() {
    let frame = df!("value" => [None::<i64>]).unwrap();
    let plan = plan(vec![Field::new(
        "value".into(),
        ArrowDataType::Int64,
        false,
    )]);
    let err = encode_documents(&plan, &frame, COLLECTION).expect_err("must reject");
    assert_eq!(err.field_path, "value");
    assert!(err.reason.contains("not nullable"));
}

// ----------------------------------------------------------------- read side

#[test]
fn reads_widening_int32_to_int64_and_int_to_float() {
    let series = decode(
        vec![
            Field::new("i".into(), ArrowDataType::Int64, true),
            Field::new("f".into(), ArrowDataType::Float64, true),
        ],
        vec![doc! {"i": 3i32, "f": 4i32}, doc! {"i": 5i64, "f": 6.5}],
    )
    .unwrap();

    assert_eq!(
        series[0]
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![3, 5]
    );
    assert_eq!(
        series[1]
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        vec![4.0, 6.5]
    );
}

#[test]
fn reads_decimal128_into_float64_with_its_literal_lossy_value() {
    let series = decode(
        vec![Field::new("d".into(), ArrowDataType::Float64, true)],
        vec![
            doc! {"d": Bson::Decimal128("2.25".parse().unwrap())},
            doc! {"d": Bson::Decimal128("0.1".parse().unwrap())},
        ],
    )
    .unwrap();
    let values: Vec<f64> = series[0].f64().unwrap().into_no_null_iter().collect();
    assert_eq!(values[0], 2.25);
    // The nearest f64 to 0.1, written out literally as the accepted lossy value.
    assert_eq!(
        values[1],
        // The nearest f64 to 0.1; the decimal expansion is written out in the
        // comment above because clippy rejects the literal's full precision.
        f64::from_bits(0x3FB999999999999A)
    );
}

#[test]
fn reads_timestamp_into_millis_discarding_the_increment() {
    let series = decode(
        vec![Field::new(
            "t".into(),
            ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        )],
        vec![doc! {"t": Bson::Timestamp(mongodb::bson::Timestamp {
            time: 1_700_000_000,
            increment: 7,
        })}],
    )
    .unwrap();
    let physical = series[0].to_physical_repr();
    assert_eq!(physical.i64().unwrap().get(0), Some(1_700_000_000_000));
}

#[test]
fn reads_object_id_bytes_exactly() {
    let bytes = object_id_from_hex(HEX).unwrap();
    let series = decode(
        vec![Field::new(
            "oid".into(),
            extension(OBJECT_ID_EXT_NAME, object_id_storage()),
            true,
        )],
        vec![doc! {"oid": mongodb::bson::oid::ObjectId::from_bytes(bytes)}],
    )
    .unwrap();
    let storage = series[0].ext().unwrap().storage().clone();
    let values: Vec<u8> = storage
        .array()
        .unwrap()
        .get_as_series(0)
        .unwrap()
        .u8()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(values.as_slice(), &bytes[..]);
}

#[test]
fn located_error_payload_carries_every_attribute() {
    let err = decode_err(
        vec![Field::new("count".into(), ArrowDataType::Int32, true)],
        vec![doc! {"_id": mongodb::bson::oid::ObjectId::from_bytes(
            object_id_from_hex(HEX).unwrap()
        ), "count": "not a number"}],
    );
    assert_eq!(err.field_path, "count");
    assert_eq!(err.declared_type, "Int32");
    assert_eq!(err.bson_type, "String");
    assert_eq!(err.collection, COLLECTION);
    assert!(err.document_id.contains(HEX));
    assert!(err.message().contains("count"));
    assert!(err.message().contains(COLLECTION));
}

#[test]
fn missing_field_is_null_when_nullable_and_an_error_when_not() {
    let series = decode(
        vec![Field::new("maybe".into(), ArrowDataType::Int64, true)],
        vec![doc! {"other": 1}],
    )
    .unwrap();
    assert_eq!(series[0].null_count(), 1);

    let err = decode_err(
        vec![Field::new("needed".into(), ArrowDataType::Int64, false)],
        vec![doc! {"other": 1}],
    );
    assert_eq!(err.bson_type, "Missing");
    assert!(err.reason.contains("missing"));

    let err = decode_err(
        vec![Field::new("needed".into(), ArrowDataType::Int64, false)],
        vec![doc! {"needed": Bson::Null}],
    );
    assert!(err.reason.contains("null"));
}

#[test]
fn rejects_binary_subtypes_other_than_zero() {
    let err = decode_err(
        vec![Field::new("blob".into(), ArrowDataType::Binary, true)],
        vec![doc! {"blob": Bson::Binary(mongodb::bson::Binary {
            subtype: mongodb::bson::spec::BinarySubtype::Uuid,
            bytes: vec![0; 16],
        })}],
    );
    assert!(err.reason.contains("subtype 0"));
}

// ------------------------------------------------------- recursive (Phase 6b)

fn nested_fields() -> Vec<Field> {
    vec![
        Field::new(
            "profile".into(),
            ArrowDataType::Struct(vec![
                Field::new("age".into(), ArrowDataType::Int64, true),
                Field::new(
                    "oid".into(),
                    extension(OBJECT_ID_EXT_NAME, object_id_storage()),
                    true,
                ),
            ]),
            true,
        ),
        Field::new(
            "tags".into(),
            ArrowDataType::LargeList(Box::new(Field::new(
                "element".into(),
                ArrowDataType::Utf8,
                true,
            ))),
            true,
        ),
        Field::new(
            "orders".into(),
            ArrowDataType::LargeList(Box::new(Field::new(
                "element".into(),
                ArrowDataType::Struct(vec![Field::new(
                    "item_id".into(),
                    extension(OBJECT_ID_EXT_NAME, object_id_storage()),
                    true,
                )]),
                true,
            ))),
            true,
        ),
    ]
}

#[test]
fn recursive_rules_apply_at_every_level() {
    let bytes = object_id_from_hex(HEX).unwrap();
    let oid = mongodb::bson::oid::ObjectId::from_bytes(bytes);
    let series = decode(
        nested_fields(),
        vec![
            doc! {
                "profile": {"age": 30i32, "oid": oid},
                "tags": ["a", "b"],
                "orders": [{"item_id": oid}, {"item_id": oid}],
            },
            doc! {
                "profile": Bson::Null,
                "tags": [],
                "orders": [],
            },
        ],
    )
    .unwrap();

    // Widening applies inside a struct exactly as it does at the top level.
    let profile = series[0].struct_().unwrap();
    let age = profile.field_by_name("age").unwrap();
    assert_eq!(age.i64().unwrap().get(0), Some(30));
    assert_eq!(series[0].null_count(), 1);

    // Extension identity is preserved at depth by the single transport.
    let names = crate::objectid::extension_names(series[0].dtype());
    assert_eq!(names, vec![OBJECT_ID_EXT_NAME]);
    let names = crate::objectid::extension_names(series[2].dtype());
    assert_eq!(names, vec![OBJECT_ID_EXT_NAME]);

    // Empty lists are empty, not null.
    let tags = series[1].list().unwrap();
    assert_eq!(tags.get_as_series(1).unwrap().len(), 0);
}

#[test]
fn rejects_an_excluded_bson_value_at_depth_with_its_dotted_path() {
    let err = decode_err(
        nested_fields(),
        vec![doc! {
            "profile": {"age": 1i32, "oid": Bson::MaxKey},
            "tags": [],
            "orders": [],
        }],
    );
    assert_eq!(err.field_path, "profile.oid");
    assert_eq!(err.bson_type, "MaxKey");
    assert!(err.reason.contains("not supported"));
}

#[test]
fn rejects_a_missing_non_nullable_child_inside_a_null_parent() {
    let fields = vec![Field::new(
        "profile".into(),
        ArrowDataType::Struct(vec![Field::new("age".into(), ArrowDataType::Int64, false)]),
        true,
    )];
    // A present parent with the child missing is an error...
    let err = decode_err(fields.clone(), vec![doc! {"profile": {}}]);
    assert_eq!(err.field_path, "profile.age");

    // ...while a null parent is simply a null row.
    let series = decode(fields, vec![doc! {"profile": Bson::Null}]).unwrap();
    assert_eq!(series[0].null_count(), 1);
}

#[test]
fn rejects_a_type_mismatch_at_depth_with_its_dotted_path() {
    let err = decode_err(
        nested_fields(),
        vec![doc! {
            "profile": {"age": 1i32, "oid": Bson::Null},
            "tags": [],
            "orders": [{"item_id": "not an object id"}],
        }],
    );
    assert_eq!(err.field_path, "orders.element.item_id");
    assert_eq!(err.bson_type, "String");
}

#[test]
fn nested_round_trip_is_logically_equal() {
    let bytes = object_id_from_hex(HEX).unwrap();
    let oid = mongodb::bson::oid::ObjectId::from_bytes(bytes);
    let source = vec![
        doc! {
            "profile": {"age": 30i64, "oid": oid},
            "tags": ["a", "b"],
            "orders": [{"item_id": oid}],
        },
        doc! {
            "profile": {"age": 31i64, "oid": Bson::Null},
            "tags": [],
            "orders": [],
        },
    ];

    let fields = nested_fields();
    let series = decode(fields.clone(), source.clone()).unwrap();
    let columns: Vec<Column> = series.into_iter().map(Column::from).collect();
    let frame = DataFrame::new(columns[0].len(), columns).unwrap();
    let encoded = encode_documents(&plan(fields), &frame, COLLECTION).unwrap();

    assert_eq!(encoded, source);
}

#[test]
fn canonical_mappings_property_round_trip_logically_at_every_shape() {
    // The lossy Decimal128 and BSON Timestamp coercions deliberately remain in
    // their hand-written example tests above. This generator contains exactly
    // the lossless canonical scalar rows from section 5.1.
    let mut state = 0xC0FFEE_u64;
    for round in 0..32 {
        for (name, dtype, value) in canonical_mapping_rows(&mut state) {
            assert_round_trip(
                vec![Field::new(name.into(), dtype.clone(), true)],
                vec![Document::from_iter([(name.to_owned(), value.clone())])],
                &format!("top-level {name}, round {round}"),
            );

            assert_round_trip(
                vec![Field::new(
                    "container".into(),
                    ArrowDataType::Struct(vec![Field::new(name.into(), dtype.clone(), true)]),
                    true,
                )],
                vec![Document::from_iter([(
                    "container".to_owned(),
                    Bson::Document(Document::from_iter([(name.to_owned(), value.clone())])),
                )])],
                &format!("struct {name}, round {round}"),
            );

            assert_round_trip(
                vec![Field::new(
                    "values".into(),
                    ArrowDataType::LargeList(Box::new(Field::new(
                        "element".into(),
                        dtype.clone(),
                        true,
                    ))),
                    true,
                )],
                vec![Document::from_iter([(
                    "values".to_owned(),
                    Bson::Array(vec![value.clone(), value.clone()]),
                )])],
                &format!("list {name}, round {round}"),
            );

            assert_round_trip(
                vec![Field::new(
                    "rows".into(),
                    ArrowDataType::LargeList(Box::new(Field::new(
                        "element".into(),
                        ArrowDataType::Struct(vec![Field::new(name.into(), dtype, true)]),
                        true,
                    ))),
                    true,
                )],
                vec![Document::from_iter([(
                    "rows".to_owned(),
                    Bson::Array(vec![
                        Bson::Document(Document::from_iter([(name.to_owned(), value.clone())])),
                        Bson::Document(Document::from_iter([(name.to_owned(), value)])),
                    ]),
                )])],
                &format!("list<struct> {name}, round {round}"),
            );
        }
    }
}

fn assert_round_trip(fields: Vec<Field>, source: Vec<Document>, context: &str) {
    let series = decode(fields.clone(), source.clone()).unwrap();
    let columns: Vec<Column> = series.into_iter().map(Column::from).collect();
    let frame = DataFrame::new(columns[0].len(), columns).unwrap();
    let encoded = encode_documents(&plan(fields), &frame, COLLECTION).unwrap();
    assert_eq!(encoded, source, "round trip failed for {context}");
}

fn canonical_mapping_rows(state: &mut u64) -> Vec<(&'static str, ArrowDataType, Bson)> {
    let next = |state: &mut u64| {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    };
    let seed = next(state);
    let object_id_bytes: [u8; 12] = seed.to_le_bytes().repeat(2)[..12]
        .try_into()
        .expect("twelve bytes");

    vec![
        ("i32", ArrowDataType::Int32, Bson::Int32(seed as i32)),
        ("i64", ArrowDataType::Int64, Bson::Int64(seed as i64)),
        (
            "f64",
            ArrowDataType::Float64,
            Bson::Double((seed % 1_000_000) as f64 / 100.0),
        ),
        ("bool", ArrowDataType::Boolean, Bson::Boolean(seed & 1 == 0)),
        (
            "str",
            ArrowDataType::Utf8,
            Bson::String(format!("value-{seed:016x}")),
        ),
        (
            "bin",
            ArrowDataType::Binary,
            Bson::Binary(mongodb::bson::Binary {
                subtype: mongodb::bson::spec::BinarySubtype::Generic,
                bytes: seed.to_le_bytes().to_vec(),
            }),
        ),
        (
            "dt",
            ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
            Bson::DateTime(mongodb::bson::DateTime::from_millis(
                (seed % 4_102_444_800) as i64 * 1_000,
            )),
        ),
        (
            "oid",
            extension(OBJECT_ID_EXT_NAME, object_id_storage()),
            Bson::ObjectId(mongodb::bson::oid::ObjectId::from_bytes(object_id_bytes)),
        ),
    ]
}

#[test]
fn a_null_struct_parent_does_not_validate_its_non_nullable_children() {
    // The children's physical slots under a null parent hold filler, not
    // values, so nullability must be enforced only under a live parent.
    let fields = vec![Field::new(
        "profile".into(),
        ArrowDataType::Struct(vec![
            Field::new("age".into(), ArrowDataType::Int64, false),
            Field::new("nick".into(), ArrowDataType::Utf8, false),
        ]),
        true,
    )];
    let source = vec![
        doc! {"profile": {"age": 30i64, "nick": "a"}},
        doc! {"profile": Bson::Null},
        doc! {"profile": {"age": 31i64, "nick": "b"}},
    ];

    let series = decode(fields.clone(), source.clone()).unwrap();
    let columns: Vec<Column> = series.into_iter().map(Column::from).collect();
    let frame = DataFrame::new(columns[0].len(), columns).unwrap();
    let encoded = encode_documents(&plan(fields), &frame, COLLECTION).unwrap();

    assert_eq!(encoded, source);
}

#[test]
fn a_null_list_of_struct_parent_does_not_validate_its_children() {
    let fields = vec![Field::new(
        "orders".into(),
        ArrowDataType::LargeList(Box::new(Field::new(
            "element".into(),
            ArrowDataType::Struct(vec![Field::new("qty".into(), ArrowDataType::Int64, false)]),
            true,
        ))),
        true,
    )];
    let source = vec![
        doc! {"orders": [{"qty": 2i64}]},
        doc! {"orders": Bson::Null},
        doc! {"orders": []},
    ];

    let series = decode(fields.clone(), source.clone()).unwrap();
    let columns: Vec<Column> = series.into_iter().map(Column::from).collect();
    let frame = DataFrame::new(columns[0].len(), columns).unwrap();
    let encoded = encode_documents(&plan(fields), &frame, COLLECTION).unwrap();

    assert_eq!(encoded, source);
}

#[test]
fn a_live_parent_still_enforces_its_non_nullable_children_on_write() {
    // The relaxation above must not weaken the rule under a valid parent.
    let fields = vec![Field::new(
        "profile".into(),
        ArrowDataType::Struct(vec![Field::new("age".into(), ArrowDataType::Int64, false)]),
        true,
    )];
    let inner = StructChunked::from_series(
        "profile".into(),
        2,
        [Series::new("age".into(), [Some(1i64), None])].iter(),
    )
    .unwrap()
    .into_series();
    let frame = DataFrame::new(2, vec![inner.into()]).unwrap();

    let err = encode_documents(&plan(fields), &frame, COLLECTION).expect_err("must reject");
    assert_eq!(err.field_path, "profile.age");
    assert!(err.reason.contains("not nullable"));
}
