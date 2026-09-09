//! The `FieldPlan`: the single shared authority for physical type, nullability
//! and extension identity.
//!
//! The caller's pyarrow schema is imported across the Arrow C schema boundary
//! once per call and compiled into a recursive plan. Nothing here inspects data:
//! the plan picks the converter ahead of time, and the converter later inspects
//! each observed BSON tag only to coerce or reject.

use std::fmt;

use polars_arrow::datatypes::{ArrowDataType, Field, IntegerType, TimeUnit};

use crate::objectid::OBJECT_ID_EXT_NAME;

/// Extension name for the caller-selectable BSON `Decimal128` target.
pub const BSON_DECIMAL128_EXT_NAME: &str = "polars_mongo.bson_decimal128";
/// Extension name for the caller-selectable BSON `Timestamp` target.
pub const BSON_TIMESTAMP_EXT_NAME: &str = "polars_mongo.bson_timestamp";

/// BSON types this package does not support in either direction.
///
/// Declaring any of them - at any depth - is rejected at plan-build time,
/// before a connection is touched (AC-19, declaration half).
pub const EXCLUDED_EXT_NAMES: [&str; 6] = [
    "polars_mongo.regex",
    "polars_mongo.javascript",
    "polars_mongo.symbol",
    "polars_mongo.min_key",
    "polars_mongo.max_key",
    "polars_mongo.db_pointer",
];

/// A located schema rejection. Rendered as `MongoSchemaError` at the Python boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError {
    pub field_path: String,
    pub reason: String,
}

impl PlanError {
    fn new(field_path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field_path: field_path.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field_path, self.reason)
    }
}

pub type PlanResult<T> = Result<T, PlanError>;

/// The BSON type a field is written as, when the caller selected one explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BsonTarget {
    ObjectId,
    Decimal128,
    Timestamp,
}

/// The resolved converter for one field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanType {
    Int32,
    Int64,
    Float64,
    Boolean,
    String,
    Binary,
    /// BSON DateTime; milliseconds since the epoch. BSON DateTime has no
    /// increment field - only `BsonTarget::Timestamp` can carry one.
    DateTimeMs,
    ObjectId,
    Decimal128,
    Timestamp,
    Struct,
    List,
}

/// One node of the recursive schema plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPlan {
    /// The field's own name.
    pub name: String,
    /// The full dotted path from the document root, used in located errors.
    pub path: String,
    pub plan_type: PlanType,
    pub nullable: bool,
    pub children: Vec<FieldPlan>,
    /// Set when the caller selected a BSON target through an extension type.
    pub bson_target: Option<BsonTarget>,
}

impl FieldPlan {
    /// Depth-first iteration over this node and its descendants.
    pub fn walk(&self, visit: &mut impl FnMut(&FieldPlan)) {
        visit(self);
        for child in &self.children {
            child.walk(visit);
        }
    }
}

/// The compiled plan for a whole document schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaPlan {
    pub fields: Vec<FieldPlan>,
}

impl SchemaPlan {
    /// Compile the caller's schema, imported as a struct field, into a plan.
    pub fn from_arrow_struct(field: &Field) -> PlanResult<Self> {
        let ArrowDataType::Struct(fields) = field.dtype().clone() else {
            return Err(PlanError::new(
                "<root>",
                "the caller schema must import as an Arrow struct",
            ));
        };
        Self::from_fields(&fields)
    }

    /// Compile a plan directly from a list of top-level fields.
    pub fn from_fields(fields: &[Field]) -> PlanResult<Self> {
        Ok(Self {
            fields: fields
                .iter()
                .map(|field| compile_field(field, ""))
                .collect::<PlanResult<Vec<_>>>()?,
        })
    }

    pub fn walk(&self, mut visit: impl FnMut(&FieldPlan)) {
        for field in &self.fields {
            field.walk(&mut visit);
        }
    }
}

fn join_path(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_owned()
    } else {
        format!("{parent}.{name}")
    }
}

fn compile_field(field: &Field, parent_path: &str) -> PlanResult<FieldPlan> {
    let name = field.name.to_string();
    let path = join_path(parent_path, &name);
    compile_dtype(&name, &path, field.dtype(), field.is_nullable)
}

fn compile_dtype(
    name: &str,
    path: &str,
    dtype: &ArrowDataType,
    nullable: bool,
) -> PlanResult<FieldPlan> {
    if let ArrowDataType::Extension(ext) = dtype {
        return compile_extension(name, path, ext.name.as_str(), &ext.inner, nullable);
    }

    let (plan_type, children) = match dtype {
        ArrowDataType::Int32 => (PlanType::Int32, vec![]),
        ArrowDataType::Int64 => (PlanType::Int64, vec![]),
        ArrowDataType::Float64 => (PlanType::Float64, vec![]),
        ArrowDataType::Boolean => (PlanType::Boolean, vec![]),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            (PlanType::String, vec![])
        }
        ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView => {
            (PlanType::Binary, vec![])
        }
        ArrowDataType::Timestamp(TimeUnit::Millisecond, timezone) => {
            if timezone.is_some() {
                return Err(PlanError::new(
                    path,
                    "timestamps with a timezone cannot map onto BSON DateTime",
                ));
            }
            (PlanType::DateTimeMs, vec![])
        }
        ArrowDataType::Timestamp(unit, _) => {
            return Err(PlanError::new(
                path,
                format!(
                    "timestamps must be declared in milliseconds to map onto BSON DateTime, \
                     got {unit:?}"
                ),
            ));
        }
        ArrowDataType::Struct(fields) => {
            let children = fields
                .iter()
                .map(|child| compile_field(child, path))
                .collect::<PlanResult<Vec<_>>>()?;
            (PlanType::Struct, children)
        }
        ArrowDataType::List(inner) | ArrowDataType::LargeList(inner) => {
            let child = compile_field(inner, path)?;
            (PlanType::List, vec![child])
        }
        ArrowDataType::FixedSizeList(_, _) => {
            return Err(PlanError::new(
                path,
                "bare fixed-size list declarations are not supported",
            ));
        }
        ArrowDataType::Dictionary(IntegerType::UInt32, inner, _) => {
            return compile_dtype(name, path, inner, nullable);
        }
        other => {
            return Err(PlanError::new(
                path,
                format!("unsupported Arrow type {other:?}"),
            ));
        }
    };

    Ok(FieldPlan {
        name: name.to_owned(),
        path: path.to_owned(),
        plan_type,
        nullable,
        children,
        bson_target: None,
    })
}

fn compile_extension(
    name: &str,
    path: &str,
    ext_name: &str,
    inner: &ArrowDataType,
    nullable: bool,
) -> PlanResult<FieldPlan> {
    if EXCLUDED_EXT_NAMES.contains(&ext_name) {
        return Err(PlanError::new(
            path,
            format!("BSON type '{ext_name}' is not supported by polars-mongo"),
        ));
    }

    let (plan_type, target, expected_storage) = match ext_name {
        OBJECT_ID_EXT_NAME => (
            PlanType::ObjectId,
            BsonTarget::ObjectId,
            "a 12-element fixed-size list of uint8",
        ),
        BSON_DECIMAL128_EXT_NAME => (PlanType::Decimal128, BsonTarget::Decimal128, "float64"),
        BSON_TIMESTAMP_EXT_NAME => (
            PlanType::Timestamp,
            BsonTarget::Timestamp,
            "timestamp in milliseconds",
        ),
        other => {
            return Err(PlanError::new(
                path,
                format!("unknown extension type '{other}'"),
            ));
        }
    };

    let storage_ok = match target {
        BsonTarget::ObjectId => matches!(
            inner,
            ArrowDataType::FixedSizeList(field, 12)
                if matches!(field.dtype(), ArrowDataType::UInt8)
        ),
        BsonTarget::Decimal128 => matches!(inner, ArrowDataType::Float64),
        BsonTarget::Timestamp => {
            matches!(inner, ArrowDataType::Timestamp(TimeUnit::Millisecond, _))
        }
    };
    if !storage_ok {
        return Err(PlanError::new(
            path,
            format!("extension type '{ext_name}' requires storage {expected_storage}"),
        ));
    }

    if let ArrowDataType::Timestamp(_, timezone) = inner {
        if timezone.is_some() {
            return Err(PlanError::new(
                path,
                "timestamps with a timezone cannot map onto BSON",
            ));
        }
    }

    Ok(FieldPlan {
        name: name.to_owned(),
        path: path.to_owned(),
        plan_type,
        nullable,
        children: vec![],
        bson_target: Some(target),
    })
}

#[cfg(test)]
mod tests {
    use polars_arrow::datatypes::{ArrowDataType, ExtensionType, Field, TimeUnit};

    use super::*;

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

    fn plan_of(fields: Vec<Field>) -> PlanResult<SchemaPlan> {
        SchemaPlan::from_fields(&fields)
    }

    #[test]
    fn compiles_scalar_fields_with_nullability() {
        let plan = plan_of(vec![
            Field::new("a".into(), ArrowDataType::Int32, false),
            Field::new("b".into(), ArrowDataType::Float64, true),
            Field::new("c".into(), ArrowDataType::Utf8, true),
            Field::new("d".into(), ArrowDataType::Boolean, false),
            Field::new("e".into(), ArrowDataType::Binary, true),
            Field::new(
                "f".into(),
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
        ])
        .unwrap();

        let types: Vec<_> = plan.fields.iter().map(|f| f.plan_type.clone()).collect();
        assert_eq!(
            types,
            vec![
                PlanType::Int32,
                PlanType::Float64,
                PlanType::String,
                PlanType::Boolean,
                PlanType::Binary,
                PlanType::DateTimeMs,
            ]
        );
        let nullability: Vec<_> = plan.fields.iter().map(|f| f.nullable).collect();
        assert_eq!(nullability, vec![false, true, true, false, true, true]);
        assert!(plan.fields.iter().all(|f| f.bson_target.is_none()));
    }

    #[test]
    fn assigns_dotted_paths_at_depth() {
        let inner_struct = ArrowDataType::Struct(vec![Field::new(
            "item_id".into(),
            extension(OBJECT_ID_EXT_NAME, object_id_storage()),
            true,
        )]);
        let list_of_struct =
            ArrowDataType::LargeList(Box::new(Field::new("element".into(), inner_struct, true)));
        let plan = plan_of(vec![Field::new("orders".into(), list_of_struct, true)]).unwrap();

        let mut paths = Vec::new();
        plan.walk(|field| paths.push(field.path.clone()));
        assert_eq!(
            paths,
            vec![
                "orders".to_string(),
                "orders.element".to_string(),
                "orders.element.item_id".to_string(),
            ]
        );
    }

    #[test]
    fn propagates_nullability_through_struct_and_list() {
        let struct_dtype = ArrowDataType::Struct(vec![
            Field::new("required".into(), ArrowDataType::Int64, false),
            Field::new("optional".into(), ArrowDataType::Int64, true),
        ]);
        let plan = plan_of(vec![
            Field::new("s".into(), struct_dtype.clone(), true),
            Field::new(
                "l".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    struct_dtype,
                    false,
                ))),
                false,
            ),
        ])
        .unwrap();

        let s = &plan.fields[0];
        assert!(s.nullable);
        assert!(!s.children[0].nullable);
        assert!(s.children[1].nullable);

        let l = &plan.fields[1];
        assert!(!l.nullable);
        let element = &l.children[0];
        assert!(!element.nullable);
        assert!(!element.children[0].nullable);
        assert!(element.children[1].nullable);
    }

    #[test]
    fn marks_bson_targets() {
        let plan = plan_of(vec![
            Field::new(
                "_id".into(),
                extension(OBJECT_ID_EXT_NAME, object_id_storage()),
                false,
            ),
            Field::new(
                "amount".into(),
                extension(BSON_DECIMAL128_EXT_NAME, ArrowDataType::Float64),
                true,
            ),
            Field::new(
                "ts".into(),
                extension(
                    BSON_TIMESTAMP_EXT_NAME,
                    ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                ),
                true,
            ),
            Field::new("plain".into(), ArrowDataType::Float64, true),
        ])
        .unwrap();

        let targets: Vec<_> = plan.fields.iter().map(|f| f.bson_target).collect();
        assert_eq!(
            targets,
            vec![
                Some(BsonTarget::ObjectId),
                Some(BsonTarget::Decimal128),
                Some(BsonTarget::Timestamp),
                None,
            ]
        );
    }

    #[test]
    fn rejects_extension_storage_mismatch() {
        let err = plan_of(vec![Field::new(
            "_id".into(),
            extension(OBJECT_ID_EXT_NAME, ArrowDataType::Utf8),
            false,
        )])
        .unwrap_err();
        assert_eq!(err.field_path, "_id");
        assert!(err.reason.contains("requires storage"));
    }

    #[test]
    fn rejects_excluded_declaration_at_any_depth() {
        for excluded in EXCLUDED_EXT_NAMES {
            let nested = ArrowDataType::Struct(vec![Field::new(
                "inner".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    extension(excluded, ArrowDataType::Utf8),
                    true,
                ))),
                true,
            )]);
            let err = plan_of(vec![Field::new("outer".into(), nested, true)]).unwrap_err();
            assert_eq!(err.field_path, "outer.inner.element");
            assert!(err.reason.contains(excluded), "{}", err.reason);
            assert!(err.reason.contains("not supported"));
        }
    }

    #[test]
    fn rejects_unknown_extension_names() {
        let err = plan_of(vec![Field::new(
            "x".into(),
            extension("some.other.extension", ArrowDataType::Utf8),
            true,
        )])
        .unwrap_err();
        assert!(err.reason.contains("unknown extension type"));
    }

    #[test]
    fn rejects_non_millisecond_timestamps() {
        let err = plan_of(vec![Field::new(
            "t".into(),
            ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        )])
        .unwrap_err();
        assert!(err.reason.contains("milliseconds"));
    }

    #[test]
    fn rejects_timezone_timestamps_at_every_depth() {
        let timestamp = ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
        let cases = vec![
            (
                "top-level",
                vec![Field::new("timestamp".into(), timestamp.clone(), true)],
                "timestamp",
            ),
            (
                "struct",
                vec![Field::new(
                    "outer".into(),
                    ArrowDataType::Struct(vec![Field::new(
                        "timestamp".into(),
                        timestamp.clone(),
                        true,
                    )]),
                    true,
                )],
                "outer.timestamp",
            ),
            (
                "list",
                vec![Field::new(
                    "outer".into(),
                    ArrowDataType::LargeList(Box::new(Field::new(
                        "element".into(),
                        timestamp,
                        true,
                    ))),
                    true,
                )],
                "outer.element",
            ),
        ];

        for (location, fields, path) in cases {
            let err = plan_of(fields).unwrap_err();
            assert_eq!(err.field_path, path, "{location}");
            assert!(
                err.reason.contains("timezone"),
                "{location}: {}",
                err.reason
            );
        }
    }

    #[test]
    fn rejects_bare_fixed_size_lists_at_every_depth() {
        let fixed_size_list = || {
            ArrowDataType::FixedSizeList(
                Box::new(Field::new("element".into(), ArrowDataType::Int64, true)),
                2,
            )
        };
        let cases = vec![
            (
                "top-level",
                vec![Field::new("values".into(), fixed_size_list(), true)],
                "values",
            ),
            (
                "struct",
                vec![Field::new(
                    "outer".into(),
                    ArrowDataType::Struct(vec![Field::new(
                        "values".into(),
                        fixed_size_list(),
                        true,
                    )]),
                    true,
                )],
                "outer.values",
            ),
            (
                "list",
                vec![Field::new(
                    "outer".into(),
                    ArrowDataType::LargeList(Box::new(Field::new(
                        "element".into(),
                        fixed_size_list(),
                        true,
                    ))),
                    true,
                )],
                "outer.element",
            ),
        ];

        for (location, fields, path) in cases {
            let err = plan_of(fields).unwrap_err();
            assert_eq!(err.field_path, path, "{location}");
            assert!(
                err.reason.contains("bare fixed-size list"),
                "{location}: {}",
                err.reason
            );
        }
    }
}
