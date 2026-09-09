"""The shared `FieldPlan` at its Python boundary (Phase 4).

The plan is compiled from the caller's pyarrow schema, once per call, and every
rejection happens here - before a connection is touched.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import polars_mongo as pm
from polars_mongo import _internal
from polars_mongo._objectid import (
    BsonDecimal128Type,
    BsonTimestampType,
    ObjectIdType,
)

EXCLUDED_EXT_NAMES = [
    "polars_mongo.regex",
    "polars_mongo.javascript",
    "polars_mongo.symbol",
    "polars_mongo.min_key",
    "polars_mongo.max_key",
    "polars_mongo.db_pointer",
]


class _Excluded(pa.ExtensionType):
    """Stands in for a caller declaring one of the unsupported BSON types."""

    def __init__(self, name: str) -> None:
        self._name = name
        super().__init__(pa.string(), name)

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type: pa.DataType, serialized: bytes) -> _Excluded:
        return cls("polars_mongo.regex")


def plan(schema: pa.Schema) -> list[tuple[str, str, str, str]]:
    return _internal.compile_schema_plan(schema)


def test_resolves_paths_types_and_targets() -> None:
    schema = pa.schema(
        [
            ("_id", ObjectIdType()),
            ("count", pa.int32()),
            ("total", pa.float64()),
            ("amount", BsonDecimal128Type()),
            ("ts", BsonTimestampType()),
            ("when", pa.timestamp("ms")),
            ("orders", pa.list_(pa.struct([("item_id", ObjectIdType())]))),
        ]
    )
    rows = plan(schema)
    by_path = {row[0]: row for row in rows}

    assert by_path["_id"][1] == "ObjectId"
    assert by_path["_id"][3] == "ObjectId"
    assert by_path["count"][1] == "Int32"
    assert by_path["total"][1] == "Float64"
    assert by_path["total"][3] == ""
    assert by_path["amount"][3] == "Decimal128"
    assert by_path["ts"][3] == "Timestamp"
    assert by_path["when"][1] == "DateTimeMs"
    assert by_path["orders"][1] == "List"
    # The dotted path is assigned at depth, which is what located errors report.
    nested = next(path for path in by_path if path.endswith("item_id"))
    assert by_path[nested][1] == "ObjectId"
    assert nested.startswith("orders.")


def test_nullability_is_carried_per_field() -> None:
    schema = pa.schema(
        [
            pa.field("required", pa.int64(), nullable=False),
            pa.field("optional", pa.int64(), nullable=True),
        ]
    )
    rows = {row[0]: row[2] for row in plan(schema)}
    assert rows["required"] == "false"
    assert rows["optional"] == "true"


@pytest.mark.parametrize("ext_name", EXCLUDED_EXT_NAMES)
def test_excluded_declaration_rejected_at_top_level(ext_name: str) -> None:
    schema = pa.schema([("field", _Excluded(ext_name))])
    with pytest.raises(pm.MongoSchemaError) as excinfo:
        plan(schema)
    assert excinfo.value.field_path == "field"
    assert ext_name in str(excinfo.value)


@pytest.mark.parametrize("ext_name", EXCLUDED_EXT_NAMES)
def test_excluded_declaration_rejected_at_depth(ext_name: str) -> None:
    schema = pa.schema(
        [("outer", pa.struct([("inner", pa.list_(_Excluded(ext_name)))]))],
    )
    with pytest.raises(pm.MongoSchemaError) as excinfo:
        plan(schema)
    assert excinfo.value.field_path.startswith("outer.inner.")
    assert ext_name in str(excinfo.value)


def test_unknown_extension_is_rejected() -> None:
    schema = pa.schema([("field", _Excluded("some.other.extension"))])
    with pytest.raises(pm.MongoSchemaError, match="unknown extension type"):
        plan(schema)


def test_unsupported_timestamp_unit_is_rejected() -> None:
    schema = pa.schema([("t", pa.timestamp("us"))])
    with pytest.raises(pm.MongoSchemaError, match="milliseconds"):
        plan(schema)


@pytest.mark.parametrize(
    "dtype",
    [
        pa.timestamp("ms", tz="UTC"),
        pa.list_(pa.int8(), 12),
    ],
)
@pytest.mark.parametrize(
    ("location", "field_path"),
    [
        ("top_level", "value"),
        ("struct", "outer.inner"),
        ("list", "outer.item"),
    ],
)
def test_unrepresentable_declarations_are_rejected_at_every_depth(
    dtype: pa.DataType, location: str, field_path: str
) -> None:
    if location == "top_level":
        schema = pa.schema([("value", dtype)])
    elif location == "struct":
        schema = pa.schema([("outer", pa.struct([("inner", dtype)]))])
    else:
        assert location == "list"
        schema = pa.schema([("outer", pa.list_(pa.field("item", dtype)))])
    with pytest.raises(pm.MongoSchemaError) as excinfo:
        plan(schema)
    assert excinfo.value.field_path == field_path


def test_objectid_extension_transport_remains_a_valid_fixed_size_list() -> None:
    rows = plan(pa.schema([("object_id", ObjectIdType())]))
    assert rows[0][0] == "object_id"
    assert rows[0][1] == "ObjectId"


def test_filter_dict_converts_to_bson() -> None:
    """AC-13, at the Python boundary; the unit half lives in `cargo test`."""
    rendered = _internal.filter_to_extended_json({"a": 1, "b": {"$gt": 2}, "c": ["x", None]})
    assert '"a": 1' in rendered
    assert '"$gt": 2' in rendered
    assert '"c": ["x", null]' in rendered


def test_filter_rejects_unconvertible_values_with_their_path() -> None:
    with pytest.raises(pm.MongoSchemaError) as excinfo:
        _internal.filter_to_extended_json({"outer": {"inner": {1, 2}}})
    assert excinfo.value.field_path == "outer.inner"
