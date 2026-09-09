"""ObjectId identity: the extension types (AC-20), hex round trip (AC-23), and
the nested transport matrix that F3 does not cover on its own (Phase 5 probe).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
import pytest

from polars_mongo import _internal
from polars_mongo._objectid import (
    OBJECT_ID_EXT_NAME,
    BsonDecimal128,
    BsonDecimal128Type,
    BsonTimestamp,
    BsonTimestampType,
    ObjectId,
    ObjectIdType,
)

if TYPE_CHECKING:
    from warnings import WarningMessage

HEX_A = "507f1f77bcf86cd799439011"
HEX_B = "5f2b4c8e1a2b3c4d5e6f7a8b"


def test_pyarrow_extension_type_importable_and_storage_shape() -> None:
    """AC-20."""
    ext = ObjectIdType()
    assert ext.extension_name == OBJECT_ID_EXT_NAME
    assert ext.storage_type == pa.list_(pa.uint8(), 12)
    assert pa.schema([("_id", ext)]).field("_id").type == ext


def test_polars_extension_type_shares_the_name_and_storage() -> None:
    dtype = ObjectId()
    assert dtype.ext_name() == OBJECT_ID_EXT_NAME
    assert dtype.ext_storage() == pl.Array(pl.UInt8, 12)
    assert pl.get_extension_type(OBJECT_ID_EXT_NAME) is ObjectId


@pytest.mark.parametrize("kind", ["flat", "struct", "list", "list_struct"])
def test_reader_objectid_schema_matches_rust_batch_identity(kind: str) -> None:
    """Declared IO schemas and Rust batches must agree, including metadata.

    Empty nested Arrow extensions must also resolve without constructing
    unsupported nested arrays through Schema.empty_table().
    """
    from polars_mongo._read import _arrow_to_polars_schema

    dtype: pa.DataType = ObjectIdType()
    if kind == "struct":
        dtype = pa.struct([pa.field("oid", dtype)])
    elif kind == "list":
        dtype = pa.large_list(pa.field("item", dtype))
    elif kind == "list_struct":
        dtype = pa.large_list(pa.field("item", pa.struct([pa.field("oid", dtype)])))

    batch = _internal.object_id_probe_series(kind).rename("value")
    resolved = _arrow_to_polars_schema(pa.schema([pa.field("value", dtype)]))
    assert resolved["value"] == batch.dtype
    assert _leaf_extension_names(resolved["value"]) == [OBJECT_ID_EXT_NAME]


def test_bson_target_extension_types_have_their_locked_storage() -> None:
    assert BsonDecimal128Type().storage_type == pa.float64()
    assert BsonTimestampType().storage_type == pa.timestamp("ms")
    assert BsonDecimal128().ext_storage() == pl.Float64
    assert BsonTimestamp().ext_storage() == pl.Datetime("ms")


def test_registration_is_idempotent() -> None:
    import importlib

    import polars_mongo._objectid as module

    importlib.reload(module)
    assert pl.get_extension_type(OBJECT_ID_EXT_NAME) is not None


def test_to_hex_from_hex_lossless_roundtrip() -> None:
    """AC-23."""
    for hex_value in (HEX_A, HEX_B, "000000000000000000000000", "ffffffffffffffffffffffff"):
        raw = _internal.object_id_hex_to_bytes(hex_value)
        assert len(raw) == 12
        assert _internal.object_id_bytes_to_hex(raw) == hex_value


def test_from_hex_rejects_invalid_input() -> None:
    from polars_mongo import MongoSchemaError

    for invalid in ("", "nope", HEX_A[:-1], "zz" + HEX_A[2:]):
        with pytest.raises(MongoSchemaError):
            _internal.object_id_hex_to_bytes(invalid)


NESTED_CASES = [
    "flat",
    "flat_with_null",
    "struct",
    "struct_null_parent",
    "struct_null_child",
    "list",
    "list_empty",
    "list_null_parent",
    "list_struct",
    "all_null",
]


@pytest.mark.parametrize("kind", NESTED_CASES)
def test_nested_extension_identity_survives_the_full_round_trip(kind: str) -> None:
    """Builder -> Series -> Python -> Rust, with identity intact at every level.

    The Rust half of the probe builds the Series; the Python half hands it back
    so Rust reports what identity actually survived, rather than assuming the
    export layer preserved it.
    """
    series = _internal.object_id_probe_series(kind)

    # Python side: the dtype resolves to the registered extension type wherever
    # an ObjectId lives, at any depth.
    assert _leaf_extension_names(series.dtype) == [OBJECT_ID_EXT_NAME]

    # Rust side, after re-import.
    assert _internal.series_extension_names(series) == [OBJECT_ID_EXT_NAME]


@pytest.mark.parametrize("kind", NESTED_CASES)
def test_identity_survives_slicing_and_chunking(kind: str) -> None:
    series = _internal.object_id_probe_series(kind)

    sliced = series.slice(1, len(series) - 1)
    assert _internal.series_extension_names(sliced) == [OBJECT_ID_EXT_NAME]

    chunked = pl.concat([series, series], rechunk=False)
    assert chunked.n_chunks() == 2
    assert _internal.series_extension_names(chunked) == [OBJECT_ID_EXT_NAME]


def _leaf_extension_names(dtype: object) -> list[str]:
    """Every extension name carried by `dtype`, outermost first."""
    names: list[str] = []
    if isinstance(dtype, pl.datatypes.BaseExtension):
        names.append(dtype.ext_name())
        names.extend(_leaf_extension_names(dtype.ext_storage()))
    elif isinstance(dtype, pl.List | pl.Array):
        names.extend(_leaf_extension_names(dtype.inner))
    elif isinstance(dtype, pl.Struct):
        for field in dtype.fields:
            names.extend(_leaf_extension_names(field.dtype))
    return names


def test_hex_expressions_round_trip_binary_object_ids() -> None:
    """The `#[polars_expr]` pair, over the extension dtype and its storage."""
    series = _internal.object_id_probe_series("flat_with_null").rename("oid")
    frame = pl.DataFrame({"oid": series})

    hexed = frame.select(pl.col("oid").pipe(_to_hex))
    assert hexed.to_series().to_list() == [HEX_A, None]

    parsed = hexed.select(pl.col("oid").pipe(_from_hex))
    assert parsed.dtypes == [pl.Array(pl.UInt8, 12)]
    assert parsed.to_series().to_list()[0] == list(_internal.object_id_hex_to_bytes(HEX_A))
    assert parsed.to_series().to_list()[1] is None


def test_from_hex_maps_invalid_input_to_null() -> None:
    frame = pl.DataFrame({"oid": [HEX_A, "nope", None]})
    parsed = frame.select(pl.col("oid").pipe(_from_hex)).to_series().to_list()
    assert parsed[0] == list(_internal.object_id_hex_to_bytes(HEX_A))
    assert parsed[1] is None
    assert parsed[2] is None


def _to_hex(expr: pl.Expr) -> pl.Expr:
    import polars_mongo

    return polars_mongo.object_id_to_hex(expr)


def _from_hex(expr: pl.Expr) -> pl.Expr:
    import polars_mongo

    return polars_mongo.object_id_from_hex(expr)


# --------------------------------------------------------------------------- #
# AC-21: real MongoDB ObjectIds preserve their registered Polars identity
# --------------------------------------------------------------------------- #


def _assert_no_unregistered_extension_warning(
    caught: list[WarningMessage],
) -> None:
    assert not [
        warning
        for warning in caught
        if "extension" in str(getattr(warning, "message", warning)).lower()
        and (
            "unregistered" in str(getattr(warning, "message", warning)).lower()
            or "not registered" in str(getattr(warning, "message", warning)).lower()
        )
    ]


def _object_id_storage_bytes(series: pl.Series) -> list[bytes]:
    """Read ObjectId leaves through Polars' extension storage accessor."""
    dtype = series.dtype
    if isinstance(dtype, pl.datatypes.BaseExtension):
        assert dtype.ext_name() == OBJECT_ID_EXT_NAME
        assert type(dtype) is ObjectId
        storage = series.ext.storage()
        values = storage.to_list()
        assert all(value is None or len(value) == 12 for value in values)
        return [bytes(value) for value in values if value is not None]
    if isinstance(dtype, pl.Struct):
        return _object_id_storage_bytes(series.struct.field("oid"))
    if isinstance(dtype, pl.List | pl.Array):
        return _object_id_storage_bytes(series.explode())
    raise AssertionError(f"ObjectId extension missing from {dtype!r}")


def _oracle_nested_object_id_bytes(value: object) -> list[bytes]:
    """Extract BSON ObjectId storage independently of polars-mongo."""
    from bson import ObjectId as BsonObjectId

    if isinstance(value, BsonObjectId):
        return [value.binary]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _oracle_nested_object_id_bytes(child)]
    if isinstance(value, list):
        return [item for child in value for item in _oracle_nested_object_id_bytes(child)]
    assert value is None
    return []


@pytest.mark.mongo
def test_read_mongo_id_dtype_is_registered_extension(
    mongo_conn: Any, raw: Any, clean_db: str
) -> None:
    """AC-21: a BSON `_id` has the registered extension dtype and 12-byte storage."""
    import warnings

    from bson import ObjectId as BsonObjectId

    import polars_mongo as pm

    collection = "objectid_registered_id"
    raw[clean_db][collection].insert_many(
        [
            {"key": "first", "_id": BsonObjectId()},
            {"key": "second", "_id": BsonObjectId()},
        ]
    )
    oracle = {
        document["key"]: document["_id"].binary
        for document in raw[clean_db][collection].find({}, {"key": 1, "_id": 1})
    }
    schema = pa.schema([pa.field("key", pa.string()), pa.field("_id", ObjectIdType())])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = pm.read_mongo(mongo_conn, clean_db, collection, schema=schema)

    _assert_no_unregistered_extension_warning(caught)
    dtype = frame.schema["_id"]
    assert type(dtype) is ObjectId
    assert dtype.ext_name() == "polars_mongo.object_id"
    assert set(frame.get_column("key")) == set(oracle)
    for key, expected in oracle.items():
        actual = frame.filter(pl.col("key") == key).get_column("_id")
        assert _object_id_storage_bytes(actual) == [expected]


LIVE_NESTED_OBJECT_ID_CASES = [
    "struct",
    "list",
    "list_struct",
    "null_parent",
    "null_child",
    "empty",
    "sliced",
    "chunked",
]


def _nested_object_id_case(kind: str) -> tuple[list[dict[str, object]], pa.DataType]:
    from bson import ObjectId as BsonObjectId

    object_id = ObjectIdType()
    struct = pa.struct([pa.field("oid", object_id)])
    list_of_ids = pa.large_list(pa.field("item", object_id))
    list_of_structs = pa.large_list(pa.field("item", struct))
    if kind == "struct":
        return (
            [{"key": key, "value": {"oid": BsonObjectId()}} for key in ("a", "b")],
            struct,
        )
    if kind == "list":
        return (
            [{"key": key, "value": [BsonObjectId()]} for key in ("a", "b")],
            list_of_ids,
        )
    if kind == "list_struct":
        return (
            [{"key": key, "value": [{"oid": BsonObjectId()}]} for key in ("a", "b")],
            list_of_structs,
        )
    if kind == "null_parent":
        return (
            [
                {"key": "null", "value": None},
                {"key": "value", "value": {"oid": BsonObjectId()}},
            ],
            struct,
        )
    if kind == "null_child":
        return (
            [
                {"key": "null", "value": {"oid": None}},
                {"key": "value", "value": {"oid": BsonObjectId()}},
            ],
            struct,
        )
    if kind == "empty":
        return (
            [
                {"key": "empty", "value": []},
                {"key": "value", "value": [BsonObjectId()]},
            ],
            list_of_ids,
        )
    if kind in {"sliced", "chunked"}:
        return (
            [
                {"key": key, "value": {"oid": BsonObjectId()}}
                for key in ("first", "second", "third")
            ],
            struct,
        )
    raise AssertionError(f"unknown ObjectId test case: {kind}")


@pytest.mark.mongo
@pytest.mark.parametrize("kind", LIVE_NESTED_OBJECT_ID_CASES)
def test_nested_objectid_dtype_preserved(
    mongo_conn: Any, raw: Any, clean_db: str, kind: str
) -> None:
    """AC-21: nested BSON ObjectIds retain identity and exact storage bytes."""
    import warnings

    import polars_mongo as pm

    collection = f"nested_objectid_{kind}"
    documents, value_type = _nested_object_id_case(kind)
    raw[clean_db][collection].insert_many(documents)
    oracle = {
        document["key"]: _oracle_nested_object_id_bytes(document["value"])
        for document in raw[clean_db][collection].find({}, {"key": 1, "value": 1})
    }
    schema = pa.schema([pa.field("key", pa.string()), pa.field("value", value_type)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = pm.read_mongo(
            mongo_conn,
            clean_db,
            collection,
            schema=schema,
            batch_size=1 if kind == "chunked" else 1024,
        )

    _assert_no_unregistered_extension_warning(caught)
    assert _leaf_extension_names(frame.schema["value"]) == [OBJECT_ID_EXT_NAME]
    if kind == "chunked":
        assert frame.get_column("value").n_chunks() > 1
    if kind == "sliced":
        frame = frame.slice(1, frame.height - 1)
        assert frame.height == 2

    actual_keys = set(frame.get_column("key"))
    assert frame.get_column("key").n_unique() == frame.height
    if kind == "sliced":
        assert actual_keys == {"second", "third"}
    else:
        assert actual_keys == set(oracle)
    for key in actual_keys:
        value = frame.filter(pl.col("key") == key).get_column("value")
        assert _object_id_storage_bytes(value) == oracle[key]


@pytest.mark.mongo
def test_full_loop_byte_identity(mongo_conn: Any, raw: Any, clean_db: str) -> None:
    """AC-22: BSON ObjectId tags and bytes survive read_mongo -> write_mongo."""
    from bson import ObjectId as BsonObjectId

    import polars_mongo as pm

    source = "objectid_full_loop_source"
    destination = "objectid_full_loop_destination"
    source_documents = [
        {"key": "alpha", "_id": BsonObjectId("000000000000000000000001")},
        {"key": "bravo", "_id": BsonObjectId("507f1f77bcf86cd799439011")},
        {"key": "charlie", "_id": BsonObjectId("5f2b4c8e1a2b3c4d5e6f7a8b")},
        {"key": "delta", "_id": BsonObjectId("abcdef0123456789abcdef01")},
        {"key": "echo", "_id": BsonObjectId("fffffffffffffffffffffffe")},
    ]
    raw[clean_db][source].insert_many(source_documents)
    expected = {
        document["key"]: document["_id"].binary
        for document in raw[clean_db][source].find({}, {"key": 1, "_id": 1})
    }
    schema = pa.schema([pa.field("key", pa.string()), pa.field("_id", ObjectIdType())])

    frame = pm.read_mongo(mongo_conn, clean_db, source, schema=schema)
    inserted = pm.write_mongo(mongo_conn, clean_db, destination, frame, schema=schema)

    assert inserted == len(expected)
    observed = {
        document["key"]: document["_id"]
        for document in raw[clean_db][destination].find({}, {"key": 1, "_id": 1})
    }
    assert set(observed) == set(expected)
    for key, expected_bytes in expected.items():
        actual = observed[key]
        assert isinstance(actual, BsonObjectId)
        assert actual.binary == expected_bytes


@pytest.mark.mongo
@pytest.mark.parametrize(
    "kind",
    ["struct", "list", "list_struct", "null_parent", "null_child", "empty"],
)
def test_write_mongo_preserves_nested_objectid_bson_type_and_bytes(
    mongo_conn: Any, raw: Any, clean_db: str, kind: str
) -> None:
    """AC-19/AC-21: pymongo verifies nested write output by key, never position."""
    from bson import ObjectId as BsonObjectId

    import polars_mongo as pm

    source = f"nested_objectid_write_source_{kind}"
    destination = f"nested_objectid_write_destination_{kind}"
    documents, value_type = _nested_object_id_case(kind)
    raw[clean_db][source].insert_many(documents)
    source_values = {
        document["key"]: document["value"]
        for document in raw[clean_db][source].find({}, {"key": 1, "value": 1})
    }
    expected = {key: _oracle_nested_object_id_bytes(value) for key, value in source_values.items()}
    schema = pa.schema([pa.field("key", pa.string()), pa.field("value", value_type)])

    frame = pm.read_mongo(mongo_conn, clean_db, source, schema=schema)
    assert pm.write_mongo(mongo_conn, clean_db, destination, frame, schema=schema) == len(documents)

    observed = {
        document["key"]: document["value"]
        for document in raw[clean_db][destination].find({}, {"key": 1, "value": 1})
    }
    assert set(observed) == set(expected)
    for key, value in observed.items():
        object_ids = _nested_bson_object_ids(value)
        assert all(isinstance(object_id, BsonObjectId) for object_id in object_ids)
        assert [object_id.binary for object_id in object_ids] == expected[key]
        assert _oracle_nested_bson_value(value) == _oracle_nested_bson_value(source_values[key])


def _nested_bson_object_ids(value: object) -> list[Any]:
    """Extract BSON ObjectIds from the independent pymongo inspection result."""
    from bson import ObjectId as BsonObjectId

    if isinstance(value, BsonObjectId):
        return [value]
    if isinstance(value, dict):
        return [
            object_id for child in value.values() for object_id in _nested_bson_object_ids(child)
        ]
    if isinstance(value, list):
        return [object_id for child in value for object_id in _nested_bson_object_ids(child)]
    assert value is None
    return []


def _oracle_nested_bson_value(value: object) -> object:
    """Compare raw pymongo values while retaining container and null shape."""
    from bson import ObjectId as BsonObjectId

    if isinstance(value, BsonObjectId):
        return value.binary
    if isinstance(value, dict):
        return {key: _oracle_nested_bson_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_oracle_nested_bson_value(child) for child in value]
    assert value is None
    return None
