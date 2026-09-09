"""Conversion acceptance at the public read boundary (AC-16, AC-18, AC-19).

The Rust halves of these criteria are unit-tested in `cargo test`
(`src/bson_arrow/tests.rs`, `src/bson_arrow/schema.rs`). This module is the
missing Python half: real documents, seeded by `pymongo`, read back through
`read_mongo` and `scan_mongo(...).collect()` and compared against hand-written
literals or against what the independent `pymongo` oracle reports. `polars-mongo`
is never its own oracle.

Both public surfaces are exercised for every criterion, because `read_mongo` is
literally `scan_mongo(...).collect()` only as an implementation fact - the
acceptance criteria are stated about the surfaces, so both are measured.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
import pytest
from bson import Binary, Code, Decimal128, Int64, MaxKey, ObjectId, Regex, Timestamp

import polars_mongo as pm
from polars_mongo import _internal

if TYPE_CHECKING:
    from pymongo import MongoClient

SURFACES = ["read_mongo", "scan_mongo.collect"]


def _read(surface: str, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    """Run one of the two public read surfaces as itself."""
    if surface == "read_mongo":
        return pm.read_mongo(*args, **kwargs)
    return pm.scan_mongo(*args, **kwargs).collect()


def _column(frame: pl.DataFrame, name: str) -> list[Any]:
    """A column's Python values, with any extension dtype dropped to storage.

    Extension dtypes are an identity marker, asserted separately; the value
    assertions want the storage the marker wraps.
    """
    series = frame.get_column(name)
    dtype = series.dtype
    if isinstance(dtype, pl.datatypes.BaseExtension):
        series = series.ext.storage()
    return series.to_list()


# ---------------------------------------------------------------------------
# AC-16: silent widening, and the section 5.1 mapping rows.
# ---------------------------------------------------------------------------

WIDENING_ID = ObjectId("507f1f77bcf86cd799439011")
# Exactly representable as float64, and the largest integer for which the next
# integer is not: a lossy widening would be visible here.
BIG_INT64 = 9007199254740991

_WIDENING_FIELDS: list[pa.Field[Any]] = [
    pa.field("i32_as_i64", pa.int64()),
    pa.field("i32_as_f64", pa.float64()),
    pa.field("i64_as_f64", pa.float64()),
    pa.field("i32_as_i32", pa.int32()),
]
WIDENING_SCHEMA = pa.schema(_WIDENING_FIELDS)


@pytest.mark.mongo
@pytest.mark.parametrize("surface", SURFACES)
def test_widening_coercions_silent(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    surface: str,
) -> None:
    """AC-16: Int32 -> int64 and Int32/Int64 -> float64 widen without an error.

    "Silent" is two claims, and both are asserted: no exception is raised, and
    the widened value is bit-exact rather than merely close. `BIG_INT64` is the
    largest integer whose float64 image is exact, so a lossy widening path would
    change the value instead of only its type.
    """
    collection = "widening"
    raw[clean_db][collection].insert_one(
        {
            "_id": WIDENING_ID,
            # pymongo encodes a plain small Python int as BSON Int32.
            "i32_as_i64": 7,
            "i32_as_f64": -3,
            "i64_as_f64": Int64(BIG_INT64),
            "i32_as_i32": 11,
        }
    )

    frame = _read(surface, mongo_conn, clean_db, collection, schema=WIDENING_SCHEMA)

    assert frame.schema["i32_as_i64"] == pl.Int64
    assert frame.schema["i32_as_f64"] == pl.Float64
    assert frame.schema["i64_as_f64"] == pl.Float64
    assert frame.schema["i32_as_i32"] == pl.Int32

    assert _column(frame, "i32_as_i64") == [7]
    assert _column(frame, "i32_as_f64") == [-3.0]
    assert _column(frame, "i64_as_f64") == [float(BIG_INT64)]
    # Exactness, not approximate equality: the round trip through float64 must
    # land back on the same integer.
    assert int(_column(frame, "i64_as_f64")[0]) == BIG_INT64
    assert _column(frame, "i32_as_i32") == [11]


MAPPING_ID = ObjectId("5f2b4c8e1a2b3c4d5e6f7a8b")
MAPPING_OID_VALUE = ObjectId("000102030405060708090a0b")
MAPPING_DATETIME = dt.datetime(2020, 5, 17, 12, 34, 56, tzinfo=dt.UTC)
MAPPING_TIMESTAMP_SECONDS = 1_589_718_896

_MAPPING_FIELDS: list[pa.Field[Any]] = [
    pa.field("f_double", pa.float64()),
    pa.field("f_string", pa.string()),
    pa.field("f_bool", pa.bool_()),
    pa.field("f_int32", pa.int32()),
    pa.field("f_int64", pa.int64()),
    pa.field("f_datetime", pa.timestamp("ms")),
    pa.field("f_binary", pa.binary()),
    pa.field("f_object_id", pm.ObjectIdType()),
    pa.field("f_decimal128", pm.BsonDecimal128Type()),
    pa.field("f_timestamp", pm.BsonTimestampType()),
]
MAPPING_SCHEMA = pa.schema(_MAPPING_FIELDS)


def _seed_mapping_document(raw: MongoClient[dict[str, Any]], database: str) -> str:
    """One document carrying every section 5.1 BSON tag, written by pymongo."""
    collection = "mapping_rows"
    raw[database][collection].insert_one(
        {
            "_id": MAPPING_ID,
            "f_double": 1.5,
            "f_string": "text",
            "f_bool": True,
            "f_int32": 7,
            "f_int64": Int64(BIG_INT64),
            "f_datetime": MAPPING_DATETIME,
            "f_binary": Binary(b"\x00\x01\xfe", 0),
            "f_object_id": MAPPING_OID_VALUE,
            "f_decimal128": Decimal128("12.25"),
            "f_timestamp": Timestamp(MAPPING_TIMESTAMP_SECONDS, 4),
        }
    )
    return collection


@pytest.mark.mongo
@pytest.mark.parametrize("surface", SURFACES)
def test_section_5_1_mapping_rows_decode_direction(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    surface: str,
) -> None:
    """AC-16: every section 5.1 row, BSON -> Arrow, at the public boundary.

    Expected values are hand-written literals matching what was seeded with
    `pymongo`; nothing here is compared against a second `polars-mongo` call.

    Direction note: only the decode direction has a public Python surface today.
    The encode direction is covered by `cargo test`
    (`writes_float64_as_double_and_decimal128_target_as_decimal128`,
    `writes_datetime_and_timestamp_targets_from_the_same_instant`,
    `writes_the_remaining_scalar_rows_with_their_declared_tags`,
    `writes_object_id_bytes_exactly` in `src/bson_arrow/tests.rs`) and is
    re-asserted at the Python boundary in Phase 8, when the write surface exists.
    """
    collection = _seed_mapping_document(raw, clean_db)

    frame = _read(surface, mongo_conn, clean_db, collection, schema=MAPPING_SCHEMA)
    assert frame.height == 1

    assert _column(frame, "f_double") == [1.5]
    assert _column(frame, "f_string") == ["text"]
    assert _column(frame, "f_bool") == [True]
    assert _column(frame, "f_int32") == [7]
    assert _column(frame, "f_int64") == [BIG_INT64]
    assert _column(frame, "f_datetime") == [MAPPING_DATETIME.replace(tzinfo=None)]
    assert _column(frame, "f_binary") == [b"\x00\x01\xfe"]
    assert _column(frame, "f_object_id") == [list(MAPPING_OID_VALUE.binary)]
    # Decimal128 is carried as its float64 storage, exactly.
    assert _column(frame, "f_decimal128") == [12.25]
    # BSON Timestamp: seconds scaled to milliseconds, increment (4) discarded.
    assert _column(frame, "f_timestamp") == [
        dt.datetime(1970, 1, 1) + dt.timedelta(seconds=MAPPING_TIMESTAMP_SECONDS)
    ]

    # The declared identity survives to the caller for the three extension rows.
    for name, identity, storage in (
        ("f_object_id", "polars_mongo.object_id", pl.Array(pl.UInt8, 12)),
        ("f_decimal128", "polars_mongo.bson_decimal128", pl.Float64),
        ("f_timestamp", "polars_mongo.bson_timestamp", pl.Datetime("ms")),
    ):
        dtype = frame.schema[name]
        assert isinstance(dtype, pl.datatypes.BaseExtension)
        assert dtype.ext_name() == identity
        assert dtype.ext_storage() == storage


def test_section_5_1_mapping_rows_declaration_direction() -> None:
    """AC-16: the same rows in the declaration direction, exactly.

    The declared pyarrow type is what selects the BSON target, so this asserts
    the exact resolved `(path, plan type, nullable, BSON target)` tuple for every
    row - never a substring or a truthiness stand-in. `BsonDecimal128Type` and
    `BsonTimestampType` are the caller's only way to say "write BSON Decimal128"
    or "write BSON Timestamp" rather than double / DateTime, so their targets are
    the load-bearing rows.

    The value half of this direction is covered by `cargo test` in
    `src/bson_arrow/tests.rs` (see the decode-direction test above) until the
    Phase 8 write surface exists.
    """
    assert _internal.compile_schema_plan(MAPPING_SCHEMA) == [
        ("f_double", "Float64", "true", ""),
        ("f_string", "String", "true", ""),
        ("f_bool", "Boolean", "true", ""),
        ("f_int32", "Int32", "true", ""),
        ("f_int64", "Int64", "true", ""),
        ("f_datetime", "DateTimeMs", "true", ""),
        ("f_binary", "Binary", "true", ""),
        ("f_object_id", "ObjectId", "true", "ObjectId"),
        ("f_decimal128", "Decimal128", "true", "Decimal128"),
        ("f_timestamp", "Timestamp", "true", "Timestamp"),
    ]


@pytest.mark.mongo
@pytest.mark.parametrize("surface", SURFACES)
def test_bson_timestamp_increment_is_discarded_on_read(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    surface: str,
) -> None:
    """AC-16: two Timestamps differing only in increment read back identically.

    A millisecond instant cannot carry the increment, so discarding it is the
    documented behaviour rather than a defect - but it must be *total*: an
    implementation that leaked the increment into the low bits would separate
    these two rows. The write direction synthesizes the increment back as zero
    and is covered by `writes_datetime_and_timestamp_targets_from_the_same_instant`
    in `src/bson_arrow/tests.rs`, re-asserted at the Python boundary in Phase 8.
    """
    collection = "timestamp_increment"
    raw[clean_db][collection].insert_many(
        [
            {"_id": 1, "ts": Timestamp(MAPPING_TIMESTAMP_SECONDS, 0)},
            {"_id": 2, "ts": Timestamp(MAPPING_TIMESTAMP_SECONDS, 65_535)},
        ]
    )

    fields: list[pa.Field[Any]] = [pa.field("ts", pm.BsonTimestampType())]
    frame = _read(surface, mongo_conn, clean_db, collection, schema=pa.schema(fields))

    expected = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=MAPPING_TIMESTAMP_SECONDS)
    assert _column(frame, "ts") == [expected, expected]


# ---------------------------------------------------------------------------
# AC-18: a missing field is null when nullable, and an error when it is not.
# ---------------------------------------------------------------------------

PRESENT_ID = ObjectId("507f1f77bcf86cd799439021")
ABSENT_ID = ObjectId("507f1f77bcf86cd799439022")

# `$$REMOVE` makes the computed field genuinely absent from the projected
# document rather than present-and-null, which is the case AC-18 is about.
COMPUTED_PROJECTION: dict[str, Any] = {
    "_id": 0,
    "total": {
        "$cond": [
            {"$gt": [{"$ifNull": ["$qty", 0]}, 0]},
            {"$multiply": ["$qty", "$price"]},
            "$$REMOVE",
        ]
    },
}


def _missing_field_call(
    kind: str, nullable: bool
) -> tuple[str, dict[str, Any], list[Any], str, str]:
    """The read arguments, expected present value, field name and document id.

    Returns `(field, kwargs, expected_when_nullable, field_name, document_id)`
    for one of the three parametrized surfaces.
    """
    if kind == "base":
        fields: list[pa.Field[Any]] = [pa.field("opt", pa.int64(), nullable=nullable)]
        return (
            "opt",
            {"schema": pa.schema(fields)},
            [10, None],
            "opt",
            f"_id={ABSENT_ID} (batch row 1)",
        )
    if kind == "projection":
        projected: list[pa.Field[Any]] = [pa.field("opt", pa.int64(), nullable=nullable)]
        return (
            "opt",
            {
                "projection": {"_id": 0, "opt": 1},
                "projection_schema": pa.schema(projected),
            },
            [10, None],
            "opt",
            "<no _id> (batch row 1)",
        )
    computed: list[pa.Field[Any]] = [pa.field("total", pa.float64(), nullable=nullable)]
    return (
        "total",
        {
            "projection": COMPUTED_PROJECTION,
            "projection_schema": pa.schema(computed),
        },
        [30.0, None],
        "total",
        "<no _id> (batch row 1)",
    )


def _seed_missing_field(raw: MongoClient[dict[str, Any]], database: str, kind: str) -> str:
    """Two documents: the first carries the field, the second does not."""
    collection = f"missing_{kind}"
    raw[database][collection].insert_many(
        [
            {"_id": PRESENT_ID, "opt": 10, "qty": 3, "price": 10.0},
            {"_id": ABSENT_ID, "other": 1},
        ]
    )
    return collection


@pytest.mark.mongo
@pytest.mark.parametrize("kind", ["base", "projection", "computed-projection"])
@pytest.mark.parametrize("surface", SURFACES)
def test_missing_field_null_when_nullable_raises_when_not(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    kind: str,
    surface: str,
) -> None:
    """AC-18: absence is null under a nullable declaration and an error otherwise.

    Nullability is the caller's contract, so the same collection must produce
    both outcomes purely from the declared schema. The failing half asserts the
    complete typed attribute set - not just the class - and that the rendered
    message locates *both* the offending document and the offending field, since
    an error that names neither is useless on a million-document scan.

    The three `kind`s are the three ways a caller states the output contract:
    `schema`, `projection` + `projection_schema`, and a server-computed field
    under `projection_schema`. All three must agree.
    """
    collection = _seed_missing_field(raw, clean_db, kind)

    field, nullable_kwargs, expected, field_path, document_id = _missing_field_call(kind, True)
    frame = _read(surface, mongo_conn, clean_db, collection, **nullable_kwargs)
    assert _column(frame, field) == expected

    _, strict_kwargs, _, _, _ = _missing_field_call(kind, False)
    with pytest.raises(pm.MongoConversionError) as excinfo:
        _read(surface, mongo_conn, clean_db, collection, **strict_kwargs)

    error = excinfo.value
    assert error.document_id == document_id
    assert error.field_path == field_path
    assert error.declared_type == ("Int64" if kind != "computed-projection" else "Float64")
    assert error.bson_type == "Missing"
    assert error.collection == f"{clean_db}.{collection}"
    assert error.reason == "the field is missing and the declared type is not nullable"

    message = str(error)
    assert document_id in message, message
    assert f"'{field_path}'" in message, message
    assert "missing" in message.lower(), message
    assert "not nullable" in message.lower(), message


# ---------------------------------------------------------------------------
# AC-19: nested documents, and the excluded BSON values.
# ---------------------------------------------------------------------------

_GEO_TYPE = pa.struct([pa.field("lat", pa.float64()), pa.field("lon", pa.float64())])
_ADDRESS_TYPE = pa.struct([pa.field("city", pa.string()), pa.field("geo", _GEO_TYPE)])
_PART_TYPE = pa.struct([pa.field("sku", pa.string()), pa.field("count", pa.int64())])
_ORDER_TYPE = pa.struct(
    [
        pa.field("id", pa.int64()),
        pa.field("parts", pa.list_(pa.field("part", _PART_TYPE))),
    ]
)

_NESTED_FIELDS: list[pa.Field[Any]] = [
    pa.field("key", pa.string()),
    # depth 3: profile -> address -> geo -> lat
    pa.field(
        "profile",
        pa.struct([pa.field("name", pa.string()), pa.field("address", _ADDRESS_TYPE)]),
    ),
    pa.field("tags", pa.list_(pa.field("tag", pa.string()))),
    # depth 4: orders -> item -> parts -> part -> sku
    pa.field("orders", pa.list_(pa.field("item", _ORDER_TYPE))),
]
NESTED_SCHEMA = pa.schema(_NESTED_FIELDS)

NESTED_DOCUMENTS: list[dict[str, Any]] = [
    {
        "_id": 1,
        "key": "complete",
        "profile": {
            "name": "ada",
            "address": {"city": "london", "geo": {"lat": 51.5, "lon": -0.1}},
        },
        "tags": ["a", "b"],
        "orders": [
            {
                "id": Int64(1),
                "parts": [{"sku": "x", "count": Int64(2)}, {"sku": "y", "count": None}],
            },
            {"id": Int64(2), "parts": []},
        ],
    },
    {
        # Null parents at every level, and an empty list.
        "_id": 2,
        "key": "empty_and_null_parent",
        "profile": {"name": "bob", "address": None},
        "tags": [],
        "orders": None,
    },
    {
        # Null children under a present parent.
        "_id": 3,
        "key": "null_children",
        "profile": {"name": None, "address": {"city": None, "geo": None}},
        "tags": [None, "c"],
        "orders": [{"id": None, "parts": None}],
    },
]


def _expected_from_oracle(value: Any, arrow_type: pa.DataType) -> Any:
    """Project a raw `pymongo` value onto the declared schema shape.

    The oracle is `pymongo`: this only reshapes what `pymongo` reported (adding
    the keys the declared struct has and the document omits, as null) so the two
    sides are logically comparable. It never consults `polars-mongo`.
    """
    if value is None:
        return None
    if pa.types.is_struct(arrow_type):
        return {
            field.name: _expected_from_oracle(value.get(field.name), field.type)
            for field in arrow_type
        }
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return [_expected_from_oracle(item, arrow_type.value_type) for item in value]
    return value


@pytest.mark.mongo
@pytest.mark.parametrize("surface", SURFACES)
def test_nested_document_logical_roundtrip(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    surface: str,
) -> None:
    """AC-19: structs, lists and lists of structs at depth 3+ survive logically.

    "Logically equal" is measured against the independent `pymongo` oracle,
    reshaped onto the declared schema (absent struct keys become null) and
    nothing else. The document set deliberately covers the shapes that a naive
    builder collapses: an empty list, a null list, a null struct parent, null
    struct children and a null element inside a list.
    """
    collection = "nested"
    raw[clean_db][collection].insert_many(NESTED_DOCUMENTS)

    oracle = list(raw[clean_db][collection].find(sort=[("_id", 1)]))
    assert len(oracle) == len(NESTED_DOCUMENTS)
    expected = [
        {
            field.name: _expected_from_oracle(document.get(field.name), field.type)
            for field in NESTED_SCHEMA
        }
        for document in oracle
    ]

    frame = _read(surface, mongo_conn, clean_db, collection, schema=NESTED_SCHEMA)
    assert frame.to_dicts() == expected

    destination = f"{collection}_{surface.replace('.', '_')}"
    assert pm.write_mongo(mongo_conn, clean_db, destination, frame, schema=NESTED_SCHEMA) == len(
        expected
    )

    expected_by_key = {
        document["key"]: {
            field.name: document[field.name] for field in NESTED_SCHEMA if field.name != "key"
        }
        for document in expected
    }
    observed_by_key = {
        document["key"]: {
            field.name: document[field.name] for field in NESTED_SCHEMA if field.name != "key"
        }
        for document in raw[clean_db][destination].find(
            {}, {"_id": 0, **{field.name: 1 for field in NESTED_SCHEMA}}
        )
    }
    assert set(observed_by_key) == set(expected_by_key)
    assert observed_by_key == expected_by_key


EXCLUDED_VALUES: dict[str, Any] = {
    "javascript": Code("function () { return 1; }"),
    "regex": Regex("^a", "i"),
    "max_key": MaxKey(),
}
EXCLUDED_TAGS = {
    "javascript": "JavaScriptCode",
    "regex": "RegularExpression",
    "max_key": "MaxKey",
}

_EXCLUDED_FIELDS: list[pa.Field[Any]] = [
    pa.field(
        "orders",
        pa.list_(pa.field("item", pa.struct([pa.field("code", pa.string())]))),
    )
]
EXCLUDED_SCHEMA = pa.schema(_EXCLUDED_FIELDS)
EXCLUDED_ID = ObjectId("507f1f77bcf86cd799439031")


@pytest.mark.mongo
@pytest.mark.parametrize("kind", sorted(EXCLUDED_VALUES))
@pytest.mark.parametrize("surface", SURFACES)
def test_excluded_bson_value_raises_with_dotted_path(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    kind: str,
    surface: str,
) -> None:
    """AC-19: an excluded BSON value is located by its full dotted path.

    The value sits three levels down, inside a list of structs. A root path
    ("orders") would be technically true and operationally worthless, so the
    assertion is exact equality with `orders.item.code` - the list element level
    included - rather than a prefix or containment check.
    """
    collection = f"excluded_{kind}"
    raw[clean_db][collection].insert_one(
        {"_id": EXCLUDED_ID, "orders": [{"code": EXCLUDED_VALUES[kind]}]}
    )

    with pytest.raises(pm.MongoConversionError) as excinfo:
        _read(surface, mongo_conn, clean_db, collection, schema=EXCLUDED_SCHEMA)

    error = excinfo.value
    assert error.field_path == "orders.item.code"
    assert error.document_id == f"_id={EXCLUDED_ID} (batch row 0)"
    assert error.declared_type == "String"
    assert error.bson_type == EXCLUDED_TAGS[kind]
    assert error.collection == f"{clean_db}.{collection}"
    assert error.reason == f"BSON {EXCLUDED_TAGS[kind]} is not supported by polars-mongo"
    assert "orders.item.code" in str(error)
