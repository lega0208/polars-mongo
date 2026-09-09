"""Phase 8 eager-write acceptance tests with pymongo as the server oracle."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
import pytest

import polars_mongo as pm

if TYPE_CHECKING:
    from pymongo import MongoClient


Profile = Callable[..., list[dict[str, Any]]]
COLLECTION = "write_acceptance"
_FIELDS: list[pa.Field[Any]] = [
    pa.field("_id", pa.int64()),
    pa.field("value", pa.string()),
]
SCHEMA = pa.schema(_FIELDS)
INTERNAL_SPLIT_SIZE = 100_001


def _frame(ids: list[int]) -> pl.DataFrame:
    """Build a declared-shape frame without using the package under test."""
    return pl.DataFrame({"_id": ids, "value": [f"value-{row_id}" for row_id in ids]})


@pytest.mark.mongo
@pytest.mark.parametrize("batch_size", [0, -1])
def test_write_mongo_rejects_non_positive_batch_size_before_insert(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
    batch_size: int,
) -> None:
    """A non-empty frame rejects an invalid size, while a valid size still writes."""
    frame = _frame([1, 2])

    with pytest.raises(pm.MongoSchemaError) as excinfo:
        pm.write_mongo(
            mongo_conn,
            clean_db,
            COLLECTION,
            frame,
            schema=SCHEMA,
            batch_size=batch_size,
        )

    error = excinfo.value
    assert error.field_path == "batch_size"
    assert str(batch_size) in error.reason
    message = str(error)
    assert "batch_size" in message
    assert str(batch_size) in message
    assert profile(namespace=f"{clean_db}.{COLLECTION}") == []

    inserted = pm.write_mongo(
        mongo_conn,
        clean_db,
        COLLECTION,
        frame,
        schema=SCHEMA,
        batch_size=1,
    )
    assert inserted == frame.height
    assert raw[clean_db][COLLECTION].count_documents({}) == frame.height


@pytest.mark.mongo
def test_unordered_insert_continues_past_duplicate_key(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """AC-10: an unordered write attempts rows after an indexed duplicate error."""
    raw[clean_db][COLLECTION].insert_one({"_id": 2, "value": "seeded"})

    with pytest.raises(pm.MongoWriteError) as raised:
        pm.write_mongo(mongo_conn, clean_db, COLLECTION, _frame([1, 2, 3]), schema=SCHEMA)

    error = raised.value
    assert len(error.failures) == 1
    row_index, server_message = error.failures[0]
    assert row_index == 1
    assert "E11000" in server_message
    assert "1" in str(error)
    assert "E11000" in str(error)
    observed = {
        document["_id"]: document["value"]
        for document in raw[clean_db][COLLECTION].find({}, {"_id": 1, "value": 1})
    }
    assert observed == {1: "value-1", 2: "seeded", 3: "value-3"}


@pytest.mark.mongo
def test_partial_failure_row_index_is_stream_global(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """AC-11: a failure in batch two retains its stream-global ordinal."""
    failing_index = 1_500
    raw[clean_db][COLLECTION].insert_one({"_id": failing_index, "value": "seeded"})

    with pytest.raises(pm.MongoWriteError) as raised:
        pm.write_mongo(
            mongo_conn,
            clean_db,
            COLLECTION,
            _frame(list(range(2_001))),
            schema=SCHEMA,
            batch_size=1_000,
        )

    error = raised.value
    assert len(error.failures) == 1
    row_index, server_message = error.failures[0]
    assert row_index == failing_index
    assert "E11000" in server_message
    assert str(failing_index) in str(error)
    assert "E11000" in str(error)


@pytest.mark.mongo
def test_oversized_insert_many_reports_later_split_indices_and_counts(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """Driver-internal insert_many splits preserve success counts and global indices."""
    success_collection = f"{COLLECTION}_split_success"
    failure_collection = f"{COLLECTION}_split_failure"
    ids = list(range(INTERNAL_SPLIT_SIZE))

    inserted = pm.write_mongo(
        mongo_conn,
        clean_db,
        success_collection,
        _frame(ids),
        schema=SCHEMA,
        batch_size=INTERNAL_SPLIT_SIZE,
    )
    assert inserted == INTERNAL_SPLIT_SIZE
    assert raw[clean_db][success_collection].count_documents({}) == inserted

    failing_index = INTERNAL_SPLIT_SIZE - 1
    raw[clean_db][failure_collection].insert_one({"_id": failing_index, "value": "seeded"})
    with pytest.raises(pm.MongoWriteError) as raised:
        pm.write_mongo(
            mongo_conn,
            clean_db,
            failure_collection,
            _frame(ids),
            schema=SCHEMA,
            batch_size=INTERNAL_SPLIT_SIZE,
        )

    error = raised.value
    assert len(error.failures) == 1
    row_index, server_message = error.failures[0]
    assert row_index == failing_index
    assert "E11000" in server_message
    assert str(failing_index) in str(error)
    assert "E11000" in str(error)
    assert raw[clean_db][failure_collection].count_documents({}) == INTERNAL_SPLIT_SIZE
