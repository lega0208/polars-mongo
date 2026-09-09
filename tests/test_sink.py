"""Phase 8 lazy-sink acceptance tests with profiler and pymongo evidence."""

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
COLLECTION = "sink_acceptance"
_FIELDS: list[pa.Field[Any]] = [
    pa.field("_id", pa.int64()),
    pa.field("value", pa.string()),
]
SCHEMA = pa.schema(_FIELDS)


def _frame(ids: list[int]) -> pl.DataFrame:
    """Build source data independently of the write implementation."""
    return pl.DataFrame({"_id": ids, "value": [f"value-{row_id}" for row_id in ids]})


def _insert_commands(entries: list[dict[str, Any]], collection: str) -> list[dict[str, Any]]:
    """Return only profiled insert operations for the destination collection."""
    return [entry for entry in entries if entry.get("command", {}).get("insert") == collection]


@pytest.mark.mongo
@pytest.mark.parametrize("batch_size", [0, -1])
def test_sink_mongo_rejects_non_positive_batch_size_before_insert(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
    batch_size: int,
) -> None:
    """A non-empty lazy frame rejects an invalid size, while a valid size still writes."""
    frame = _frame([1, 2]).lazy()

    with pytest.raises(pm.MongoSchemaError) as excinfo:
        pm.sink_mongo(
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

    inserted = pm.sink_mongo(
        mongo_conn,
        clean_db,
        COLLECTION,
        frame,
        schema=SCHEMA,
        batch_size=1,
    )
    assert inserted == 2
    assert raw[clean_db][COLLECTION].count_documents({}) == inserted


@pytest.mark.mongo
def test_non_row_failure_at_later_batch_after_partial_row_failures(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
    rust_app_name: str,
) -> None:
    """AC-12: a later write-concern error wins over earlier indexed failures."""
    batch_size = 2
    preceding_batches = 2
    failure_row_start = batch_size * preceding_batches
    failure_row_stop = failure_row_start + batch_size
    failure_text = "planned later-batch write concern failure"
    raw[clean_db][COLLECTION].insert_many(
        [{"_id": 1, "value": "seeded-first"}, {"_id": 3, "value": "seeded-second"}]
    )
    raw.admin.command(
        "configureFailPoint",
        "failCommand",
        # `skip` is a fail-point *mode*, not a data field: the first
        # `preceding_batches` inserts proceed and the next one fails.
        mode={"skip": preceding_batches},
        data={
            "failCommands": ["insert"],
            # Scope the fail point to this scan's client so the independent
            # pymongo oracle keeps working while it is armed.
            "appName": rust_app_name,
            "writeConcernError": {
                "code": 64,
                "codeName": "WriteConcernFailed",
                "errmsg": failure_text,
            },
        },
    )

    try:
        with pytest.raises(pm.MongoBatchError) as raised:
            pm.sink_mongo(
                mongo_conn,
                clean_db,
                COLLECTION,
                _frame(list(range(8))).lazy(),
                schema=SCHEMA,
                batch_size=batch_size,
            )
    finally:
        failpoint_off = raw.admin.command("configureFailPoint", "failCommand", mode="off")

    error = raised.value
    assert error.row_start == failure_row_start
    assert error.row_stop == failure_row_stop
    assert isinstance(error.cause, str)
    assert failure_text in error.cause
    assert not hasattr(error, "failures")
    assert f"{failure_row_start}..{failure_row_stop}" in str(error)
    assert failure_text in str(error)

    # `count` returned by turning the fail point off is the number of times it
    # ACTIVATED, not the number of inserts it saw: the skipped ones do not
    # count. Once `skip` is exhausted the fail point stays armed and fires on
    # every later matching insert, so exactly one activation is observed
    # evidence that no insert followed the failing one -- not a presumed
    # skip-mode budget.
    assert failpoint_off["count"] == 1
    entries = profile(rust_app_name, namespace=f"{clean_db}.{COLLECTION}")
    assert len(_insert_commands(entries, COLLECTION)) == preceding_batches + 1


@pytest.mark.mongo
def test_sink_mongo_stream_ordinal_across_batches(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """A lazy sink reports duplicate rows by ordinal in its ordered output stream."""
    failing_index = 3
    raw[clean_db][COLLECTION].insert_one({"_id": failing_index, "value": "seeded"})

    with pytest.raises(pm.MongoWriteError) as raised:
        pm.sink_mongo(
            mongo_conn,
            clean_db,
            COLLECTION,
            _frame(list(range(6))).lazy(),
            schema=SCHEMA,
            batch_size=2,
        )

    error = raised.value
    assert len(error.failures) == 1
    row_index, server_message = error.failures[0]
    assert row_index == failing_index
    assert "E11000" in server_message
    assert str(failing_index) in str(error)
    assert "E11000" in str(error)


@pytest.mark.mongo
def test_sink_mongo_returns_total_inserted_count(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """A successful multi-batch sink returns only its total acknowledged inserts."""
    ids = list(range(7))

    inserted = pm.sink_mongo(
        mongo_conn,
        clean_db,
        COLLECTION,
        _frame(ids).lazy(),
        schema=SCHEMA,
        batch_size=2,
    )

    assert inserted == len(ids)
    assert raw[clean_db][COLLECTION].count_documents({}) == inserted
