"""Harness validation (plan section 4, F10).

Profiler output is operation-dependent and truncating, so each observation
mechanism is proven before any acceptance criterion depends on it. This half --
`appName` scoping -- needs only `_test_ping` and pymongo, so it is a Phase 2
exit. The outgoing-insert-document half needs the real `insert_batch` primitive
and is therefore the Phase 8 entry gate.

**Recorded 8.0 gate outcome: FAILED, and AC-9 is blocked and escalated.**
Neither observer specified by the plan exposes the outgoing insert document on
the pinned MongoDB 8.0.4 standalone:

* `system.profile` records the insert command body only
  (`insert`, `ordered`, `lsid`, `$db`) with no `documents` field;
* the fixture's own structured JSON log records the same body, both in the
  "About to run the command" record and in the "Slow query" record, with
  `logComponentVerbosity` raised to `{command: 5, write: 5}`.

The documents travel as an OP_MSG document sequence which the server never
writes to either sink, so the loss is a server-side capability limit rather than
a truncation this package could avoid with smaller documents: it reproduces
identically for a one-field pymongo insert issued without this package. Missing
evidence is a rejection, so AC-9 (`test_outgoing_insert_omits_client_id`) has no
valid observer and does **not** ship. Independently, the pinned driver
contradicts AC-9's substantive claim at the source: `mongodb` 3.9.0
`src/operation/insert.rs:80` calls `get_or_prepend_id_field` on **every**
document before it goes on the wire, so the outgoing document always carries an
`_id`, client-generated when the caller did not supply one.

The test below is the permanent record of that failing outcome: it asserts the
double failure, so a future server that does expose the documents turns this
into a red test and reopens AC-9 rather than letting the block rot silently.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
import pytest

from polars_mongo._internal import MongoBatchWriter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pymongo import MongoClient

    import polars_mongo as pm


@pytest.mark.mongo
def test_profiler_scopes_by_appname(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Callable[..., list[dict[str, Any]]],
    rust_app_name: str,
    pymongo_app_name: str,
) -> None:
    """Rust-client operations are separable from pymongo's own traffic."""
    marker = "appname_scope_probe"
    raw[clean_db][marker].insert_one({"seeded_by": "pymongo"})

    mongo_conn._test_ping(clean_db)

    rust_entries = profile(rust_app_name)
    pymongo_entries = profile(pymongo_app_name)

    # The server attributes each operation to the client that issued it.
    assert rust_entries, "no profiler entries recorded for the Rust client's appName"
    assert any(entry["ns"].endswith(marker) for entry in pymongo_entries), (
        "the pymongo seed insert was not recorded under pymongo's appName"
    )
    assert not any(entry["ns"].endswith(marker) for entry in rust_entries), (
        "pymongo's insert leaked into the Rust client's appName scope"
    )
    assert all(entry["appName"] == rust_app_name for entry in rust_entries)
    assert all(entry["appName"] == pymongo_app_name for entry in pymongo_entries)


DOCUMENT = {"_id": "known-profiler-id", "marker": "visible"}


@pytest.mark.mongo
def test_profiler_exposes_outgoing_insert_documents(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    mongod_logpath: Path,
    clean_db: str,
    profile: Callable[..., list[dict[str, Any]]],
    rust_app_name: str,
) -> None:
    """The Phase 8 entry gate, recorded as FAILED with its evidence.

    A real, deliberately tiny insert goes out through `MongoBatchWriter`, and
    both specified observers are then asked for its documents. The gate would
    pass only if one of them returned the document keys intact; both return the
    command body without them, so AC-9 does not ship (see the module docstring).
    """
    collection = "outgoing_insert_visibility"
    raw.admin.command(
        {
            "setParameter": 1,
            "logComponentVerbosity": {"command": {"verbosity": 5}, "write": {"verbosity": 5}},
        }
    )
    log_offset = mongod_logpath.stat().st_size

    schema = pa.schema([pa.field("_id", pa.string()), pa.field("marker", pa.string())])
    writer = MongoBatchWriter(mongo_conn, clean_db, collection, schema)
    outcome = writer.insert_batch(
        pl.DataFrame({key: [value] for key, value in DOCUMENT.items()}), 0
    )

    assert outcome.attempted == 1
    assert outcome.inserted == 1
    assert outcome.failures == []
    # The write itself landed: the observation is what is missing, not the insert.
    assert raw[clean_db][collection].find_one({"_id": DOCUMENT["_id"]}) == DOCUMENT

    # Observer 1: the database profiler.
    inserts = [
        entry
        for entry in profile(rust_app_name, namespace=f"{clean_db}.{collection}")
        if entry.get("op") == "insert"
    ]
    assert inserts, "the Rust insert was not recorded in system.profile at all"
    profiled = inserts[-1].get("command", {})
    assert profiled.get("insert") == collection
    assert "documents" not in profiled, (
        "the profiler now exposes outgoing insert documents: AC-9 has a valid "
        f"observer again and must be reinstated (command: {profiled!r})"
    )

    # Observer 2: the fixture's own structured mongod log, command verbosity 5.
    with mongod_logpath.open("rb") as log:
        log.seek(log_offset)
        records = [json.loads(line) for line in log if line.strip()]
    logged = [
        command
        for record in records
        for command in (
            record.get("attr", {}).get("command", {}),
            record.get("attr", {}).get("commandArgs", {}),
        )
        if isinstance(command, dict) and command.get("insert") == collection
    ]
    assert logged, "the structured mongod log recorded no insert command for the namespace"
    assert all("documents" not in command for command in logged), (
        "the structured mongod log now exposes outgoing insert documents: AC-9 "
        f"has a valid observer again and must be reinstated (records: {logged!r})"
    )
