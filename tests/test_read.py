"""Phase 7 read-path acceptance: what the two public surfaces actually do.

Every expectation here is independent of `polars_mongo`. Frames come from
literal row data in `tests/expected.py`; row sets that depend on server-side
windowing are re-established with an identical `pymongo` query; and what reached
the server is read back out of `system.profile`, which records what the *Rust*
driver emitted rather than what this process believes it sent.

The lazy surface is held to the stronger claim: building a `LazyFrame` must
issue no operation at all, so the tests assert an empty profiler slice for the
scan's `appName` on the scanned namespace - not merely the absence of a `find`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, override

import polars as pl
import polars.testing
import pytest

import expected as ex
import polars_mongo as pm
from polars_mongo import _read
from polars_mongo._internal import _detach_counters
from polars_mongo._read import DEFAULT_BATCH_SIZE, MongoBatchReader, read_mongo, scan_mongo

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pymongo import MongoClient

Profile = Callable[..., list[dict[str, Any]]]

COLLECTION = "read_acceptance"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _seed(raw: MongoClient[dict[str, Any]], database: str, rows: list[dict[str, Any]]) -> None:
    """Insert `rows` in order through the independent pymongo oracle.

    The documents are inserted exactly as written in `tests/expected.py`, so an
    unsorted `find` returns them in the order the expected frames declare.
    """
    if rows:
        raw[database][COLLECTION].insert_many([dict(row) for row in rows])


def _finds(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The profiled `find` commands among `entries`."""
    return [entry for entry in entries if "find" in entry.get("command", {})]


def _namespace(database: str, collection: str = COLLECTION) -> str:
    return f"{database}.{collection}"


# --------------------------------------------------------------------------- #
# AC-4 / AC-5: laziness and the two surfaces agreeing with pymongo
# --------------------------------------------------------------------------- #


@pytest.mark.mongo
def test_lazy_scan_does_not_touch_the_server_until_collect(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
) -> None:
    """A scan that is built and discarded must cost the server nothing (AC-4)."""
    _seed(raw, clean_db, ex.BASE_ROWS)

    frame = scan_mongo(mongo_conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA)
    assert isinstance(frame, pl.LazyFrame)
    del frame

    assert profile(namespace=_namespace(clean_db)) == []


@pytest.mark.mongo
def test_read_mongo_matches_pymongo_documents(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """A collected scan returns exactly what pymongo reports for the same query.

    The oracle is a second, independent driver reading the same collection: the
    row set, its order and every value are compared against what `pymongo`
    decoded, never against another `polars_mongo` call.
    """
    _seed(raw, clean_db, ex.BASE_ROWS)

    oracle = [
        {key: document[key] for key in ("k", "name", "score", "active")}
        for document in raw[clean_db][COLLECTION].find({})
    ]
    frame = read_mongo(mongo_conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA)

    assert frame.columns == ["k", "name", "score", "active"]
    assert frame.to_dicts() == oracle
    polars.testing.assert_frame_equal(
        frame, ex.expected_frame(oracle, ex.BASE_SCHEMA), check_dtypes=True
    )


@pytest.mark.mongo
def test_scan_issues_no_operation_before_collect(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
) -> None:
    """AC-5: zero operations of any kind, then exactly one `find` on collect.

    The assertion is deliberately not "no `find`". Schema inference, row
    counting or index inspection would each be a server round trip the lazy
    contract forbids, so the whole profiler slice for the scan's `appName` on
    the scanned namespace must be empty - `find`, `count`, `aggregate`,
    `distinct`, `listIndexes` and anything else alike.
    """
    _seed(raw, clean_db, ex.BASE_ROWS)
    namespace = _namespace(clean_db)

    lazy = scan_mongo(mongo_conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA)
    shaped = lazy.filter(pl.col("k") >= 2).select("k", "name").head(3)

    before = profile(namespace=namespace)
    assert before == [], f"the lazy scan issued {[entry['op'] for entry in before]}"

    result = shaped.collect()

    after = profile(namespace=namespace)
    assert len(_finds(after)) == 1, [entry.get("command") for entry in after]
    assert result.columns == ["k", "name"]
    assert result["k"].to_list() == [2, 3, 4]


# --------------------------------------------------------------------------- #
# AC-6: the expected-frame matrix on both surfaces
# --------------------------------------------------------------------------- #


@pytest.mark.mongo
@pytest.mark.parametrize("case", ex.MATRIX, ids=[case.id for case in ex.MATRIX])
@pytest.mark.parametrize("surface", ["read_mongo", "scan_mongo.collect"])
def test_expected_frames(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    case: ex.MatrixCase,
    surface: str,
) -> None:
    """AC-6: filter x projection x limit, against literal expected frames.

    Both public surfaces are asserted against the *same* independently written
    frame, so the eager and lazy paths cannot drift apart without one of them
    disagreeing with a literal.
    """
    _seed(raw, clean_db, ex.BASE_ROWS)

    kwargs: dict[str, Any] = {
        "schema": case.schema,
        "filter": case.filter,
        "projection": case.projection,
        "projection_schema": case.projection_schema,
        "limit": case.limit,
    }
    if surface == "read_mongo":
        frame = read_mongo(mongo_conn, clean_db, COLLECTION, **kwargs)
    else:
        frame = scan_mongo(mongo_conn, clean_db, COLLECTION, **kwargs).collect()

    polars.testing.assert_frame_equal(frame, case.expected, check_dtypes=True)


def test_read_mongo_forwards_arguments_and_collects_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`read_mongo` is `scan_mongo(...).collect()` - proven, not asserted in prose.

    Every argument must arrive at `scan_mongo` unchanged and the resulting frame
    must be collected exactly once: a second collect would run the whole scan a
    second time against the server.
    """
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = pl.DataFrame({"sentinel": [1]})

    class _CollectOnce:
        def __init__(self) -> None:
            self.collects = 0

        def collect(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
            self.collects += 1
            return sentinel

    lazy = _CollectOnce()

    def fake_scan(*args: Any, **kwargs: Any) -> _CollectOnce:
        calls.append((args, kwargs))
        return lazy

    monkeypatch.setattr(_read, "scan_mongo", fake_scan)

    connection = object()
    schema = ex.BASE_SCHEMA
    projection_schema = ex.PROJECTED_SCHEMA
    result = read_mongo(
        connection,  # ty: ignore[invalid-argument-type]
        "db",
        "coll",
        schema=schema,
        filter={"k": 1},
        projection=ex.PROJECTION,
        projection_schema=projection_schema,
        limit=7,
        batch_size=13,
    )

    assert result is sentinel
    assert lazy.collects == 1
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (connection, "db", "coll")
    assert kwargs == {
        "schema": schema,
        "filter": {"k": 1},
        "projection": ex.PROJECTION,
        "projection_schema": projection_schema,
        "limit": 7,
        "batch_size": 13,
    }


def test_read_mongo_forwards_the_default_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is forwarded explicitly, so the two surfaces share one default."""
    calls: list[dict[str, Any]] = []

    class _Lazy:
        def collect(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
            return pl.DataFrame()

    def fake_scan(*args: Any, **kwargs: Any) -> _Lazy:
        calls.append(kwargs)
        return _Lazy()

    monkeypatch.setattr(_read, "scan_mongo", fake_scan)
    read_mongo(object(), "db", "coll", schema=ex.BASE_SCHEMA)  # ty: ignore[invalid-argument-type]

    assert calls[0]["batch_size"] == DEFAULT_BATCH_SIZE


# --------------------------------------------------------------------------- #
# AC-7: the io-source callback's own contract
# --------------------------------------------------------------------------- #

AC7_ROWS = [ex.wide_row(k) for k in range(12)]
AC7_BATCH_SIZE = 3
AC7_WITH_COLUMNS = ["name", "k"]  # deliberately not the schema's order
AC7_N_ROWS = 8
AC7_EXCLUDED = [3, 4, 5]
#: The exact frames the callback must yield: batch 2 filters away completely and
#: must not end the scan, and the cumulative cap trims the fourth batch.
AC7_EXPECTED_K = [[0, 1, 2], [], [6, 7, 8], [9, 10]]


def _ac7_predicate() -> pl.Expr:
    return ~pl.col("k").is_in(AC7_EXCLUDED)


def _assert_ac7_contract(frames: list[pl.DataFrame]) -> None:
    """The acceptance assertion set for the captured callback.

    Kept in one function so the negative controls below can prove it actually
    fails when the transform loop regresses, rather than trusting that it would.
    """
    assert [frame.columns for frame in frames] == [AC7_WITH_COLUMNS] * len(AC7_EXPECTED_K), (
        "projection pushdown must be applied, in the requested column order"
    )
    assert [frame["k"].to_list() for frame in frames] == AC7_EXPECTED_K
    assert [frame["name"].to_list() for frame in frames] == [
        [ex.wide_row(k)["name"] for k in batch] for batch in AC7_EXPECTED_K
    ]
    assert sum(frame.height for frame in frames) == AC7_N_ROWS, (
        "the n_rows cap is cumulative across batches, not per batch"
    )
    assert frames[1].height == 0, "an empty intermediate batch must be handled, not fatal"
    assert sum(1 for frame in frames if frame.height) == 3, (
        "the cap must be reached across at least three contributing batches"
    )


def _ac7_raw_batches() -> list[pl.DataFrame]:
    """The untransformed batches, rebuilt from literals for the negative controls."""
    batches = []
    for start in range(0, len(AC7_ROWS), AC7_BATCH_SIZE):
        chunk = AC7_ROWS[start : start + AC7_BATCH_SIZE]
        batches.append(ex.expected_frame(chunk, ex.BASE_SCHEMA))
    return batches


@pytest.mark.mongo
def test_captured_callback_applies_transforms(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-7: the real callback, driven directly with explicit pushdown arguments.

    The engine is removed from the path: `register_io_source` is replaced by a
    recorder that stores the callable and still returns the real `LazyFrame`, so
    the callback under test is the production one. The recorder performs no
    transform of its own - every column, row and cap observed below was produced
    by `_source` itself.
    """
    _seed(raw, clean_db, AC7_ROWS)

    captured: list[Callable[..., Iterator[pl.DataFrame]]] = []
    from polars.io.plugins import register_io_source as original

    def recording_register(callable_: Any, *, schema: Any) -> pl.LazyFrame:
        captured.append(callable_)
        return original(callable_, schema=schema)

    monkeypatch.setattr(_read, "register_io_source", recording_register)
    scan_mongo(
        mongo_conn,
        clean_db,
        COLLECTION,
        schema=ex.BASE_SCHEMA,
        batch_size=AC7_BATCH_SIZE,
    )
    assert len(captured) == 1

    frames = list(
        captured[0](
            AC7_WITH_COLUMNS,  # with_columns
            _ac7_predicate(),  # predicate
            AC7_N_ROWS,  # n_rows
            None,  # batch_size (the reader's own batch_size governs)
        )
    )

    _assert_ac7_contract(frames)


def _broken_loop(
    batches: list[pl.DataFrame],
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
    *,
    reset_cap_per_batch: bool,
) -> list[pl.DataFrame]:
    """A deliberately breakable local copy of the transform loop.

    The negative controls feed the same literal batches through this copy with
    one behaviour removed at a time, proving `_assert_ac7_contract` detects that
    regression instead of passing on any plausible output.
    """
    frames: list[pl.DataFrame] = []
    produced = 0
    for batch in batches:
        if reset_cap_per_batch:
            produced = 0
        frame = batch
        if with_columns is not None:
            frame = frame.select(with_columns)
        if predicate is not None:
            frame = frame.filter(predicate)
        if n_rows is not None:
            remaining = n_rows - produced
            if remaining <= 0:
                break
            if frame.height > remaining:
                frame = frame.head(remaining)
        produced += frame.height
        frames.append(frame)
        if n_rows is not None and not reset_cap_per_batch and produced >= n_rows:
            break
    return frames


def test_negative_control_select_bypassed_fails_the_contract() -> None:
    """If projection pushdown were ignored, the contract must fail."""
    frames = _broken_loop(
        _ac7_raw_batches(), None, _ac7_predicate(), AC7_N_ROWS, reset_cap_per_batch=False
    )
    with pytest.raises(AssertionError):
        _assert_ac7_contract(frames)


def test_negative_control_filter_bypassed_fails_the_contract() -> None:
    """If predicate pushdown were ignored, the contract must fail."""
    frames = _broken_loop(
        _ac7_raw_batches(), AC7_WITH_COLUMNS, None, AC7_N_ROWS, reset_cap_per_batch=False
    )
    with pytest.raises(AssertionError):
        _assert_ac7_contract(frames)


def test_negative_control_n_rows_reset_per_batch_fails_the_contract() -> None:
    """If the cap were re-armed per batch, the contract must fail."""
    frames = _broken_loop(
        _ac7_raw_batches(),
        AC7_WITH_COLUMNS,
        _ac7_predicate(),
        AC7_N_ROWS,
        reset_cap_per_batch=True,
    )
    with pytest.raises(AssertionError):
        _assert_ac7_contract(frames)


# --------------------------------------------------------------------------- #
# AC-8: the server-side limit applies before any local filter
# --------------------------------------------------------------------------- #


@pytest.mark.mongo
@pytest.mark.parametrize("case", ["beyond-window", "in-window-crossing-batches-with-cap"])
def test_server_limit_before_local_filter(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
    case: str,
) -> None:
    """AC-8: `limit` is a server-side window, not a post-filter row count.

    Preconditions
    -------------
    * The collection holds `ex.WIDE_ROWS` inserted in `k` order and is never
      modified afterwards, so an unsorted `find` is deterministic.
    * Neither surface issues a sort, so the window `W` is the first `limit`
      documents matching the filter in natural order.

    The window is established *independently*: the same `find` (same filter, no
    sort, same limit) is run through `pymongo`, and the expected frame is
    derived from what that query returned. The local Polars filter is then
    applied to `W` only - it cannot pull in rows the server never sent - and the
    profiled command is read back to confirm the `limit` really travelled.

    * `beyond-window` selects rows that exist in the collection but lie outside
      `W`, so the correct answer is empty.
    * `in-window-crossing-batches-with-cap` uses a `batch_size` smaller than the
      limit and adds a `head` cap, so the window spans several batches.
    """
    _seed(raw, clean_db, ex.WIDE_ROWS)
    namespace = _namespace(clean_db)

    server_filter: dict[str, Any] = {"active": False}
    limit = 5 if case == "beyond-window" else 7
    batch_size = 2

    window = list(raw[clean_db][COLLECTION].find(server_filter, limit=limit))
    assert len(window) == limit, "precondition: the seed must fill the window"

    lazy = scan_mongo(
        mongo_conn,
        clean_db,
        COLLECTION,
        schema=ex.BASE_SCHEMA,
        filter=server_filter,
        limit=limit,
        batch_size=batch_size,
    )
    cap = 3
    if case == "beyond-window":
        local = pl.col("k") >= 15
        expected_rows = [ex.wide_row(doc["k"]) for doc in window if doc["k"] >= 15]
        result = lazy.filter(local).collect()
    else:
        local = pl.col("k") % 2 == 1
        expected_rows = [ex.wide_row(doc["k"]) for doc in window if doc["k"] % 2 == 1][:cap]
        result = lazy.filter(local).head(cap).collect()

    polars.testing.assert_frame_equal(
        result, ex.expected_frame(expected_rows, ex.BASE_SCHEMA), check_dtypes=True
    )
    if case == "beyond-window":
        assert result.height == 0, "rows outside the server window must be unreachable"
    else:
        assert result.height > 0
        assert result.height <= cap

    finds = _finds(profile(namespace=namespace))
    assert len(finds) == 1
    assert finds[0]["command"]["limit"] == limit
    assert finds[0]["command"]["filter"] == server_filter


# --------------------------------------------------------------------------- #
# AC-13 / AC-14: what reaches the server
# --------------------------------------------------------------------------- #


@pytest.mark.mongo
def test_profiled_find_carries_filter(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
) -> None:
    """AC-13: the filter is executed by the server, not re-applied locally."""
    _seed(raw, clean_db, ex.BASE_ROWS)
    server_filter: dict[str, Any] = {"k": {"$gte": 5}, "active": True}

    frame = read_mongo(
        mongo_conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA, filter=server_filter
    )

    assert frame["k"].to_list() == [6]
    finds = _finds(profile(namespace=_namespace(clean_db)))
    assert len(finds) == 1
    assert finds[0]["command"]["filter"] == server_filter


@pytest.mark.mongo
def test_profiled_find_carries_projection(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    profile: Profile,
) -> None:
    """AC-14: the projection reaches the server, so unwanted fields never travel."""
    _seed(raw, clean_db, ex.BASE_ROWS)

    frame = read_mongo(
        mongo_conn,
        clean_db,
        COLLECTION,
        projection=ex.PROJECTION,
        projection_schema=ex.PROJECTED_SCHEMA,
    )

    assert frame.columns == ["k", "name"]
    finds = _finds(profile(namespace=_namespace(clean_db)))
    assert len(finds) == 1
    assert finds[0]["command"]["projection"] == ex.PROJECTION


@pytest.mark.mongo
@pytest.mark.parametrize("surface", ["read_mongo", "scan_mongo"])
def test_projection_without_projection_schema_raises_MongoSchemaError_at_call_time(  # noqa: N802
    mongo_uri: str,
    clean_db: str,
    profile: Profile,
    surface: str,
) -> None:
    """AC-14: the rejection happens before any connection could be used.

    The connection handed in is closed, so any server contact would fail loudly
    with `MongoConnectionClosedError` instead. Getting `MongoSchemaError` proves
    the argument validation runs first; the empty profiler slice proves nothing
    was sent.
    """
    connection = pm.MongoConnection(mongo_uri)
    connection.close()
    assert connection.closed

    with pytest.raises(pm.MongoSchemaError) as excinfo:
        if surface == "read_mongo":
            read_mongo(
                connection,
                clean_db,
                COLLECTION,
                schema=ex.BASE_SCHEMA,
                projection=ex.PROJECTION,
            )
        else:
            scan_mongo(
                connection,
                clean_db,
                COLLECTION,
                schema=ex.BASE_SCHEMA,
                projection=ex.PROJECTION,
            )

    error = excinfo.value
    assert error.field_path == "projection"
    assert "projection_schema" in error.reason
    message = str(error)
    assert "projection" in message
    assert "projection_schema" in message
    assert profile(namespace=_namespace(clean_db)) == []


@pytest.mark.mongo
@pytest.mark.parametrize("batch_size", [0, -1])
@pytest.mark.parametrize("surface", ["read_mongo", "scan_mongo"])
def test_read_batch_size_is_validated_at_call_time_without_server_contact(
    mongo_conn: pm.MongoConnection,
    clean_db: str,
    profile: Profile,
    batch_size: int,
    surface: str,
) -> None:
    """Both read surfaces reject non-positive batch sizes before opening a cursor."""
    with pytest.raises(pm.MongoSchemaError) as excinfo:
        if surface == "read_mongo":
            read_mongo(
                mongo_conn,
                clean_db,
                COLLECTION,
                schema=ex.BASE_SCHEMA,
                batch_size=batch_size,
            )
        else:
            scan_mongo(
                mongo_conn,
                clean_db,
                COLLECTION,
                schema=ex.BASE_SCHEMA,
                batch_size=batch_size,
            )

    error = excinfo.value
    assert error.field_path == "batch_size"
    assert str(batch_size) in error.reason
    message = str(error)
    assert "batch_size" in message
    assert str(batch_size) in message
    assert profile(namespace=_namespace(clean_db)) == []


@pytest.mark.mongo
@pytest.mark.parametrize("surface", ["read_mongo", "scan_mongo.collect"])
def test_projection_schema_is_sole_output_contract(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    surface: str,
) -> None:
    """AC-15: `projection_schema` replaces `schema` outright - no merge, no fallback.

    The base `schema` is deliberately unusable against the seeded documents: it
    declares `name` as Int64 (every document stores a string) and names a column
    that does not exist. If either schema were merged in, or used as a fallback,
    the read would raise instead of returning the projected frame.
    """
    _seed(raw, clean_db, ex.BASE_ROWS)

    kwargs: dict[str, Any] = {
        "schema": ex.INCOMPATIBLE_BASE_SCHEMA,
        "projection": ex.PROJECTION,
        "projection_schema": ex.PROJECTED_SCHEMA,
    }
    if surface == "read_mongo":
        frame = read_mongo(mongo_conn, clean_db, COLLECTION, **kwargs)
    else:
        frame = scan_mongo(mongo_conn, clean_db, COLLECTION, **kwargs).collect()

    polars.testing.assert_frame_equal(
        frame,
        ex.expected_frame(ex.SOLE_CONTRACT_EXPECTED_ROWS, ex.PROJECTED_SCHEMA),
        check_dtypes=True,
    )


# --------------------------------------------------------------------------- #
# AC-17: the typed conversion error at the public boundary
# --------------------------------------------------------------------------- #

AC17_BATCH_SIZE = 4
AC17_BAD_INDEX = 6  # batch 1, row 2


def _seed_conversion_failure(raw: MongoClient[dict[str, Any]], database: str) -> str:
    """Seed 8 documents, one with a String where the schema declares Int64.

    The `_id`s are assigned here rather than by the server so the expected
    `document_id` attribute is an exact value, not a shape.
    """
    from bson import ObjectId

    documents: list[dict[str, Any]] = []
    bad_id = ""
    for index, row in enumerate(ex.BASE_ROWS):
        oid = ObjectId()
        document = dict(row)
        document["_id"] = oid
        if index == AC17_BAD_INDEX:
            document["k"] = "not-an-int"
            bad_id = str(oid)
        documents.append(document)
    raw[database][COLLECTION].insert_many(documents)
    return bad_id


@pytest.mark.mongo
@pytest.mark.parametrize("surface", ["read_mongo", "scan_mongo.collect"])
def test_conversion_error_attributes_and_message_at_public_boundary(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    surface: str,
) -> None:
    """AC-17: the caller receives the located, typed failure on both surfaces.

    All six declared attributes are compared with exact expected values - the
    offending `_id` is chosen by the test, and the batch-row ordinal follows
    from the seeded position and `batch_size` - and the message must name the
    located facts.
    """
    bad_id = _seed_conversion_failure(raw, clean_db)

    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011 - the class IS the assertion
        if surface == "read_mongo":
            read_mongo(
                mongo_conn,
                clean_db,
                COLLECTION,
                schema=ex.BASE_SCHEMA,
                batch_size=AC17_BATCH_SIZE,
            )
        else:
            scan_mongo(
                mongo_conn,
                clean_db,
                COLLECTION,
                schema=ex.BASE_SCHEMA,
                batch_size=AC17_BATCH_SIZE,
            ).collect()

    error = excinfo.value
    assert isinstance(error, pm.MongoConversionError), (
        f"expected MongoConversionError, got {type(error).__name__}: {error}"
    )
    assert isinstance(error, pm.MongoError)
    assert error.document_id == f"_id={bad_id} (batch row {AC17_BAD_INDEX % AC17_BATCH_SIZE})"
    assert error.field_path == "k"
    assert error.declared_type == "Int64"
    assert error.bson_type == "String"
    assert error.collection == f"{clean_db}.{COLLECTION}"
    assert error.reason == "BSON String cannot be converted to the declared type"

    message = str(error)
    for fact in ("k", "Int64", "String", COLLECTION, bad_id):
        assert fact in message, f"{fact!r} missing from {message!r}"


# --------------------------------------------------------------------------- #
# AC-2: connection lifetime across a lazy scan
# --------------------------------------------------------------------------- #


@pytest.mark.mongo
def test_connection_context_manager_with_late_collect(
    mongo_uri: str,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """AC-2: the scan is usable while the connection is open, and only then.

    A `LazyFrame` outlives the `with` block by construction, so the failure must
    surface at collect time as the typed `MongoConnectionClosedError` carrying
    the URI - not as a driver timeout, a panic, or a silently empty frame.
    """
    _seed(raw, clean_db, ex.BASE_ROWS)

    late: pl.LazyFrame | None = None
    with pm.MongoConnection(mongo_uri) as conn:
        inside = scan_mongo(conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA)
        collected = inside.collect()
        polars.testing.assert_frame_equal(
            collected, ex.expected_frame(ex.BASE_ROWS, ex.BASE_SCHEMA), check_dtypes=True
        )
        late = scan_mongo(conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA)

    assert conn.closed
    assert late is not None

    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011 - the class IS the assertion
        late.collect()

    error = excinfo.value
    assert isinstance(error, pm.MongoConnectionClosedError), (
        f"expected MongoConnectionClosedError, got {type(error).__name__}: {error}"
    )
    assert error.uri == mongo_uri
    message = str(error)
    assert "connection to" in message.lower()
    assert "is closed" in message.lower()
    assert "requires it open at collect time" in message.lower()

    with pytest.raises(BaseException) as eager_excinfo:  # noqa: B017, PT011 - the class IS the assertion
        read_mongo(conn, clean_db, COLLECTION, schema=ex.BASE_SCHEMA)

    eager_error = eager_excinfo.value
    assert isinstance(eager_error, pm.MongoConnectionClosedError), (
        f"expected MongoConnectionClosedError, got {type(eager_error).__name__}: {eager_error}"
    )
    assert eager_error.uri == mongo_uri
    eager_message = str(eager_error)
    assert "connection to" in eager_message.lower()
    assert "is closed" in eager_message.lower()
    assert "requires it open at collect time" in eager_message.lower()


# --------------------------------------------------------------------------- #
# AC-3: the GIL is released around the blocking driver call
# --------------------------------------------------------------------------- #

GIL_ROWS = 20_000
GIL_BATCH_SIZE = 16
GIL_SCHEMA = ex.BASE_SCHEMA


class _DetachProbe(threading.Thread):
    """Counts its own iterations strictly inside observed detach windows.

    `_detach_counters()` reports `(entered, exited)`. `entered > exited` means
    the reader is *observably* inside the blocking driver call with the GIL
    released. The probe only counts iterations while that holds, so progress it
    reports cannot come from incidental Python work outside the window, and no
    wall-clock duration is ever asserted.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.windows = 0
        self.iterations_inside = 0
        self.polls = 0

    @override
    def run(self) -> None:
        inside_previous = False
        while not self.stop.is_set():
            entered, exited = _detach_counters()
            self.polls += 1
            inside = entered > exited
            if inside:
                if not inside_previous:
                    self.windows += 1
                self.iterations_inside += 1
            inside_previous = inside


@pytest.mark.mongo
def test_gil_is_released_during_the_blocking_driver_call(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """AC-3: another Python thread makes real progress while the driver blocks.

    This is a synchronized experiment, not a timing one. The reader signals
    entry to and exit from the `py.detach` window through `_detach_counters`;
    the probe thread counts only iterations it performed while that window was
    open. If the GIL were held across the driver call the probe could not
    execute a single bytecode inside the window, so `iterations_inside` would be
    zero.

    The corpus and the small `batch_size` guarantee many windows: if a window
    were too short to observe, the fix is more server work, never a weaker
    assertion.
    """
    rows = [ex.wide_row(k) for k in range(GIL_ROWS)]
    collection = raw[clean_db][COLLECTION]
    for start in range(0, GIL_ROWS, 5_000):
        collection.insert_many([dict(row) for row in rows[start : start + 5_000]])
    assert collection.count_documents({}) == GIL_ROWS

    probe = _DetachProbe()
    probe.start()
    try:
        frame = read_mongo(
            mongo_conn,
            clean_db,
            COLLECTION,
            schema=GIL_SCHEMA,
            batch_size=GIL_BATCH_SIZE,
        )
    finally:
        probe.stop.set()
        probe.join(timeout=30)
    assert not probe.is_alive()

    assert frame.height == GIL_ROWS
    assert probe.windows >= 1, "no detach window was ever observed; the experiment is void"
    assert probe.iterations_inside > 0, (
        "the probe thread executed nothing while the reader was inside the blocking "
        f"driver call ({probe.windows} windows observed over {probe.polls} polls)"
    )


@pytest.mark.mongo
def test_batch_reader_is_exposed_for_direct_iteration(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """The reader used by the callback is the one the package exports.

    AC-7's captured-callback probe is only meaningful if `_read.MongoBatchReader`
    is the real batching reader; iterating it directly must reproduce the seeded
    documents under the declared batch size.
    """
    _seed(raw, clean_db, ex.BASE_ROWS)

    reader = MongoBatchReader(mongo_conn, clean_db, COLLECTION, ex.BASE_SCHEMA, None, None, None, 3)
    batches = list(reader)

    assert [batch.height for batch in batches] == [3, 3, 2]
    polars.testing.assert_frame_equal(
        pl.concat(batches), ex.expected_frame(ex.BASE_ROWS, ex.BASE_SCHEMA), check_dtypes=True
    )
