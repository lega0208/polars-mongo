"""The section 3.1 contract-enforcing error-transport gate (7.0).

This is a gate, not a survey. `polars_error::wrap_msg` reconstructs an exception
*class* from a message and copies the traceback, carrying neither instance
attributes nor a `__cause__` link (F7), so a synthetic-chain unit test would
validate a walker rather than the boundary. This probe drives the real
`register_io_source` callback and asserts fidelity on the exception the caller
actually receives.

Every cell of the matrix must pass:

    surface                  x  failure position  x  execution path
    read_mongo(...)             first batch          each path a default
    scan_mongo(...).collect()   later batch          collect() can take

The only passing outcome is the correct class from the five-class hierarchy,
every declared typed attribute equal to what the reader set, and the located
facts present in the message. Nothing weaker passes, and there is no
default-fidelity exemption for direct `scan_mongo(...).collect()`.

Because the two surfaces are exercised as themselves, `read_mongo` is only run
under the engine a plain default `collect()` selects; the other engine paths are
covered on the surface that can name them. `POLARS_ENGINE_AFFINITY` lets the
default-collect cell be pinned to each supported path without substituting a
different public API.

The probe ships permanently, so a Polars upgrade that repairs (or re-breaks)
this boundary fails loudly instead of silently changing the shipped default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
import pytest

import polars_mongo as pm
from polars_mongo._read import MongoBatchReader, read_mongo, scan_mongo

if TYPE_CHECKING:
    from pymongo import MongoClient

# The execution paths a plain default `collect()` can take on the pinned
# Polars. The probe enumerates and exercises them rather than assuming a
# single one.
ENGINES = ["in-memory", "streaming"]

_FIELDS: list[pa.Field[Any]] = [
    pa.field("k", pa.int64()),
    pa.field("name", pa.string()),
]
SCHEMA = pa.schema(_FIELDS)

BATCH_SIZE = 4
GOOD_ROWS = 10
BAD_VALUE = "not-an-int"


def _seed(raw: MongoClient[dict[str, Any]], database: str, bad_at: int) -> tuple[str, str]:
    """Seed the collection, returning its name and the offending `_id` in hex.

    The `_id`s are assigned here rather than by the server so the expected
    `document_id` attribute is known exactly, not merely by shape.
    """
    from bson import ObjectId

    collection = f"transport_bad_at_{bad_at}"
    documents: list[dict[str, Any]] = []
    bad_id = ""
    for index in range(GOOD_ROWS):
        oid = ObjectId()
        value: Any = BAD_VALUE if index == bad_at else index
        if index == bad_at:
            bad_id = str(oid)
        documents.append({"_id": oid, "k": value, "name": f"n{index}"})
    raw[database][collection].insert_many(documents)
    return collection, bad_id


def _expected_attributes(
    database: str, collection: str, bad_at: int, bad_id: str
) -> dict[str, str]:
    """Exactly what the reader sets when it rejects the seeded document."""
    return {
        "document_id": f"_id={bad_id} (batch row {bad_at % BATCH_SIZE})",
        "field_path": "k",
        "declared_type": "Int64",
        "bson_type": "String",
        "collection": f"{database}.{collection}",
        "reason": "BSON String cannot be converted to the declared type",
    }


def _assert_contract(
    error: BaseException, database: str, collection: str, bad_at: int, bad_id: str
) -> None:
    """The only passing outcome: class + typed attributes + located message."""
    assert isinstance(error, pm.MongoConversionError), (
        f"expected MongoConversionError, got {type(error).__name__}: {error}"
    )
    assert isinstance(error, pm.MongoError)

    expected = _expected_attributes(database, collection, bad_at, bad_id)
    for attribute, value in expected.items():
        assert getattr(error, attribute) == value, (
            f"attribute {attribute!r} was {getattr(error, attribute, None)!r}, expected {value!r}"
        )
    # Located facts in the message: targeted content, never literal equality.
    message = str(error)
    for fact in ("k", "Int64", "String", collection):
        assert fact in message


class _YieldObserver:
    """Records the batches the engine actually consumed before the failure."""

    def __init__(self) -> None:
        self.batches = 0

    def wrap(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.batches += 1
        return frame


@pytest.mark.mongo
@pytest.mark.parametrize("position", ["first-batch", "later-batch"])
def test_the_reader_raises_the_full_typed_contract_before_the_engine(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    position: str,
) -> None:
    """What the reader raises, measured without the engine in the path.

    The matrix cells above assert the same values on the exception the *caller*
    receives. Driving the reader directly separates the two questions: this test
    pins down what is put into the boundary, so a failing cell can only mean the
    boundary lost it.
    """
    bad_at = 0 if position == "first-batch" else GOOD_ROWS - 1
    collection, bad_id = _seed(raw, clean_db, bad_at)

    reader = MongoBatchReader(
        mongo_conn, clean_db, collection, SCHEMA, None, None, None, BATCH_SIZE
    )
    with pytest.raises(pm.MongoConversionError) as excinfo:
        for _ in reader:
            pass

    _assert_contract(excinfo.value, clean_db, collection, bad_at, bad_id)


@pytest.mark.mongo
@pytest.mark.parametrize("position", ["first-batch", "later-batch"])
def test_read_mongo_default_collect_transports_the_typed_error(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    position: str,
) -> None:
    """`read_mongo(...)` on the path a plain default `collect()` takes."""
    bad_at = 0 if position == "first-batch" else GOOD_ROWS - 1
    collection, bad_id = _seed(raw, clean_db, bad_at)

    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011 - the class IS the assertion
        read_mongo(mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE)

    _assert_contract(excinfo.value, clean_db, collection, bad_at, bad_id)


@pytest.mark.mongo
@pytest.mark.parametrize("position", ["first-batch", "later-batch"])
def test_scan_mongo_default_collect_transports_the_typed_error(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    position: str,
) -> None:
    """Direct `scan_mongo(...).collect()` carries the identical contract."""
    bad_at = 0 if position == "first-batch" else GOOD_ROWS - 1
    collection, bad_id = _seed(raw, clean_db, bad_at)

    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011 - the class IS the assertion
        scan_mongo(mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE).collect()

    _assert_contract(excinfo.value, clean_db, collection, bad_at, bad_id)


@pytest.mark.mongo
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("position", ["first-batch", "later-batch"])
@pytest.mark.parametrize("surface", ["read_mongo", "scan_mongo.collect"])
def test_each_supported_execution_path_transports_the_typed_error(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    position: str,
    surface: str,
) -> None:
    """The same default surfaces, pinned to each path a default collect can take.

    The engine is selected through the environment rather than by calling a
    different public API, so every cell exercises the *default* surface.
    """
    bad_at = 0 if position == "first-batch" else GOOD_ROWS - 1
    collection, bad_id = _seed(raw, clean_db, bad_at)
    monkeypatch.setenv("POLARS_ENGINE_AFFINITY", engine)
    monkeypatch.setenv("POLARS_AUTO_NEW_STREAMING", "1" if engine == "streaming" else "0")

    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011 - the class IS the assertion
        if surface == "read_mongo":
            read_mongo(mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE)
        else:
            scan_mongo(
                mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE
            ).collect()

    _assert_contract(excinfo.value, clean_db, collection, bad_at, bad_id)


@pytest.mark.mongo
def test_the_later_batch_cell_fails_after_an_observed_successful_yield(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mid-stream contextualization is only exercised if a batch really yielded.

    The observation is made *inside the failing invocation*: the io-source
    generator is wrapped so each frame handed to the engine is counted before
    the failing batch is reached.
    """
    bad_at = GOOD_ROWS - 1
    collection, bad_id = _seed(raw, clean_db, bad_at)
    observer = _YieldObserver()

    from polars.io import plugins

    original = plugins.register_io_source

    def counting_register(callable_: Any, *, schema: Any) -> pl.LazyFrame:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            for frame in callable_(*args, **kwargs):
                yield observer.wrap(frame)

        return original(wrapped, schema=schema)

    monkeypatch.setattr(plugins, "register_io_source", counting_register)
    monkeypatch.setattr("polars_mongo._read.register_io_source", counting_register)

    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011 - the class IS the assertion
        read_mongo(mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE)

    assert observer.batches >= 1, "the later-batch cell never yielded a batch first"
    _assert_contract(excinfo.value, clean_db, collection, bad_at, bad_id)


@pytest.mark.mongo
def test_unrelated_engine_errors_propagate_unchanged(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
) -> None:
    """No annotation, no re-typing: the engine's own errors stay its own."""
    collection, _ = _seed(raw, clean_db, bad_at=-1)
    frame = scan_mongo(mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE)
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        frame.select(pl.col("no_such_column")).collect()


@pytest.mark.mongo
def test_original_instance_identity_is_recorded_not_required(
    mongo_conn: pm.MongoConnection,
    raw: MongoClient[dict[str, Any]],
    clean_db: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Diagnostic evidence only: class + attributes + message are the contract.

    Identity is measured honestly - the identity of the exception the reader
    raised is compared with the one the caller receives - and the observed
    answer, including a negative one, is recorded without being asserted. Only
    the identity and class name are retained: holding the exception object would
    keep its traceback, and with it the reader and its live cursor, alive past
    the end of the test.
    """
    collection, bad_id = _seed(raw, clean_db, bad_at=0)
    raised: list[tuple[int, str]] = []

    from polars_mongo import _read

    original_reader = _read.MongoBatchReader

    class RecordingReader:  # noqa: D101 - test shim around the real reader
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = original_reader(*args, **kwargs)

        def __iter__(self) -> RecordingReader:
            return self

        def __next__(self) -> Any:
            try:
                return self._inner.__next__()
            except StopIteration:
                raise
            except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
                raised.append((id(error), type(error).__name__))
                raise

    monkeypatch.setattr(_read, "MongoBatchReader", RecordingReader)
    with pytest.raises(BaseException) as excinfo:  # noqa: B017, PT011
        read_mongo(mongo_conn, clean_db, collection, schema=SCHEMA, batch_size=BATCH_SIZE)
    received = (id(excinfo.value), type(excinfo.value).__name__)

    assert raised, "the reader never raised"
    identical = received == raised[0]
    print(
        "error-transport diagnostic: raised is original = "
        f"{identical} (reader raised {raised[0][1]}, caller received {received[1]})"
    )
    assert "raised is original" in capsys.readouterr().out
