"""The write path.

`write_mongo` writes an eager frame in batches. `sink_mongo` evaluates a lazy
frame through Polars' ordered batch sink. Both use the same batch writer and
counter semantics, so indexed write failures identify positions consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polars_mongo._errors import _schema_error
from polars_mongo._internal import MongoBatchWriter, MongoWriteError

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

    from polars_mongo._internal import BatchOutcome, MongoConnection

__all__ = ["sink_mongo", "write_mongo"]

DEFAULT_BATCH_SIZE = 1024


type WriteFailure = tuple[int, str]


def _validate_batch_size(batch_size: int) -> None:
    """Reject values that cannot define a positive MongoDB batch size."""
    if type(batch_size) is not int or batch_size < 1:
        raise _schema_error(
            "batch_size",
            f"must be a positive integer; got {batch_size!r}",
        )


@dataclass
class RowCounter:
    """Track attempted stream positions separately from acknowledged inserts."""

    attempted_rows: int = 0
    inserted_rows: int = 0

    def record_batch(self, attempted: int, inserted: int) -> None:
        """Advance positions by every attempted row and counts by successes only."""
        self.attempted_rows += attempted
        self.inserted_rows += inserted


def _insert_batch(
    writer: MongoBatchWriter,
    counter: RowCounter,
    failures: list[WriteFailure],
    batch: pl.DataFrame,
) -> None:
    """Insert a batch and retain indexed failures while preserving stream offsets."""
    outcome: BatchOutcome = writer.insert_batch(batch, counter.attempted_rows)
    failures.extend(outcome.failures)
    counter.record_batch(batch.height, outcome.inserted)


def _write_error(failures: list[WriteFailure]) -> MongoWriteError:
    details = "; ".join(f"{row_index}: {message}" for row_index, message in failures)
    error = MongoWriteError(
        "MongoDB indexed write failures at row_index values "
        + "(the zero-based ordinal in the inserted stream): "
        + f"{details}"
    )
    error.failures = failures
    return error


def write_mongo(
    connection: MongoConnection,
    database: str,
    collection: str,
    frame: pl.DataFrame,
    *,
    schema: pa.Schema,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Write a DataFrame to MongoDB in unordered batches.

    An indexed failure reports `row_index` as the ordinal of the row within the
    inserted stream, from 0 across the whole call. For a DataFrame, that ordinal
    is the source frame row. `batch_size` is a batching knob, not a memory bound.
    On indexed failures, raises `MongoWriteError` after all batches have been
    attempted; its `failures` attribute contains `(row_index, server_message)`
    tuples. Non-row batch failures propagate immediately as `MongoBatchError`.
    """
    _validate_batch_size(batch_size)
    writer = MongoBatchWriter(connection, database, collection, schema)
    counter = RowCounter()
    failures: list[WriteFailure] = []

    for start in range(0, frame.height, batch_size):
        _insert_batch(writer, counter, failures, frame.slice(start, batch_size))

    if failures:
        raise _write_error(failures)
    return counter.inserted_rows


def sink_mongo(
    connection: MongoConnection,
    database: str,
    collection: str,
    frame: pl.LazyFrame,
    *,
    schema: pa.Schema,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Write a LazyFrame to MongoDB through Polars' ordered batch sink.

    An indexed failure reports `row_index` as the ordinal of the row within the
    inserted stream, from 0 across the whole stream. It is the position in the
    sink's ordered stream; after an unordered group-by or join, it is not the
    ancestral source row. `batch_size` is a batching knob, not a memory bound.
    On indexed failures, raises `MongoWriteError` after the sink finishes; its
    `failures` attribute contains `(row_index, server_message)` tuples. Non-row
    batch failures propagate immediately as `MongoBatchError`.
    """
    _validate_batch_size(batch_size)
    writer = MongoBatchWriter(connection, database, collection, schema)
    counter = RowCounter()
    failures: list[WriteFailure] = []

    def _sink(batch: pl.DataFrame) -> None:
        _insert_batch(writer, counter, failures, batch)

    frame.sink_batches(_sink, chunk_size=batch_size, maintain_order=True)

    if failures:
        raise _write_error(failures)
    return counter.inserted_rows
