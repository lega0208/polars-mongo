"""The read path.

`read_mongo` is literally `scan_mongo(...).collect()`: there is one conversion
engine and one execution path, so the eager and lazy surfaces cannot diverge.

Error transport
---------------
A conversion failure raised inside the `register_io_source` callback must reach
the caller as the typed `MongoConversionError` it was raised as - same class,
same `document_id`, `field_path`, `declared_type`, `bson_type`, `collection` and
`reason` attributes - on both public surfaces and on every execution path a
default `collect()` can take.

That is a property of the Polars version in use, not something this module can
compensate for: released Polars through 1.44.1 (and the 2.0.0-rc.1 tag) rewrote
any non-`StopIteration` callback exception into a string `ComputeError` in
`polars-mem-engine/src/executors/scan/python_scan.rs` and
`polars-stream/src/physical_plan/to_graph.rs`, which destroyed the class and
every typed attribute. Upstream now propagates the original error
(`Err(err) => return Err(err.into())`), so this project pins Polars to that
revision; see `Cargo.toml` and `pyproject.toml`.

`tests/test_error_transport.py` is the permanent gate for the property. It ships
with the package so a Polars upgrade that re-breaks the boundary fails loudly
instead of silently degrading the shipped default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa
from polars.io.plugins import register_io_source

from polars_mongo._errors import _schema_error
from polars_mongo._internal import MongoBatchReader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from polars_mongo._internal import MongoConnection

__all__ = ["MongoBatchReader", "read_mongo", "scan_mongo"]

DEFAULT_BATCH_SIZE = 1024


def _validate_batch_size(batch_size: int) -> None:
    """Reject values that cannot define a positive MongoDB batch size."""
    if type(batch_size) is not int or batch_size < 1:
        raise _schema_error(
            "batch_size",
            f"must be a positive integer; got {batch_size!r}",
        )


def _resolve_schema(
    schema: pa.Schema | None, projection_schema: pa.Schema | None, projection: dict[str, Any] | None
) -> pa.Schema:
    """Validate the caller's schema arguments before any connection is used."""
    if projection is not None and projection_schema is None:
        raise _schema_error(
            "projection",
            "requires projection_schema: the projected documents no longer "
            + "match `schema`, so the output contract would be unstated",
        )
    if projection_schema is not None:
        # `projection_schema` fully replaces `schema`; there is one output
        # contract, never a merge of two.
        return projection_schema
    if schema is None:
        raise _schema_error(
            "schema",
            "a pyarrow schema is required: polars-mongo never infers one from data",
        )
    return schema


def _arrow_to_polars_schema(schema: pa.Schema) -> pl.Schema:
    # `schema.empty_table()` raises `ArrowNotImplementedError: extension` for an
    # extension type nested inside a Struct or List; building a zero-batch table
    # keeps the nested extension identity intact.
    empty = pl.from_arrow(pa.Table.from_batches([], schema=schema))
    assert isinstance(empty, pl.DataFrame)
    return empty.schema


def scan_mongo(
    connection: MongoConnection,
    database: str,
    collection: str,
    *,
    schema: pa.Schema | None = None,
    filter: dict[str, Any] | None = None,  # noqa: A002 - mirrors the MongoDB argument
    projection: dict[str, Any] | None = None,
    projection_schema: pa.Schema | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pl.LazyFrame:
    """Lazily scan a MongoDB collection under a caller-supplied pyarrow schema.

    No server operation is issued until the frame is collected. `filter`,
    `projection` and `limit` are sent to the server, so the server-side limit
    applies before anything Polars sees.
    """
    _validate_batch_size(batch_size)
    resolved = _resolve_schema(schema, projection_schema, projection)
    output_schema = _arrow_to_polars_schema(resolved)

    def _source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        _batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        reader = MongoBatchReader(
            connection,
            database,
            collection,
            resolved,
            filter,
            projection,
            limit,
            batch_size,
        )
        # Cumulative across batches: resetting per batch would silently return
        # up to `n_rows` rows *per batch*.
        produced = 0
        for frame in reader:
            if with_columns is not None:
                frame = frame.select(with_columns)
            if predicate is not None:
                frame = frame.filter(predicate)
            if n_rows is not None:
                remaining = n_rows - produced
                if remaining <= 0:
                    return
                if frame.height > remaining:
                    frame = frame.head(remaining)
            produced += frame.height
            yield frame
            if n_rows is not None and produced >= n_rows:
                return

    return register_io_source(_source, schema=output_schema)


def read_mongo(
    connection: MongoConnection,
    database: str,
    collection: str,
    *,
    schema: pa.Schema | None = None,
    filter: dict[str, Any] | None = None,  # noqa: A002 - mirrors the MongoDB argument
    projection: dict[str, Any] | None = None,
    projection_schema: pa.Schema | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pl.DataFrame:
    """Read a MongoDB collection eagerly.

    This is literally `scan_mongo(...).collect()`; there is no second engine and
    no second error path.
    """
    return scan_mongo(
        connection,
        database,
        collection,
        schema=schema,
        filter=filter,
        projection=projection,
        projection_schema=projection_schema,
        limit=limit,
        batch_size=batch_size,
    ).collect()
