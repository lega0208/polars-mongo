"""Independently specified expectations for the read-path acceptance tests.

Nothing in this module calls the code under test. Every expected frame is built
from hand-written literal row data and a pyarrow schema declared here, then
materialised through `pyarrow` -> `polars`. `tests/test_read.py` seeds MongoDB
from the same literals through `pymongo`, so an expectation can never be derived
from `polars_mongo`'s own conversion engine: a regression in that engine shows up
as a mismatch instead of moving the target.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pyarrow as pa

__all__ = [
    "BASE_ROWS",
    "BASE_SCHEMA",
    "INCOMPATIBLE_BASE_SCHEMA",
    "MATRIX",
    "PROJECTED_SCHEMA",
    "PROJECTION",
    "WIDE_ROWS",
    "MatrixCase",
    "expected_frame",
    "wide_row",
]

_BASE_FIELDS: list[pa.Field[Any]] = [
    pa.field("k", pa.int64()),
    pa.field("name", pa.string()),
    pa.field("score", pa.float64()),
    pa.field("active", pa.bool_()),
]
BASE_SCHEMA = pa.schema(_BASE_FIELDS)

_PROJECTED_FIELDS: list[pa.Field[Any]] = [
    pa.field("k", pa.int64()),
    pa.field("name", pa.string()),
]
PROJECTED_SCHEMA = pa.schema(_PROJECTED_FIELDS)

#: Sent to the server for every projected case: `_id` is suppressed so the
#: projected documents match `PROJECTED_SCHEMA` exactly.
PROJECTION: dict[str, Any] = {"k": 1, "name": 1, "_id": 0}

_INCOMPATIBLE_FIELDS: list[pa.Field[Any]] = [
    # `name` is a BSON string in every seeded document, so declaring it Int64
    # cannot be converted: if `projection_schema` were merged with (or fell back
    # to) this schema, the read would raise `MongoConversionError` instead of
    # returning the projected frame.
    pa.field("name", pa.int64()),
    pa.field("nonexistent_column", pa.string()),
]
INCOMPATIBLE_BASE_SCHEMA = pa.schema(_INCOMPATIBLE_FIELDS)

#: The seed corpus, written out in full rather than generated, so the expected
#: values are literals a reader can check by eye.
BASE_ROWS: list[dict[str, Any]] = [
    {"k": 0, "name": "alpha", "score": 0.5, "active": True},
    {"k": 1, "name": "bravo", "score": 1.5, "active": False},
    {"k": 2, "name": "charlie", "score": 2.5, "active": True},
    {"k": 3, "name": "delta", "score": 3.5, "active": False},
    {"k": 4, "name": "echo", "score": 4.5, "active": True},
    {"k": 5, "name": "foxtrot", "score": 5.5, "active": False},
    {"k": 6, "name": "golf", "score": 6.5, "active": True},
    {"k": 7, "name": "hotel", "score": 7.5, "active": False},
]


def expected_frame(rows: list[dict[str, Any]], schema: pa.Schema) -> pl.DataFrame:
    """Materialise literal `rows` under `schema` without touching `polars_mongo`."""
    frame = pl.from_arrow(pa.Table.from_pylist(rows, schema=schema))
    assert isinstance(frame, pl.DataFrame)
    return frame


def _project(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"k": row["k"], "name": row["name"]} for row in rows]


class MatrixCase:
    """One cell of the filter x projection x limit acceptance matrix."""

    def __init__(
        self,
        case_id: str,
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002 - mirrors the API argument
        projected: bool = False,
        limit: int | None = None,
        rows: list[dict[str, Any]],
    ) -> None:
        self.id = case_id
        self.filter = filter
        self.projected = projected
        self.limit = limit
        self.rows = rows

    @property
    def projection(self) -> dict[str, Any] | None:
        return PROJECTION if self.projected else None

    @property
    def schema(self) -> pa.Schema | None:
        """The base `schema` argument; `None` whenever a projection is used."""
        return None if self.projected else BASE_SCHEMA

    @property
    def projection_schema(self) -> pa.Schema | None:
        return PROJECTED_SCHEMA if self.projected else None

    @property
    def expected(self) -> pl.DataFrame:
        return expected_frame(self.rows, PROJECTED_SCHEMA if self.projected else BASE_SCHEMA)


#: Filter x projection x limit. The expected rows are written out per case, so a
#: case cannot silently agree with a broken implementation by sharing its logic.
MATRIX: list[MatrixCase] = [
    MatrixCase("plain", rows=BASE_ROWS),
    MatrixCase("limit", limit=3, rows=BASE_ROWS[:3]),
    MatrixCase(
        "filter",
        filter={"k": {"$gte": 5}},
        rows=[
            {"k": 5, "name": "foxtrot", "score": 5.5, "active": False},
            {"k": 6, "name": "golf", "score": 6.5, "active": True},
            {"k": 7, "name": "hotel", "score": 7.5, "active": False},
        ],
    ),
    MatrixCase(
        "filter-and-limit",
        filter={"active": True},
        limit=2,
        rows=[
            {"k": 0, "name": "alpha", "score": 0.5, "active": True},
            {"k": 2, "name": "charlie", "score": 2.5, "active": True},
        ],
    ),
    MatrixCase("projection", projected=True, rows=_project(BASE_ROWS)),
    MatrixCase("projection-and-limit", projected=True, limit=4, rows=_project(BASE_ROWS[:4])),
    MatrixCase(
        "projection-and-filter",
        projected=True,
        filter={"k": {"$lt": 3}},
        rows=[
            {"k": 0, "name": "alpha"},
            {"k": 1, "name": "bravo"},
            {"k": 2, "name": "charlie"},
        ],
    ),
    MatrixCase(
        "projection-filter-and-limit",
        projected=True,
        filter={"active": False},
        limit=2,
        rows=[
            {"k": 1, "name": "bravo"},
            {"k": 3, "name": "delta"},
        ],
    ),
    MatrixCase("filter-matching-nothing", filter={"k": {"$gte": 10_000}}, rows=[]),
]

#: `projection_schema` is the sole output contract (AC-15): the expected frame is
#: the projected one even though the base `schema` argument is unusable.
SOLE_CONTRACT_EXPECTED_ROWS: list[dict[str, Any]] = _project(BASE_ROWS)


def wide_row(k: int) -> dict[str, Any]:
    """One row of the deterministic wide corpus used by the window tests."""
    return {"k": k, "name": f"row-{k:04d}", "score": float(k) + 0.25, "active": k % 3 == 0}


#: A larger deterministic corpus for the limit-window and GIL tests. Generated
#: from `wide_row`, whose definition is the specification.
WIDE_ROWS: list[dict[str, Any]] = [wide_row(k) for k in range(20)]
