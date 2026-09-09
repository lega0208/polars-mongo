"""Polars expression plugin for MongoDB-flavoured data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_mongo._errors import (
    MongoBatchError,
    MongoConnectionClosedError,
    MongoConversionError,
    MongoError,
    MongoSchemaError,
    MongoWriteError,
)
from polars_mongo._internal import MongoConnection
from polars_mongo._objectid import (
    BsonDecimal128Type,
    BsonTimestampType,
    ObjectIdType,
)
from polars_mongo._read import read_mongo, scan_mongo
from polars_mongo._write import sink_mongo, write_mongo

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn

__all__ = [
    "BsonDecimal128Type",
    "BsonTimestampType",
    "MongoBatchError",
    "MongoConnection",
    "MongoConnectionClosedError",
    "MongoConversionError",
    "MongoError",
    "MongoSchemaError",
    "MongoWriteError",
    "ObjectIdType",
    "is_object_id",
    "object_id_from_hex",
    "object_id_timestamp",
    "object_id_to_hex",
    "read_mongo",
    "scan_mongo",
    "sink_mongo",
    "write_mongo",
]

try:  # pragma: no cover - trivial metadata lookup
    from importlib.metadata import version as _version

    __version__ = _version("polars-mongo")
except Exception:  # pragma: no cover - source checkout without an install
    __version__ = "0.0.0+unknown"

# Directory holding the compiled `_internal` shared library.
PLUGIN_PATH = Path(__file__).parent

TimeUnit = Literal["ms", "us", "ns"]


def is_object_id(expr: IntoExprColumn) -> pl.Expr:
    """Return `True` where `expr` is a valid 24-character hex ObjectId."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="is_object_id",
        args=expr,
        is_elementwise=True,
    )


def object_id_timestamp(expr: IntoExprColumn, *, time_unit: TimeUnit = "ms") -> pl.Expr:
    """Extract the creation timestamp embedded in a binary ObjectId."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="object_id_timestamp",
        args=expr,
        kwargs={"time_unit": time_unit},
        is_elementwise=True,
    )


def object_id_to_hex(expr: IntoExprColumn) -> pl.Expr:
    """Render a binary ObjectId column as 24-character hex; null stays null."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="object_id_to_hex",
        args=expr,
        is_elementwise=True,
    )


def object_id_from_hex(expr: IntoExprColumn) -> pl.Expr:
    """Parse a hex ObjectId column into its 12-byte storage; invalid becomes null."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="object_id_from_hex",
        args=expr,
        is_elementwise=True,
    )


@pl.api.register_expr_namespace("mongo")
class MongoExprNamespace:
    """`pl.col("_id").mongo.<fn>()` access to the plugin expressions."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr: pl.Expr = expr

    def is_object_id(self) -> pl.Expr:
        return is_object_id(self._expr)

    def object_id_timestamp(self, *, time_unit: TimeUnit = "ms") -> pl.Expr:
        return object_id_timestamp(self._expr, time_unit=time_unit)

    def to_hex(self) -> pl.Expr:
        return object_id_to_hex(self._expr)

    def from_hex(self) -> pl.Expr:
        return object_id_from_hex(self._expr)
