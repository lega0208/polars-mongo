"""Polars expression plugin for MongoDB-flavoured data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExprColumn

__all__ = ["MongoExprNamespace", "is_object_id", "object_id_timestamp"]

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
    """Extract the creation timestamp embedded in an ObjectId; null when invalid."""
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="object_id_timestamp",
        args=expr,
        kwargs={"time_unit": time_unit},
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
