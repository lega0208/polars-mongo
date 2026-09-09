"""The fixed five-class exception hierarchy, re-exported from the Rust module."""

from __future__ import annotations

from polars_mongo._internal import (
    MongoBatchError,
    MongoConnectionClosedError,
    MongoConversionError,
    MongoError,
    MongoSchemaError,
    MongoWriteError,
)


def _schema_error(field_path: str, reason: str) -> MongoSchemaError:
    """Build a schema error with the typed attributes shared with Rust errors."""
    error = MongoSchemaError(f"schema error at {field_path}: {reason}")
    error.field_path = field_path
    error.reason = reason
    return error


__all__ = [
    "MongoBatchError",
    "MongoConnectionClosedError",
    "MongoConversionError",
    "MongoError",
    "MongoSchemaError",
    "MongoWriteError",
    "_schema_error",
]
