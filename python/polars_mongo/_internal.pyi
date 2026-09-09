"""Type stubs for the compiled `polars_mongo._internal` extension module."""

from types import TracebackType
from typing import Any

import pyarrow as pa
from polars import DataFrame, Series

class MongoError(Exception): ...

class MongoConnectionClosedError(MongoError):
    uri: str

class MongoSchemaError(MongoError):
    field_path: str
    reason: str

class MongoConversionError(MongoError):
    document_id: str
    field_path: str
    declared_type: str
    bson_type: str
    collection: str
    reason: str

class MongoWriteError(MongoError):
    failures: list[tuple[int, str]]

class MongoBatchError(MongoError):
    row_start: int
    row_stop: int
    cause: str

class MongoConnection:
    def __init__(self, uri: str) -> None: ...
    @property
    def uri_(self) -> str: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def __enter__(self) -> MongoConnection: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = ...,
        exc: BaseException | None = ...,
        tb: TracebackType | None = ...,
    ) -> bool: ...
    def _test_ping(self, database: str = ...) -> None: ...

class MongoBatchReader:
    def __init__(
        self,
        connection: MongoConnection,
        database: str,
        collection: str,
        schema: Any,
        filter: dict[str, Any] | None = ...,
        projection: dict[str, Any] | None = ...,
        limit: int | None = ...,
        batch_size: int = ...,
    ) -> None: ...
    def __iter__(self) -> MongoBatchReader: ...
    def __next__(self) -> DataFrame: ...

class BatchOutcome:
    @property
    def attempted(self) -> int: ...
    @property
    def inserted(self) -> int: ...
    @property
    def failures(self) -> list[tuple[int, str]]: ...

class MongoBatchWriter:
    def __init__(
        self,
        connection: MongoConnection,
        database: str,
        collection: str,
        schema: pa.Schema,
    ) -> None: ...
    def insert_batch(self, frame: DataFrame, row_offset: int) -> BatchOutcome: ...

def _detach_counters() -> tuple[int, int]: ...
def compile_schema_plan(schema: Any) -> list[tuple[str, str, str, str]]: ...
def filter_to_extended_json(filter: dict[str, Any]) -> str: ...
def object_id_bytes_to_hex(raw: bytes) -> str: ...
def object_id_hex_to_bytes(hex: str) -> bytes: ...
def object_id_probe_series(kind: str) -> Series: ...
def series_extension_names(series: Series) -> list[str]: ...
