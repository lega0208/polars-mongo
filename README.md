# polars-mongo

MongoDB IO and ObjectId expressions for [Polars](https://pola.rs), implemented
in Rust and packaged with [maturin](https://www.maturin.rs).

## Quickstart

```bash
make dev     # uv sync + maturin develop --uv
make test    # cargo test + pytest
make lint    # fmt, clippy, ruff, ty
```

`object_id_timestamp` operates on binary ObjectId storage, not hex strings.
The following example is serverless and evaluates to the shown result.

```python
from datetime import datetime

import polars as pl
import polars_mongo as pm

hex_ids = pl.DataFrame({"_id": ["507f1f77bcf86cd799439011", "nope"]})
object_ids = hex_ids.select(pm.object_id_from_hex("_id"))
result = object_ids.select(
    _id=pl.col("_id").mongo.to_hex(),
    created_at=pl.col("_id").mongo.object_id_timestamp(),
)

assert result.to_dicts() == [
    {"_id": "507f1f77bcf86cd799439011", "created_at": datetime(2012, 10, 17, 21, 13, 27)},
    {"_id": None, "created_at": None},
]
```

`is_object_id` remains the Utf8-only validator for 24-character hex strings.
Use `.mongo.from_hex()` to make binary storage and `.mongo.to_hex()` to render
it. These are expression namespace methods; they do not monkeypatch Polars.

## IO API

```python
read_mongo(connection, database, collection, *, schema, filter=None,
           projection=None, projection_schema=None, limit=None) -> pl.DataFrame
scan_mongo(connection, database, collection, *, schema, filter=None,
           projection=None, projection_schema=None, limit=None) -> pl.LazyFrame
write_mongo(connection, database, collection, frame, *, schema,
            batch_size=1024) -> int
sink_mongo(connection, database, collection, frame, *, schema,
           batch_size=1024) -> int
```

`read_mongo` eagerly collects the lazy `scan_mongo` result. `write_mongo` and
`sink_mongo` return the inserted-row count only on full success. `batch_size`
is a batching knob, not a memory bound.

For `MongoWriteError`, `row_index` is the zero-based ordinal in the inserted
stream across the entire call or stream. For `write_mongo` it is the source
frame row; for `sink_mongo`, an unordered group-by or join may make it differ
from the ancestral source row.

## Errors

All errors derive from `MongoError`. Programmatic recovery must use the typed
attributes, not error-message text.

| Exception | Typed attributes |
| --- | --- |
| `MongoConnectionClosedError` | `uri` |
| `MongoSchemaError` | `field_path`, `reason` |
| `MongoConversionError` | `document_id`, `field_path`, `declared_type`, `bson_type`, `collection`, `reason` |
| `MongoWriteError` | `failures`: `list[tuple[int, str]]` of `(row_index, server_message)` |
| `MongoBatchError` | `row_start`, `row_stop`, `cause` |

## Schema to BSON mapping

The supplied PyArrow schema is the contract; types are not inferred from data.

| PyArrow declaration | Written BSON type | Accepted BSON types | Notes |
| --- | --- | --- | --- |
| `int32` | Int32 | Int32 | |
| `int64` | Int64 | Int32, Int64 | Int32 widens |
| `float64` | Double | Double, Int32, Int64, Decimal128 | Decimal128 input is lossy |
| `BsonDecimal128Type` (storage `float64`) | Decimal128 | Decimal128, Double, Int32, Int64 | Lossy |
| `bool_` | Boolean | Boolean | |
| `string` | String | String | |
| `binary` | Binary subtype 0 | Binary subtype 0 | |
| `timestamp("ms")` | DateTime | DateTime, Timestamp | Timestamp increment is discarded |
| `BsonTimestampType` (storage `timestamp("ms")`) | Timestamp (`i = 0`) | Timestamp, DateTime | Increment is not preserved |
| `ObjectIdType` | ObjectId | ObjectId | Byte-exact |
| `struct_` | Embedded document | Embedded document | |
| `list_(T)` | Array of `T` | Array of `T` | |
| `list_(struct_)` | Array of documents | Array of documents | |

`ObjectIdType`, `BsonDecimal128Type`, and `BsonTimestampType` are exported
PyArrow extension types that select their BSON write targets.

## IO execution context

`MongoConnection` may be shared between threads. Calling any IO entrypoint from
a thread with an active Tokio runtime context is unsupported; use a plain
non-async thread. This package documents rather than detects runtime contexts,
and unrelated engine errors propagate unchanged.

## AC-9 observation limitation

AC-9's outgoing-document observation is blocked on MongoDB 8.0.4: neither
`system.profile` nor the structured mongod log exposes insert documents. The
MongoDB driver prepends `_id` to every outgoing document.

## Layout

| Path | Purpose |
| --- | --- |
| `src/lib.rs` | PyO3 `_internal` module and plugin registration |
| `src/api.rs` | Schema/ObjectId Python API and test transport probes |
| `src/conn.rs` | `MongoConnection` lifecycle and client access |
| `src/errors.rs` | Exception hierarchy and typed error construction |
| `src/read.rs` / `src/write.rs` | Rust batch read and write primitives |
| `src/expressions.rs` / `src/objectid.rs` | Polars expressions and ObjectId transport |
| `src/bson_arrow/` | Schema planning and BSON/Arrow conversion |
| `python/polars_mongo/__init__.py` | Public Python exports and expression namespace |
| `python/polars_mongo/_read.py` / `_write.py` | Python IO entrypoints |
| `python/polars_mongo/_objectid.py` | Registered Polars and PyArrow extension types |
| `python/polars_mongo/_errors.py` | Exception re-exports |
| `python/polars_mongo/_internal.pyi` | Compiled extension module type declarations |
| `python/polars_mongo/py.typed` | PEP 561 typing marker |
| `tests/` | Serverless and MongoDB integration tests |
| `Cargo.toml` / `pyproject.toml` | Rust and Python package metadata |
