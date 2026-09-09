"""ObjectId as one registered extension identity, on both the pyarrow and the
Polars side.

There is exactly one transport: the Rust builder constructs the extension dtype
at *every* ObjectId node and the recursive Arrow C schema export carries the
name across the boundary. Nothing here repairs a lost name after the fact.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, override

import polars as pl
import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import PolarsDataType

OBJECT_ID_EXT_NAME = "polars_mongo.object_id"
BSON_DECIMAL128_EXT_NAME = "polars_mongo.bson_decimal128"
BSON_TIMESTAMP_EXT_NAME = "polars_mongo.bson_timestamp"

OBJECT_ID_STORAGE = pa.list_(pa.uint8(), 12)


class ObjectIdType(pa.ExtensionType):
    """A BSON ObjectId: 12 raw bytes, the only byte-exact round trip."""

    def __init__(self) -> None:
        super().__init__(OBJECT_ID_STORAGE, OBJECT_ID_EXT_NAME)

    @override
    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @override
    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> ObjectIdType:
        return cls()


class BsonDecimal128Type(pa.ExtensionType):
    """Selects BSON `Decimal128` as the write target. Storage is `float64`."""

    def __init__(self) -> None:
        super().__init__(pa.float64(), BSON_DECIMAL128_EXT_NAME)

    @override
    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @override
    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> BsonDecimal128Type:
        return cls()


class BsonTimestampType(pa.ExtensionType):
    """Selects BSON `Timestamp` as the write target. Storage is `timestamp("ms")`."""

    def __init__(self) -> None:
        super().__init__(pa.timestamp("ms"), BSON_TIMESTAMP_EXT_NAME)

    @override
    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @override
    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> BsonTimestampType:
        return cls()


# The Polars extension API is unstable (documented as such upstream), so its
# absence is a hard failure rather than a silent degradation: without it the
# ObjectId dtype would quietly collapse to its storage type.
assert hasattr(pl, "register_extension_type"), (
    "this polars build has no `register_extension_type`; polars-mongo requires "
    "the (unstable) extension-type API"
)
assert hasattr(pl, "BaseExtension"), (
    "this polars build has no `BaseExtension`; polars-mongo "
    "requires the (unstable) extension-type API"
)


class ObjectId(pl.BaseExtension):
    """The Polars side of the same extension identity."""

    def __init__(self) -> None:
        super().__init__(OBJECT_ID_EXT_NAME, pl.Array(pl.UInt8, 12))


class BsonDecimal128(pl.BaseExtension):
    """The Polars side of the BSON `Decimal128` write target."""

    def __init__(self) -> None:
        super().__init__(BSON_DECIMAL128_EXT_NAME, pl.Float64)


class BsonTimestamp(pl.BaseExtension):
    """The Polars side of the BSON `Timestamp` write target."""

    def __init__(self) -> None:
        super().__init__(BSON_TIMESTAMP_EXT_NAME, pl.Datetime("ms"))


_POLARS_TYPES: dict[str, type[Any]] = {
    OBJECT_ID_EXT_NAME: ObjectId,
    BSON_DECIMAL128_EXT_NAME: BsonDecimal128,
    BSON_TIMESTAMP_EXT_NAME: BsonTimestamp,
}

_PYARROW_TYPES: dict[str, Callable[[], pa.ExtensionType]] = {
    OBJECT_ID_EXT_NAME: ObjectIdType,
    BSON_DECIMAL128_EXT_NAME: BsonDecimal128Type,
    BSON_TIMESTAMP_EXT_NAME: BsonTimestampType,
}


def _register() -> None:
    """Register both sides. Idempotent: importing twice is not an error."""
    for name, polars_type in _POLARS_TYPES.items():
        if pl.get_extension_type(name) is None:
            pl.register_extension_type(name, polars_type)
    for arrow_type in _PYARROW_TYPES.values():
        # Already registered in this interpreter: registration is idempotent.
        with contextlib.suppress(pa.ArrowKeyError):
            pa.register_extension_type(arrow_type())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]


_register()


def object_id_dtype() -> PolarsDataType:
    """The Polars dtype an ObjectId column carries."""
    return ObjectId()
