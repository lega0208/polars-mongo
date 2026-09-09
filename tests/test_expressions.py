from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_mongo as pm

OID = "507f1f77bcf86cd799439011"
OID_CREATED_AT = datetime(2012, 10, 17, 21, 13, 27, tzinfo=UTC).replace(tzinfo=None)

type TimeUnit = Literal["ms", "us", "ns"]
TIME_UNITS: list[TimeUnit] = ["ms", "us", "ns"]


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"_id": [OID, "not-an-object-id", None]})


@pytest.fixture
def object_id_df(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pm.object_id_from_hex("_id"))


def test_is_object_id(df: pl.DataFrame) -> None:
    out = df.select(pm.is_object_id("_id"))
    assert_frame_equal(out, pl.DataFrame({"_id": [True, False, None]}))


def test_object_id_timestamp(object_id_df: pl.DataFrame) -> None:
    out = object_id_df.select(pm.object_id_timestamp("_id"))
    expected = pl.DataFrame(
        {"_id": [OID_CREATED_AT, None, None]},
        schema={"_id": pl.Datetime("ms")},
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("time_unit", TIME_UNITS)
def test_object_id_timestamp_time_units(object_id_df: pl.DataFrame, time_unit: TimeUnit) -> None:
    out = object_id_df.select(pm.object_id_timestamp("_id", time_unit=time_unit))
    assert out.schema["_id"] == pl.Datetime(time_unit)
    assert out.item(0, 0) == OID_CREATED_AT


def test_object_id_timestamp_rejects_utf8(df: pl.DataFrame) -> None:
    """AC-24: hex strings are no longer a valid `object_id_timestamp` input.

    A plugin's `InvalidOperation` reaches the caller as `ComputeError` (the same
    shape `test_invalid_time_unit` observes), so the class assertion is paired
    with the located message content.
    """
    with pytest.raises(pl.exceptions.ComputeError, match="expected ObjectId dtype"):
        df.select(pm.object_id_timestamp("_id"))


def test_invalid_time_unit(object_id_df: pl.DataFrame) -> None:
    with pytest.raises(pl.exceptions.ComputeError):
        object_id_df.select(
            pm.object_id_timestamp("_id", time_unit="weeks")  # ty: ignore[invalid-argument-type]
        )


# `register_expr_namespace` attaches `.mongo` at runtime, which ty cannot see.
def test_expr_namespace(df: pl.DataFrame) -> None:
    out = df.select(pl.col("_id").mongo.is_object_id())  # ty: ignore[unresolved-attribute]
    assert out.to_series().to_list() == [True, False, None]


def test_expr_namespace_timestamp(object_id_df: pl.DataFrame) -> None:
    out = object_id_df.select(
        pl.col("_id").mongo.object_id_timestamp()  # ty: ignore[unresolved-attribute]
    )
    assert out.item(0, 0) == OID_CREATED_AT


def test_lazy_streaming_timestamp(object_id_df: pl.DataFrame) -> None:
    out = (
        object_id_df.lazy()
        .select(
            pl.col("_id").mongo.object_id_timestamp()  # ty: ignore[unresolved-attribute]
        )
        .collect()
    )
    assert out.item(0, 0) == OID_CREATED_AT
