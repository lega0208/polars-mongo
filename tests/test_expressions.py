from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_mongo as pm

OID = "507f1f77bcf86cd799439011"
OID_CREATED_AT = datetime(2012, 10, 17, 21, 13, 27, tzinfo=timezone.utc).replace(tzinfo=None)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"_id": [OID, "not-an-object-id", None]})


def test_is_object_id(df: pl.DataFrame) -> None:
    out = df.select(pm.is_object_id("_id"))
    assert_frame_equal(out, pl.DataFrame({"_id": [True, False, None]}))


def test_object_id_timestamp(df: pl.DataFrame) -> None:
    out = df.select(pm.object_id_timestamp("_id"))
    expected = pl.DataFrame(
        {"_id": [OID_CREATED_AT, None, None]},
        schema={"_id": pl.Datetime("ms")},
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
def test_object_id_timestamp_time_units(df: pl.DataFrame, time_unit: str) -> None:
    out = df.select(pm.object_id_timestamp("_id", time_unit=time_unit))  # type: ignore[arg-type]
    assert out.schema["_id"] == pl.Datetime(time_unit)  # type: ignore[arg-type]
    assert out.item(0, 0) == OID_CREATED_AT


def test_invalid_time_unit(df: pl.DataFrame) -> None:
    with pytest.raises(pl.exceptions.ComputeError):
        df.select(pm.object_id_timestamp("_id", time_unit="weeks"))  # type: ignore[arg-type]


# `register_expr_namespace` attaches `.mongo` at runtime, which mypy cannot see.
def test_expr_namespace(df: pl.DataFrame) -> None:
    out = df.select(pl.col("_id").mongo.is_object_id())  # type: ignore[attr-defined]
    assert out.to_series().to_list() == [True, False, None]


def test_lazy_streaming_roundtrip(df: pl.DataFrame) -> None:
    out = df.lazy().filter(pl.col("_id").mongo.is_object_id()).collect()  # type: ignore[attr-defined]
    assert out.height == 1
