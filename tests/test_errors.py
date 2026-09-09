"""The fixed five-class exception hierarchy (M3), Phase 0."""

from __future__ import annotations

import pytest

import polars_mongo as pm

SUBCLASSES = [
    pm.MongoConnectionClosedError,
    pm.MongoSchemaError,
    pm.MongoConversionError,
    pm.MongoWriteError,
    pm.MongoBatchError,
]


def test_base_class_is_an_exception() -> None:
    assert issubclass(pm.MongoError, Exception)


@pytest.mark.parametrize("cls", SUBCLASSES)
def test_every_class_subclasses_mongo_error(cls: type[BaseException]) -> None:
    assert issubclass(cls, pm.MongoError)
    assert cls is not pm.MongoError


def test_write_and_batch_errors_are_siblings() -> None:
    # AC-12 must distinguish them in a single `except`, so neither may be a
    # subclass of the other.
    assert not issubclass(pm.MongoWriteError, pm.MongoBatchError)
    assert not issubclass(pm.MongoBatchError, pm.MongoWriteError)


def test_classes_are_exported() -> None:
    for cls in [pm.MongoError, *SUBCLASSES]:
        assert cls.__name__ in pm.__all__
