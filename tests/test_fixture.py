"""The ephemeral mongod fixture itself (AC-25, Phase 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pymongo import MongoClient


@pytest.mark.mongo
def test_ephemeral_mongod_accepts_writes(raw: MongoClient[dict[str, Any]], clean_db: str) -> None:
    collection = raw[clean_db]["fixture_smoke"]
    collection.insert_many([{"key": index} for index in range(3)])
    assert sorted(doc["key"] for doc in collection.find({})) == [0, 1, 2]
