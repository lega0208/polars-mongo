"""`MongoConnection` lifecycle. Phase 1 covers the serverless half."""

from __future__ import annotations

import pytest

import polars_mongo as pm

URI = "mongodb://127.0.0.1:27099/?appName=polars-mongo-unit"


def test_constructs_without_a_server() -> None:
    # The driver connects lazily, so construction must not contact anything.
    conn = pm.MongoConnection(URI)
    assert conn.closed is False
    conn.close()


def test_close_is_idempotent() -> None:
    conn = pm.MongoConnection(URI)
    conn.close()
    conn.close()
    assert conn.closed is True


def test_closed_accessor_raises_with_the_uri() -> None:
    conn = pm.MongoConnection(URI)
    conn.close()
    with pytest.raises(pm.MongoConnectionClosedError) as excinfo:
        conn._test_ping()
    assert excinfo.value.uri == URI
    message = str(excinfo.value)
    assert URI in message
    assert "closed" in message
    assert "open at collect time" in message


def test_context_manager_closes_on_exit() -> None:
    with pm.MongoConnection(URI) as conn:
        assert conn.closed is False
    assert conn.closed is True


def test_context_manager_propagates_exceptions() -> None:
    conn = pm.MongoConnection(URI)
    with pytest.raises(ValueError), conn:
        raise ValueError("boom")
    assert conn.closed is True


@pytest.mark.mongo
def test_context_manager_closes_client(mongo_uri: str) -> None:
    """AC-1, proven against *this* Rust client rather than pymongo."""
    with pm.MongoConnection(mongo_uri) as conn:
        conn._test_ping()
    with pytest.raises(pm.MongoConnectionClosedError) as excinfo:
        conn._test_ping()
    assert excinfo.value.uri == mongo_uri
    assert "closed" in str(excinfo.value)
