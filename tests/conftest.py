"""Ephemeral mongod fixture and the independent inspection oracle (plan section 4).

Two mechanisms, both independent of this package's conversion code:

1. `pymongo` as the seed/inspect oracle (dev-only, never a runtime dependency).
2. Server-side profiling scoped by `appName`, which observes what the *Rust*
   driver emits. A client-side pymongo listener would only see pymongo's own
   connections.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import polars_mongo as pm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pymongo import MongoClient

pytest_plugins = ["pytester", "_require_mongo_gate"]

TEST_DB = "polars_mongo_test"
RUST_APP_NAME = "polars-mongo-rust"
PYMONGO_APP_NAME = "polars-mongo-pymongo"


def _mongod_binary() -> str | None:
    override = os.environ.get("POLARS_MONGO_TEST_MONGOD")
    if override:
        return override if Path(override).exists() else None
    return shutil.which("mongod")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MongodProcess:
    """A self-managed mongod: we own the port, dbpath, profiling and test commands."""

    def __init__(self, binary: str, tmp_path: Path) -> None:
        self.binary: str = binary
        self.port: int = _free_port()
        self.dbpath: Path = tmp_path / "db"
        self.dbpath.mkdir(parents=True, exist_ok=True)
        self.logpath: Path = tmp_path / "mongod.log"
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def host_port(self) -> str:
        return f"127.0.0.1:{self.port}"

    def uri(self, app_name: str) -> str:
        return f"mongodb://{self.host_port}/?appName={app_name}"

    def start(self) -> None:
        self.process = subprocess.Popen(  # noqa: S603
            [
                self.binary,
                "--dbpath",
                str(self.dbpath),
                "--port",
                str(self.port),
                "--bind_ip",
                "127.0.0.1",
                "--profile",
                "2",
                "--slowms",
                "0",
                "--setParameter",
                "enableTestCommands=1",
                "--logpath",
                str(self.logpath),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_until_healthy()

    def _wait_until_healthy(self, timeout: float = 60.0) -> None:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError

        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"mongod exited early with code {self.process.returncode}; "
                    + f"log at {self.logpath}"
                )
            try:
                client: MongoClient[dict[str, Any]] = MongoClient(
                    f"mongodb://{self.host_port}/?appName=healthcheck",
                    serverSelectionTimeoutMS=500,
                    directConnection=True,
                )
                client.admin.command("ping")
                client.close()
                return
            except PyMongoError as err:  # pragma: no cover - timing dependent
                last_error = err
                time.sleep(0.1)
        raise RuntimeError(f"mongod never became healthy; log at {self.logpath}: {last_error}")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            self.process.kill()
            self.process.wait(timeout=30)
        self.process = None


@pytest.fixture(scope="session")
def mongod(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[MongodProcess]:
    """A session-scoped ephemeral mongod with profiling and test commands enabled."""
    binary = _mongod_binary()
    if binary is None:
        message = (
            "mongod not found on PATH or at POLARS_MONGO_TEST_MONGOD; "
            "server-backed tests cannot run"
        )
        if request.config.getoption("--require-mongo"):
            raise RuntimeError(message)
        pytest.skip(message)
    server = MongodProcess(binary, tmp_path_factory.mktemp("mongod"))
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def mongod_logpath(mongod: MongodProcess) -> Path:
    """The fixture-owned structured mongod log, the specified alternative observer."""
    return mongod.logpath


@pytest.fixture
def rust_app_name() -> str:
    """The `appName` every `MongoConnection` built by the fixture carries."""
    return RUST_APP_NAME


@pytest.fixture
def pymongo_app_name() -> str:
    """The `appName` the independent pymongo oracle carries."""
    return PYMONGO_APP_NAME


@pytest.fixture
def mongo_uri(mongod: MongodProcess) -> str:
    """The URI carrying the Rust client's distinctive appName."""
    return mongod.uri(RUST_APP_NAME)


@pytest.fixture
def mongo_conn(mongo_uri: str) -> Iterator[pm.MongoConnection]:
    """A real `MongoConnection` whose operations the server records by appName."""
    conn = pm.MongoConnection(mongo_uri)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def raw(mongod: MongodProcess) -> Iterator[MongoClient[dict[str, Any]]]:
    """The independent pymongo oracle. Never `polars_mongo` itself."""
    from pymongo import MongoClient

    client: MongoClient[dict[str, Any]] = MongoClient(
        mongod.uri(PYMONGO_APP_NAME), directConnection=True
    )
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def clean_db(raw: MongoClient[dict[str, Any]]) -> Iterator[str]:
    """Drop the test database, reset the profile collection and clear fail points."""

    def reset() -> None:
        db = raw[TEST_DB]
        failpoint_off = raw.admin.command("configureFailPoint", "failCommand", mode="off")
        assert failpoint_off["ok"] == 1
        db.command("profile", 0)
        raw.drop_database(TEST_DB)
        db.command("profile", 2, slowms=0)

    reset()
    yield TEST_DB
    reset()


@pytest.fixture
def profile(raw: MongoClient[dict[str, Any]], clean_db: str) -> Callable[..., list[dict[str, Any]]]:
    """Profiler entries for one `appName`.

    `system.profile` is per-database, so the database is a parameter: operations
    against `admin` (such as the lifecycle ping) are recorded there, not in the
    test database.
    """
    start = time.time()

    def query(
        app_name: str = RUST_APP_NAME,
        *,
        database: str | None = None,
        namespace: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        import datetime as dt

        cutoff = dt.datetime.fromtimestamp(since if since is not None else start, tz=dt.UTC)
        criteria: dict[str, Any] = {
            "appName": app_name,
            "ts": {"$gte": cutoff},
        }
        if namespace is not None:
            criteria["ns"] = namespace
        target = raw[database if database is not None else clean_db]
        return list(target["system.profile"].find(criteria).sort("ts", 1))

    return query
