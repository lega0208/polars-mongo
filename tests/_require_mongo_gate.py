"""The `--require-mongo` gate hooks, in one importable module.

Kept separate from `conftest.py` so the gate self-tests can load the *same*
implementation into a `pytester` sub-session instead of re-describing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

_SKIPPED_MONGO_TESTS: list[str] = []
_COLLECTED_MONGO_TESTS: list[str] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-mongo",
        action="store_true",
        default=False,
        help="Fail instead of skipping when the mongod fixture is unavailable, "
        "and fail the session if any `mongo`-marked test was skipped, "
        "deselected, or not collected at all.",
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    _COLLECTED_MONGO_TESTS.clear()
    _COLLECTED_MONGO_TESTS.extend(
        item.nodeid for item in items if item.get_closest_marker("mongo") is not None
    )


def pytest_deselected(items: list[pytest.Item]) -> None:
    _SKIPPED_MONGO_TESTS.extend(
        f"deselected: {item.nodeid}"
        for item in items
        if item.get_closest_marker("mongo") is not None
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker("mongo") is not None:
        _SKIPPED_MONGO_TESTS.append(f"skipped: {item.nodeid}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("--require-mongo", default=False):
        return
    problems: list[str] = []
    if not _COLLECTED_MONGO_TESTS:
        problems.append("no `mongo`-marked tests were collected")
    problems.extend(_SKIPPED_MONGO_TESTS)
    if not problems:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("--require-mongo gate failed:")
        for problem in problems:
            reporter.write_line(f"  {problem}")
    session.exitstatus = 1
