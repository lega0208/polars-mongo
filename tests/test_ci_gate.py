"""Self-tests for the `--require-mongo` gate (AC-25, Phase 3).

The gate exists so a server-backed acceptance criterion cannot quietly degrade
into a skip in CI. These run pytest-in-pytest through `pytester` against the
*same* gate implementation the real suite loads, so they need no server.

**Accepted residual gap:** the zero-collected check detects removal of *all*
markers, or a rename that de-selects the whole subset. It does **not** detect a
single test accidentally left unmarked while other tests remain marked.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

GATE_DIR = str(Path(__file__).parent)

CONFTEST = f"""
import sys

sys.path.insert(0, {GATE_DIR!r})

pytest_plugins = ["_require_mongo_gate"]
"""

INI = """
[pytest]
markers =
    mongo: requires a live server
"""


def _write_gate(pytester: pytest.Pytester) -> None:
    """Install the real gate into a sub-session.

    The sub-sessions run in a subprocess: the gate keeps session state in module
    globals, so an in-process run would let one sub-session's skips leak into
    the next one and into the parent suite.
    """
    pytester.makeconftest(CONFTEST)
    pytester.makefile(".ini", pytest=INI)


def test_fails_when_mongo_tests_skipped(pytester: pytest.Pytester) -> None:
    _write_gate(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mongo
        def test_needs_server():
            pytest.skip("no server")
        """
    )
    assert pytester.runpytest_subprocess("--require-mongo").ret != 0


def test_fails_when_no_mongo_tests_collected(pytester: pytest.Pytester) -> None:
    _write_gate(pytester)
    pytester.makepyfile(
        """
        def test_unmarked():
            assert True
        """
    )
    assert pytester.runpytest_subprocess("--require-mongo").ret != 0


def test_passes_on_successful_marked_session(pytester: pytest.Pytester) -> None:
    _write_gate(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mongo
        def test_needs_server():
            assert True
        """
    )
    assert pytester.runpytest_subprocess("--require-mongo").ret == 0


def test_fails_on_marked_deselection(pytester: pytest.Pytester) -> None:
    _write_gate(pytester)
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.mongo
        def test_needs_server():
            assert True

        def test_pure():
            assert True
        """
    )
    assert pytester.runpytest_subprocess("--require-mongo", "-k", "pure").ret != 0
