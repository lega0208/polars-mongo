from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_release = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "release.py"))
ReleaseVersionError = _release["ReleaseVersionError"]
main = _release["main"]
select_release_version = _release["select_release_version"]
update_version_files = _release["update_version_files"]


def test_equal_release_version_bumps_patch() -> None:
    assert select_release_version("1.2.3", "v1.2.3") == "1.2.4"


def test_higher_explicit_version_is_unchanged() -> None:
    assert select_release_version("1.2.4", "v1.2.3") == "1.2.4"


@pytest.mark.parametrize(
    ("current", "latest"),
    [
        ("1.2.2", "v1.2.3"),
        ("1.2", "v1.2.3"),
        ("1.2.3-alpha", "v1.2.3"),
        ("1.2.3", "latest"),
        ("1.2.3", "v1.2"),
    ],
)
def test_lower_or_malformed_versions_are_rejected(current: str, latest: str) -> None:
    with pytest.raises(ReleaseVersionError):
        select_release_version(current, latest)


def test_initial_release_uses_explicit_version() -> None:
    assert select_release_version("0.1.0", None) == "0.1.0"


def test_initial_release_cli_synchronizes_stale_metadata(tmp_path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "polars-mongo"\nversion = "0.1.0"\n\n[dependencies]\nserde = "1"\n'
    )
    (tmp_path / "Cargo.lock").write_text(
        'version = 4\n\n[[package]]\nname = "polars-mongo"\nversion = "0.0.9"\n'
        'dependencies = ["serde"]\n\n[[package]]\nname = "serde"\nversion = "1.0.0"\n'
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "polars-mongo"\nsource = { editable = "." }\n'
        'version = "0.0.9"\ndependencies = [{ name = "serde" }]\n\n'
        '[[package]]\nname = "serde"\nversion = "1.0.0"\n'
    )

    assert main(["--root", str(tmp_path)]) == 0
    assert 'version = "0.1.0"' in (tmp_path / "Cargo.lock").read_text()
    assert 'version = "0.1.0"' in (tmp_path / "uv.lock").read_text()
    assert 'dependencies = ["serde"]' in (tmp_path / "Cargo.lock").read_text()
    assert '{ name = "serde" }' in (tmp_path / "uv.lock").read_text()
    assert 'name = "serde"\nversion = "1.0.0"' in (tmp_path / "Cargo.lock").read_text()
    assert 'name = "serde"\nversion = "1.0.0"' in (tmp_path / "uv.lock").read_text()


def test_initial_release_cli_preserves_omitted_uv_version(tmp_path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "polars-mongo"\nversion = "0.1.0"\n\n[dependencies]\nserde = "1"\n'
    )
    (tmp_path / "Cargo.lock").write_text(
        'version = 4\n\n[[package]]\nname = "polars-mongo"\nversion = "0.0.9"\n'
        'dependencies = ["serde"]\n\n[[package]]\nname = "serde"\nversion = "1.0.0"\n'
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "polars-mongo"\nsource = { editable = "." }\n'
        'dependencies = [{ name = "serde" }]\n\n[[package]]\nname = "serde"\n'
        'version = "1.0.0"\n'
    )

    assert main(["--root", str(tmp_path)]) == 0
    assert 'version = "0.1.0"' in (tmp_path / "Cargo.lock").read_text()
    uv_lock = (tmp_path / "uv.lock").read_text()
    assert 'name = "polars-mongo"\nsource = { editable = "." }\nversion =' not in uv_lock
    assert '{ name = "serde" }' in uv_lock
    assert 'name = "serde"\nversion = "1.0.0"' in uv_lock


def test_bump_synchronizes_metadata_without_dependency_changes(tmp_path) -> None:
    cargo_toml = tmp_path / "Cargo.toml"
    cargo_lock = tmp_path / "Cargo.lock"
    uv_lock = tmp_path / "uv.lock"
    cargo_toml.write_text(
        '[package]\nname = "polars-mongo"\nversion = "0.1.0"\n\n[dependencies]\nserde = "1"\n'
    )
    cargo_lock.write_text(
        'version = 4\n\n[[package]]\nname = "polars-mongo"\nversion = "0.1.0"\n'
        'dependencies = ["serde"]\n\n[[package]]\nname = "serde"\nversion = "1.0.0"\n'
    )
    uv_lock.write_text(
        'version = 1\n\n[[package]]\nname = "polars-mongo"\nsource = { editable = "." }\n'
        'dependencies = [{ name = "serde" }]\n\n[package.metadata]\nrequires-dist = []\n'
    )
    originals = (cargo_toml.read_text(), cargo_lock.read_text(), uv_lock.read_text())

    assert update_version_files("0.1.1", root=tmp_path)
    assert 'version = "0.1.1"' in cargo_toml.read_text()
    assert 'name = "polars-mongo"\nversion = "0.1.1"' in cargo_lock.read_text()
    assert (
        'name = "polars-mongo"\nsource = { editable = "." }\nversion =' not in uv_lock.read_text()
    )
    assert 'serde = "1"' in cargo_toml.read_text()
    assert 'dependencies = ["serde"]' in cargo_lock.read_text()
    assert '{ name = "serde" }' in uv_lock.read_text()
    assert cargo_toml.read_text() == originals[0].replace("0.1.0", "0.1.1")
    assert cargo_lock.read_text() == originals[1].replace("0.1.0", "0.1.1")
    assert uv_lock.read_text() == originals[2]


def test_higher_explicit_version_updates_stale_cargo_lock(tmp_path) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "polars-mongo"\nversion = "0.1.1"\n')
    (tmp_path / "Cargo.lock").write_text('[[package]]\nname = "polars-mongo"\nversion = "0.1.0"\n')
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "polars-mongo"\nsource = { editable = "." }\n'
    )

    assert main(["--root", str(tmp_path), "--latest", "v0.1.0"]) == 0
    assert 'version = "0.1.1"' in (tmp_path / "Cargo.lock").read_text()


def test_existing_uv_version_record_is_updated(tmp_path) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "polars-mongo"\nversion = "0.1.0"\n')
    (tmp_path / "Cargo.lock").write_text('[[package]]\nname = "polars-mongo"\nversion = "0.1.0"\n')
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "polars-mongo"\nsource = { editable = "." }\nversion = "0.1.0"\n'
    )

    assert update_version_files("0.1.1", root=tmp_path)
    assert 'version = "0.1.1"' in (tmp_path / "uv.lock").read_text()
