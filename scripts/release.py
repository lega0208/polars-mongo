"""Select and apply the package version used by automated releases.

The Cargo package is the source of truth for the Python package's dynamic
version.  This module deliberately edits the three lock/metadata files as
text: updating a version must not resolve or otherwise change dependencies.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Final

_VERSION_RE: Final = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_VERSION_LINE_RE: Final = re.compile(
    r'^(?P<prefix>\s*version\s*=\s*")(?P<version>[^"]+)(?P<suffix>"\s*(?:#.*)?\r?\n?)$'
)


class ReleaseVersionError(ValueError):
    """Raised when release version metadata is invalid or inconsistent."""


def parse_version(value: str) -> tuple[int, int, int]:
    """Parse a strict ``MAJOR.MINOR.PATCH`` version."""

    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ReleaseVersionError(f"invalid version: {value!r}")
    major, minor, patch = (int(part) for part in value.split("."))
    return major, minor, patch


def parse_release_tag(tag: str) -> tuple[int, int, int]:
    """Parse a strict ``vMAJOR.MINOR.PATCH`` release tag."""

    if not isinstance(tag, str) or not tag.startswith("v"):
        raise ReleaseVersionError(f"invalid release tag: {tag!r}")
    return parse_version(tag[1:])


def format_version(version: tuple[int, int, int]) -> str:
    """Format a parsed version tuple."""

    return ".".join(str(part) for part in version)


def select_release_version(current: str, latest: str | None) -> str:
    """Choose the release version for *current* metadata.

    A missing latest release leaves the explicit current version unchanged.  A
    current version equal to the latest release receives a patch bump.  An
    explicit higher version is accepted, while a lower version is rejected.
    """

    current_parts = parse_version(current)
    if latest is None or latest == "":
        return current
    if not isinstance(latest, str):
        raise ReleaseVersionError(f"invalid release tag: {latest!r}")
    latest_parts = parse_release_tag(latest) if latest.startswith("v") else parse_version(latest)
    if current_parts < latest_parts:
        raise ReleaseVersionError(
            f"Cargo.toml version {current!r} is lower than latest release {latest!r}"
        )
    if current_parts == latest_parts:
        return format_version((current_parts[0], current_parts[1], current_parts[2] + 1))
    return current


def _replace_version_line(line: str, version: str) -> str | None:
    match = _VERSION_LINE_RE.fullmatch(line)
    if match is None:
        return None
    newline = "\n" if line.endswith("\n") else ""
    return (
        f"{match.group('prefix')}{version}"
        f"{match.group('suffix').rstrip(chr(10) + chr(13))}{newline}"
    )


def _replace_package_version(text: str, package_name: str, version: str, *, table: str) -> str:
    """Replace the version in one named TOML package record."""

    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == table]
    for start_index in starts:
        end_index = next(
            (i for i in range(start_index + 1, len(lines)) if lines[i].strip() == table),
            len(lines),
        )
        name_index = next(
            (
                i
                for i in range(start_index + 1, end_index)
                if re.fullmatch(r'\s*name\s*=\s*"[^"]+"\s*\r?\n?', lines[i])
            ),
            None,
        )
        if name_index is None or lines[name_index].split('"', 2)[1] != package_name:
            continue
        for i in range(name_index + 1, end_index):
            replacement = _replace_version_line(lines[i], version)
            if replacement is not None:
                return "".join(lines[:i] + [replacement] + lines[i + 1 :])
        raise ReleaseVersionError(f"{table} record {package_name!r} has no version field")
    raise ReleaseVersionError(f"{table} record {package_name!r} not found")


def _replace_cargo_version(text: str, version: str) -> str:
    lines = text.splitlines(keepends=True)
    package_start = next((i for i, line in enumerate(lines) if line.strip() == "[package]"), None)
    if package_start is None:
        raise ReleaseVersionError("Cargo.toml has no [package] table")
    package_end = next(
        (i for i in range(package_start + 1, len(lines)) if lines[i].startswith("[")),
        len(lines),
    )
    for i in range(package_start + 1, package_end):
        replacement = _replace_version_line(lines[i], version)
        if replacement is not None:
            return "".join(lines[:i] + [replacement] + lines[i + 1 :])
    raise ReleaseVersionError("Cargo.toml [package] table has no version field")


def _replace_uv_version(text: str, version: str) -> str:
    """Update the editable project's root package record in uv.lock.

    uv intentionally omits ``version`` for an editable dynamic project.  Keep
    that omission; update a version only when a lock record already has one.
    """

    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == "[[package]]"]
    for start_index in starts:
        end_index = next(
            (i for i in range(start_index + 1, len(lines)) if lines[i].strip() == "[[package]]"),
            len(lines),
        )
        name_index = next(
            (
                i
                for i in range(start_index + 1, end_index)
                if re.fullmatch(r'\s*name\s*=\s*"[^"]+"\s*\r?\n?', lines[i])
            ),
            None,
        )
        if name_index is None or lines[name_index].split('"', 2)[1] != "polars-mongo":
            continue
        # Only inspect package-record fields, not nested package metadata tables.
        record_end = next(
            (i for i in range(name_index + 1, end_index) if lines[i].startswith("[")),
            end_index,
        )
        for i in range(name_index + 1, record_end):
            replacement = _replace_version_line(lines[i], version)
            if replacement is not None:
                return "".join(lines[:i] + [replacement] + lines[i + 1 :])
        return text
    raise ReleaseVersionError("uv.lock package record 'polars-mongo' not found")


def update_version_files(
    version: str,
    *,
    root: Path | str = ".",
) -> bool:
    """Synchronize package records and return whether files changed.

    The editable project entry in ``uv.lock`` commonly has no version because
    its version is dynamic; that intentional omission is preserved.
    """

    parse_version(version)
    root_path = Path(root)
    cargo_toml_path = root_path / "Cargo.toml"
    cargo_lock_path = root_path / "Cargo.lock"
    uv_lock_path = root_path / "uv.lock"

    cargo_toml_text = cargo_toml_path.read_text()
    cargo_lock_text = cargo_lock_path.read_text()
    uv_lock_text = uv_lock_path.read_text()
    updated = (
        _replace_cargo_version(cargo_toml_text, version),
        _replace_package_version(cargo_lock_text, "polars-mongo", version, table="[[package]]"),
        _replace_uv_version(uv_lock_text, version),
    )
    changed = updated != (cargo_toml_text, cargo_lock_text, uv_lock_text)
    if changed:
        cargo_toml_path.write_text(updated[0])
        cargo_lock_path.write_text(updated[1])
        uv_lock_path.write_text(updated[2])
    return changed


def read_cargo_version(path: Path | str = "Cargo.toml") -> str:
    try:
        package = tomllib.loads(Path(path).read_text()).get("package")
    except tomllib.TOMLDecodeError as exc:
        raise ReleaseVersionError(f"invalid Cargo.toml: {exc}") from exc
    if not isinstance(package, dict):
        raise ReleaseVersionError("Cargo.toml has no [package] table")
    version = package.get("version")
    if not isinstance(version, str):
        raise ReleaseVersionError("Cargo.toml [package] table has no version field")
    parse_version(version)
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", default=None, help="latest published vMAJOR.MINOR.PATCH tag")
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    args = parser.parse_args(argv)
    try:
        current = read_cargo_version(args.root / "Cargo.toml")
        selected = select_release_version(current, args.latest)
        # A first release has no prior version to synchronize.  For any
        # comparison against a published release, synchronize the lock records
        # even when an explicit higher Cargo version is retained.
        changed = bool(args.latest) and update_version_files(selected, root=args.root)
    except (OSError, ReleaseVersionError) as exc:
        print(f"release version error: {exc}", file=sys.stderr)
        return 1
    print(selected)
    if changed:
        print("version metadata updated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
