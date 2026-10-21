"""The package version, from one source.

An installed distribution reports the version in its metadata (`pyproject.toml` at build time).
A source checkout that is run without installing reads the same field from the `pyproject.toml`
next to the package, so the two never disagree.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path


def _read() -> str:
    try:
        return _dist_version("ampzip")
    except PackageNotFoundError:
        pass
    try:
        import tomllib
        data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        return str(data["project"]["version"]) + "+source"
    except Exception:  # noqa: BLE001  (no pyproject next to the package)
        return "0+unknown"


__version__ = _read()
