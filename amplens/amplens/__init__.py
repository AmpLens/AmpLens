"""AmpLens meta package: installs the components and reports which are present.

The package itself has no algorithms. `components()` lists the installed component distributions with their versions; the `amplens` command forwards to the `ampzip` command line.
"""
from __future__ import annotations

from importlib import metadata as _metadata

__version__ = "0.1.0"

# PyPI distribution name -> (component, import name). Extended as components are added.
COMPONENTS = {
    "ampzip": ("AmpZip", "ampzip"),
    "amplens-engine": ("AmpZip engine", "ampzip_engine"),
}


def components() -> dict[str, str | None]:
    """Installed version of each component distribution, or None when it is not installed."""
    out: dict[str, str | None] = {}
    for dist in COMPONENTS:
        try:
            out[dist] = _metadata.version(dist)
        except _metadata.PackageNotFoundError:
            out[dist] = None
    return out


__all__ = ["COMPONENTS", "components", "__version__"]
