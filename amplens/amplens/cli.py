"""List component versions with `amplens components`, `--version`, `-V` or no arguments.

Other invocations are passed to the `ampzip` command line.
"""
from __future__ import annotations

import sys

from . import COMPONENTS, __version__, components


def _print_components() -> int:
    print(f"amplens {__version__}")
    for dist, version in components().items():
        label = COMPONENTS[dist][0]
        print(f"  {dist:<16} {version if version is not None else 'not installed':<16} {label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["components"]:
        return _print_components()
    if args in ([], ["--version"], ["-V"]):
        return _print_components()
    try:
        from ampzip.simplifier.cli import main as ampzip_main
    except ImportError:
        print("amplens: the ampzip component is not installed (pip install ampzip)", file=sys.stderr)
        return 2
    return int(ampzip_main(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
