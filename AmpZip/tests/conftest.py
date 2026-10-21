"""pytest bootstrap for the Simplifier test suite.

Puts the Simplifier root (for `import ampzip.simplifier`, `ampzero`) and this tests/ dir (for
`import test_atomize`) on sys.path, and — if the Rust engine / RL backend is not importable
(e.g. a fresh clone without the built `ampzip_engine`) — cleanly SKIPS the whole suite by
collection-ignore rather than erroring out. On the dev box the engine is present, so the
suite runs and genuinely fails on a regression.
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)  # the package root
for _p in (_ROOT, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ENGINE_OK = True
try:
    import ampzip.simplifier._paths  # noqa: F401  (bootstraps the RL backend onto sys.path)
    import ampzip_engine  # noqa: F401
except Exception:  # noqa: BLE001
    _ENGINE_OK = False

# If the engine/backend is unavailable, every test_*.py (which imports it at module load)
# would raise a collection error — skip them all instead.
collect_ignore_glob = [] if _ENGINE_OK else ["test_*.py"]
