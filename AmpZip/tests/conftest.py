"""pytest bootstrap for the Simplifier test suite.

It puts the package root, for `import ampzip.simplifier`, and this tests/ directory, for
`import test_atomize`, on sys.path. When the engine is not importable, as in a fresh clone
without the built `ampzip_engine`, it skips the whole suite through collection-ignore.
With the engine present the suite runs and fails on a regression.
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

# Without the engine every test_*.py, which imports it at module load, would raise a
# collection error, so they are all skipped.
collect_ignore_glob = [] if _ENGINE_OK else ["test_*.py"]
