"""Checkpoint RSS is optional and uses GiB on Linux and macOS.

    PYTHONPATH=. python tests/test_gbfs_checkpoint.py
"""
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ampzip
from ampzip.search import gbfs_checkpoint as C
from ampzip.search.gbfs import simplify


def test_without_resource():
    # A fresh interpreter catches an unconditional import even if another suite
    # has already imported resource or the search modules in this process.
    code = textwrap.dedent('''
        import sys
        sys.path.insert(0, sys.argv[1])
        sys.modules["resource"] = None

        import tempfile
        from pathlib import Path
        import sympy as sp
        from ampzip.atoms import ab, sb
        from ampzip.search import gbfs_checkpoint as C
        from ampzip.search.gbfs import simplify
        from ampzip.simplifier.runner import run_all

        assert C.peak_rss_gb() is None
        expr = sp.sympify("ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))",
                          locals={"ab": ab, "sb": sb})
        for algo in ("model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f"):
            (result,) = run_all([algo], expr, 4,
                                gbfs_max_apps=32, gbfs_max_seconds=5)
            assert result.extra.get("error") is None, (algo, result.extra)
            assert 0 <= result.bracket_count_min <= result.bracket_count_start

        start = "ab(1,2)*ab(3,4) - ab(1,3)*ab(2,4)"
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "search.ckpt")
            result = simplify(start, 4, max_apps=32, max_seconds=5,
                              checkpoint_path=path)
            assert result["checkpoint_error"] is None, result["checkpoint_error"]
            header, *_ = C.load(path)
            assert header["best_expr"] == result["best_expr"]
            assert header["segments"][-1]["peak_rss_gb"] is None

            resumed = simplify(start, 4, max_apps=64, max_seconds=5,
                               resume_from=path, checkpoint_path=path)
            assert resumed["checkpoint_error"] is None, resumed["checkpoint_error"]
            header, *_ = C.load(path)
            assert header["best_expr"] == resumed["best_expr"]
            assert header["apps"] == resumed["apps"]
            assert len(header["segments"]) == 2
            assert all(s["peak_rss_gb"] is None for s in header["segments"])
    ''')
    # Follow the package selected by the caller, including an installed wheel.
    package_root = str(Path(ampzip.__file__).resolve().parent.parent)
    result = subprocess.run([sys.executable, "-c", code, package_root],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_peak_rss_units():
    for system, raw in (("Linux", 1 << 20), ("Darwin", 1 << 30)):
        resource = SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=Mock(return_value=SimpleNamespace(ru_maxrss=raw)),
        )
        with patch.dict(sys.modules, {"resource": resource}), \
                patch.object(C.platform, "system", return_value=system):
            assert C.peak_rss_gb() == 1.0, system
        resource.getrusage.assert_called_once_with(resource.RUSAGE_SELF)


def test_rss_failure_does_not_block_checkpoint():
    resource = SimpleNamespace(RUSAGE_SELF=0, getrusage=Mock(side_effect=OSError("unavailable")))
    with patch.dict(sys.modules, {"resource": resource}), tempfile.TemporaryDirectory() as tmp:
        assert C.peak_rss_gb() is None
        path = str(Path(tmp) / "search.ckpt")
        result = simplify("ab(1,2)*ab(3,4)", 4, max_apps=32, max_seconds=5,
                          checkpoint_path=path)
        assert result["checkpoint_error"] is None, result["checkpoint_error"]
        header, *_ = C.load(path)
        assert header["segments"][-1]["peak_rss_gb"] is None
        assert header["best_expr"] == result["best_expr"]


def main():
    for fn in (test_without_resource, test_peak_rss_units,
               test_rss_failure_does_not_block_checkpoint):
        fn()
        print(f"  ok {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
