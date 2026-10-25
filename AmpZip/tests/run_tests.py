#!/usr/bin/env python
"""Standalone test runner.

    python tests/run_tests.py        # exits non-zero if any suite fails

Runs the check suites and a smoke test of the GBFS backends, then reports a
pass/fail summary. Suite exceptions and nonzero return values are reported as
failures. The script requires the dependencies used by those suites, including
the web test dependencies, but does not require pytest. `pytest tests/` is a
separate test entry point.
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)  # the package root
for _p in (_ROOT, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ampzip.simplifier._paths  # noqa: F401,E402  (bootstraps the RL backend)


def _run(label, fn):
    try:
        ok = (fn() == 0)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {label}: {e!r}")
        return False
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}")
    return ok


def _gbfs_smoke():
    import sympy as sp
    from ampzip.atoms import ab, sb
    from ampzip.simplifier.runner import run_all
    expr = sp.sympify("ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))", locals={"ab": ab, "sb": sb})
    fails = []
    for algo in ("model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f"):
        (r,) = run_all([algo], expr, 4, gbfs_max_apps=2000, gbfs_max_seconds=15)
        err = (r.extra or {}).get("error")
        if err is not None:
            fails.append(f"{algo}: {err}")
        elif not (0 <= r.bracket_count_min <= r.bracket_count_start):
            fails.append(f"{algo}: min {r.bracket_count_min} > start {r.bracket_count_start}")
    for f in fails:
        print(f"       gbfs smoke FAIL: {f}")
    return 1 if fails else 0


def main():
    try:
        import ampzip_engine  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[skip] Rust engine unavailable ({e!r}); build ampzip_engine to run the suite.")
        return 0

    import test_atomize
    import test_reconstruct
    import test_gbfs_f
    import test_spurious
    import test_parse
    import test_atoms
    import test_verify_engine
    import test_gbfs_checkpoint
    import test_gbfs_packing
    import test_gbfs_stop_reasons
    import test_gbfs_e_budget
    import test_registry_defaults
    import test_validate_engine_coeff
    import test_post_timeout
    import test_webservice_clamps
    import test_display_count
    import test_target_judge
    import test_cds_timeout
    import test_engine_limits

    results = [
        _run("atomize", test_atomize.main),
        _run("reconstruct", test_reconstruct.main),
        _run("gbfs_f", test_gbfs_f.main),
        _run("spurious", test_spurious.main),
        _run("parse", test_parse.main),
        _run("atoms", test_atoms.main),
        _run("verify_engine", test_verify_engine.main),
        _run("gbfs_checkpoint", test_gbfs_checkpoint.main),
        _run("gbfs_packing", test_gbfs_packing.main),
        _run("gbfs_stop_reasons", test_gbfs_stop_reasons.main),
        _run("gbfs_e_budget", test_gbfs_e_budget.main),
        _run("registry_defaults", test_registry_defaults.main),
        _run("validate_engine_coeff", test_validate_engine_coeff.main),
        _run("post_timeout", test_post_timeout.main),
        _run("display_count", test_display_count.main),
        _run("target_judge", test_target_judge.main),
        _run("cds_timeout", test_cds_timeout.main),
        _run("webservice_clamps", test_webservice_clamps.main),
        _run("engine_limits", test_engine_limits.main),
        _run("gbfs_family_smoke", _gbfs_smoke),
    ]
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} suites passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
