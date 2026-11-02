#!/usr/bin/env python
"""Standalone test runner.

    python tests/run_tests.py        # exits non-zero if any suite fails

Runs the check suites and a smoke test of the GBFS backends, then reports a
pass/fail summary. Suite exceptions and nonzero return values are reported as
failures. The script requires the dependencies used by those suites, including
the web test dependencies, but does not require pytest. `pytest tests/` is a
separate test entry point. The engine must be importable; without it the run
fails instead of passing with nothing checked.
"""
import importlib
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)  # the package root
for _p in (_ROOT, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ampzip.simplifier._paths  # noqa: F401,E402  (bootstraps the RL backend)

# Each entry is a tests/test_<name>.py module with a main() returning 0 on success.
SUITES = [
    "atomize", "reconstruct", "gbfs_f", "spurious", "parse", "atoms", "verify_engine",
    "gbfs_checkpoint", "gbfs_packing", "gbfs_stop_reasons", "gbfs_e_budget",
    "registry_defaults", "validate_engine_coeff", "post_timeout", "display_count",
    "target_judge", "cds_timeout", "webservice_clamps", "engine_limits", "input_limits",
    "input_property", "numerical_check", "gbfs_invariants",
]

# Test files that are deliberately not run here, with the reason.
NOT_REGISTERED = {
    "backends_smoke": "pytest form of the inline gbfs smoke below",
    "gate": "pytest wrapper around atomize, reconstruct and gbfs_f",
    "gbfs_resume": "needs data/az_v2 and runs 200k moves",
    "cds_completed_timing": "paper script, not exported",
    "cds_end_to_end": "paper script, not exported",
    "comparison_protocol": "paper script, not exported",
    "default_continuation": "paper script, not exported",
    "paper_data_sources": "paper script, not exported",
    "paper_number_checks": "paper script, not exported",
    "ym5pt_sign_release": "research script, not exported",
    "ym5pt_uniform_budget": "research script, not exported",
}


def _run(label, fn):
    try:
        ok = (fn() == 0)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {label}: {e!r}")
        return False
    print(f"[{'ok  ' if ok else 'FAIL'}] {label}")
    return ok


def _registration():
    """Every tests/test_*.py present is either a registered suite or listed with a reason."""
    present = sorted(f[5:-3] for f in os.listdir(_TESTS) if f.startswith("test_") and f.endswith(".py"))
    missing = [n for n in present if n not in SUITES and n not in NOT_REGISTERED]
    absent = [n for n in SUITES if n not in present]
    for n in missing:
        print(f"       tests/test_{n}.py is not registered in run_tests.py")
    for n in absent:
        print(f"       registered suite test_{n}.py is missing")
    return 1 if missing or absent else 0


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
        print(f"[FAIL] the engine is unavailable, {e!r}. Build ampzip_engine to run the suite.")
        return 1

    results = [_run("registration", _registration)]
    for name in SUITES:
        module = importlib.import_module(f"test_{name}")
        results.append(_run(name, module.main))
    results.append(_run("gbfs_family_smoke", _gbfs_smoke))
    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} suites passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
