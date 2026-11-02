"""The search family on fixed samples: the output equals the input on the finite-field test,
the counts never rise above the start, every recorded step passes the audit, and the result
at a fixed move budget matches the stored golden.

The samples in tests/fixtures/gbfs_invariants.json are 20 scrambled expressions per number
of legs, drawn with a fixed seed from the size-depth suite, with the goldens recorded by the
same call this test makes. A golden changes only when the search or its post-processing
changes on purpose; regenerate the file with the same budget and single thread.

Run:  python tests/test_gbfs_invariants.py   (or via tests/run_tests.py)
"""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import sympy as sp  # noqa: E402

import ampzip.simplifier._paths  # noqa: E402,F401
from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.runner import run_all  # noqa: E402
from ampzip.verify import equal  # noqa: E402

FIXTURE = os.path.join(_HERE, "fixtures", "gbfs_invariants.json")


def _check(record, result, golden):
    n, raw = record["n"], record["scrambled"]
    problems = []
    err = (result.extra or {}).get("error")
    if err is not None:
        return [f"error: {err}"]
    if not equal(result.final_expr_str, raw, n):
        problems.append("final expression differs from the input on the finite-field test")
    if not (0 <= result.bracket_count_min <= result.bracket_count_start):
        problems.append(f"min {result.bracket_count_min} outside [0, start {result.bracket_count_start}]")
    if not (0 <= result.bracket_count_final <= result.bracket_count_start):
        problems.append(f"final {result.bracket_count_final} above start {result.bracket_count_start}")
    audit = (result.extra or {}).get("audit") or {}
    if audit.get("step_bad"):
        problems.append(f"audit rejected steps {audit['step_bad']}")
    if result.derivation is not None and len(result.derivation) != result.n_steps:
        problems.append(f"{len(result.derivation)} derivation rows for {result.n_steps} steps")
    for key in ("final_expr_str", "bracket_count_start", "bracket_count_final", "bracket_count_min", "n_steps"):
        if getattr(result, key) != golden[key]:
            problems.append(f"{key}: {getattr(result, key)!r} differs from the golden {golden[key]!r}")
    return problems


def main() -> int:
    with open(FIXTURE) as f:
        fixture = json.load(f)
    algos, max_apps = fixture["algorithms"], fixture["max_apps"]
    fails, t0 = 0, time.time()
    for record in fixture["records"]:
        expr = sp.sympify(record["scrambled"], locals={"ab": ab, "sb": sb})
        results = run_all(algos, expr, record["n"], capture_derivation=True, gbfs_max_apps=max_apps,
                          gbfs_threads=1, gbfs_audit_steps=True, raw_input=record["scrambled"], max_parallel=1)
        for result in results:
            problems = _check(record, result, record["golden"][result.algorithm])
            if problems:
                fails += 1
                print(f"  FAIL n={record['n']} index={record['source_index']} {result.algorithm}: " + "; ".join(problems))
    print(f"  {len(fixture['records'])} samples x {len(algos)} algorithms in {time.time() - t0:.1f} s, {fails} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
