"""`sample_timeout` bounds how long decoding is waited for: on expiry the result says so and the
decode runs on in the background. Skipped (exit 0) where the CDS backend is unavailable.

    python tests/test_cds_timeout.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the package root

import sympy as sp  # noqa: E402

from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.runner import run_all  # noqa: E402

EXPR = "ab(1,3)*ab(2,4) - ab(1,4)*ab(2,3)"


def _check(cond, msg):
    print(("  ok " if cond else "  FAIL ") + msg)
    if not cond:
        raise AssertionError(msg)


def main():
    expr = sp.sympify(EXPR, locals={"ab": ab, "sb": sb})
    # A normal call first, which loads the checkpoint. An unavailable backend skips the test.
    r = run_all(["cds_oneshot"], expr=expr, n_point=4, sample_timeout=300)[0]
    err = (r.extra or {}).get("error") or ""
    if err.startswith("backend unavailable") or "No module named" in err or "checkpoint" in err.lower():
        print(f"  skip: cds_oneshot unavailable here ({err[:80]})")
        return 0
    _check(not err, f"a normal decode has no error ({err[:120]})")
    _check(not r.timed_out and r.extra.get("stop_reason") != "time", "a normal decode is not timed out")

    # A sequential call with an expired budget reports at once while the decode still runs.
    t0 = time.time()
    r = run_all(["cds_oneshot"], expr=expr, n_point=4, sample_timeout=0.001)[0]
    waited = time.time() - t0
    _check(r.timed_out is True and r.extra.get("stop_reason") == "time",
           f"sequential: timed_out with stop_reason=time after {waited:.2f}s")
    _check(r.extra.get("decode_still_running") is True, "sequential: decode reported as still running")
    _check(r.solved is False and r.n_steps == 0, "sequential: not solved, no steps")
    _check(r.final_expr_str == "" and r.bracket_count_min == -1, "sequential: no expression reported")
    # A timed-out decode is not judged solved when the target is the input itself.
    r = run_all(["cds_oneshot"], expr=expr, n_point=4, target=expr, sample_timeout=0.001)[0]
    _check(r.timed_out is True and r.solved is False and "target_equal" not in r.extra,
           "timeout with target == input stays unsolved and unjudged")

    # in a parallel call the backend's own timeout fires before the collector placeholder
    out = run_all(["cds_oneshot", "model_free_gbfs"], expr=expr, n_point=4, sample_timeout=0.001,
                  max_parallel=2, gbfs_max_apps=2000, gbfs_max_seconds=30)
    cds = next(x for x in out if x.algorithm == "cds_oneshot")
    _check(cds.timed_out is True and cds.extra.get("stop_reason") == "time",
           "parallel: CDS reports its own timeout")
    _check(not any(x.extra.get("stop_reason") == "collector_timeout" for x in out),
           "parallel: no collector placeholder needed")
    # the stray decodes are non-daemon threads: the interpreter waits for them on exit
    print("ALL CDS_TIMEOUT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
