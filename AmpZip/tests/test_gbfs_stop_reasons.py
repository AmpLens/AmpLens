"""Every way a search ends is named, every rejected move is counted, timed_out == time."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import threading
import sympy as sp
import ampzip_engine as E
from ampzip.search.gbfs import simplify
from ampzip.atoms import ab, sb
from ampzip.simplifier.runner import run_all

L = {"ab": ab, "sb": sb}
X = "ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4) + ab(2,3)*ab(1,4)/ab(2,4)"
MONO = "ab(1,2)*sb(3,4)"
TWO = "ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4)"   # reaches -ab(2,3): raw count 1


def test_reasons():
    assert simplify(X, 4, max_apps=50)["stop_reason"] == "budget"
    assert simplify(X, 4, max_apps=10**6, max_seconds=0.0)["stop_reason"] == "time"
    assert simplify(TWO, 4, max_apps=10**6, goal_raw_brackets=1)["stop_reason"] == "goal"
    assert simplify(X, 4, max_apps=10**6, patience_apps=20)["stop_reason"] == "patience"
    ev = threading.Event(); ev.set()
    assert simplify(X, 4, max_apps=10**6, should_stop=ev.is_set)["stop_reason"] == "cancelled"
    r = simplify(MONO, 4, max_apps=10**6)
    assert r["stop_reason"] in ("frontier_empty", "budget"), r["stop_reason"]


def test_skip_counters_and_thread_independence():
    # A tiny per-move term cap makes the engine reject expanding rewrites; the counters must
    # be identical across thread counts, as must every other result field.
    # The cap also bounds the input expansion (X has 3 terms), so 3 is the tightest cap that
    # still parses the root; any rewrite that grows past 3 terms is then skipped and counted.
    r1 = simplify(X, 4, max_apps=400, max_terms=3, threads=1)
    r4 = simplify(X, 4, max_apps=400, max_terms=3, threads=4)
    assert r1["term_limit_skips"] > 0
    keys = [k for k in r1 if k != "best"]
    assert all(r1[k] == r4[k] for k in keys), [k for k in keys if r1[k] != r4[k]]


def test_backend_timed_out_matches_reason():
    e = sp.sympify(X, locals=L)
    for algo in ("model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f"):
        (r,) = run_all([algo], e, 4, gbfs_max_apps=10**7, gbfs_max_seconds=0.001, raw_input=X)
        assert r.extra.get("error") is None, (algo, r.extra.get("error"))
        assert r.timed_out == (r.extra["stop_reason"] == "time"), (algo, r.extra["stop_reason"], r.timed_out)
        (r,) = run_all([algo], e, 4, gbfs_max_apps=30, gbfs_max_seconds=30, raw_input=X)
        assert r.extra["stop_reason"] == "budget", (algo, r.extra["stop_reason"])
        assert r.extra["apps"] <= 30, (algo, r.extra["apps"])
        assert not r.timed_out


def test_target_criteria():
    e = sp.sympify(X, locals=L)
    tgt = sp.sympify("ab(1,2)*ab(3,4)*ab(2,4) - ab(1,3)*ab(2,4)**2 + ab(2,3)*ab(1,4)**2", locals=L) / sp.sympify("ab(1,4)*ab(2,4)", locals=L)
    (r,) = run_all(["model_free_gbfs"], e, 4, target=tgt, gbfs_max_apps=2000, gbfs_target_criterion="brackets")
    assert r.extra["stop_reason"] == "goal" and r.extra["target_equal"] is True and r.solved
    (r,) = run_all(["model_free_gbfs"], e, 4, target=tgt, gbfs_max_apps=2000, gbfs_target_criterion="none")
    assert r.extra["stop_reason"] == "budget" and r.solved      # target-free run, post-hoc equal
    (r,) = run_all(["model_free_gbfs"], e, 4, target=sp.sympify("ab(2,3)", locals=L), gbfs_max_apps=2000)
    assert r.solved is False and r.extra["target_equal"] is False


def main():
    for fn in (test_reasons, test_skip_counters_and_thread_independence,
               test_backend_timed_out_matches_reason, test_target_criteria):
        fn()
        print(f"  ok {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
