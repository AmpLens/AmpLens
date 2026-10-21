"""gbfs_e / gbfs_f: the variant-level move budget is a true ceiling over all pieces; the
split cap is a parameter and is reported."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import sympy as sp
from ampzip.atoms import ab, sb
from ampzip.simplifier.runner import run_all

X = "ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4) + ab(2,3)*ab(1,4)/ab(2,4)"


def test_apps_ceiling():
    e = sp.sympify(X, locals={"ab": ab, "sb": sb})
    for algo in ("gbfs_e", "gbfs_f"):
        (r,) = run_all([algo], e, 4, gbfs_max_apps=100, gbfs_max_seconds=30, raw_input=X)
        per = r.extra["per_term"]
        assert len(per) == 3 and sum(p.get("apps", 0) for p in per) <= 100, per
        assert r.extra["apps"] <= 100 and r.extra["stop_reason"] == "budget"


def test_split_cap_parameter():
    e = sp.sympify(X, locals={"ab": ab, "sb": sb})
    (r,) = run_all(["gbfs_e"], e, 4, gbfs_max_apps=100, raw_input=X, gbfs_split_max_terms=2)
    assert r.extra["raw_split_skipped"] == "cap" and r.extra["split_capped"] is True
    assert r.extra["split_max_terms"] == 2
    (r,) = run_all(["gbfs_e"], e, 4, gbfs_max_apps=100, raw_input=X, gbfs_split_max_terms=0)
    assert r.extra["n_terms"] == 3 and r.extra["split_max_terms"] == 0


def main():
    test_apps_ceiling(); print("  ok test_apps_ceiling")
    test_split_cap_parameter(); print("  ok test_split_cap_parameter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
