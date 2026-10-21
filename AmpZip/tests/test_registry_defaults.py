"""Registry: the search family, den-aware off by default, Python and engine constants agree."""
import os, re, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import sympy as sp
from ampzip.atoms import ab, sb
from ampzip.simplifier.runner import run_all, list_algorithms, is_gbfs, GBFS_FAMILY
from ampzip.simplifier.algorithms.gbfs import GBFS
from ampzip.simplifier import limits


def test_family():
    assert GBFS_FAMILY == ("model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f")
    assert all(is_gbfs(a) for a in GBFS_FAMILY) and all(a in list_algorithms() for a in GBFS_FAMILY)
    assert not is_gbfs("sft_greedy")
    e = sp.sympify("ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4)", locals={"ab": ab, "sb": sb})
    (r,) = run_all(["model_free_gbfs"], e, 4, gbfs_max_apps=50)
    assert r.algorithm == "model_free_gbfs" and r.solved and r.extra["den_aware"] is False


def test_den_aware_default_off():
    assert GBFS().den_aware is False and GBFS()._value_fn() is None
    assert GBFS(den_aware=True)._value_fn() is not None
    assert GBFS(ordering="raw")._value_fn() is not None
    try:
        GBFS(den_aware=True, ordering="raw")
    except ValueError:
        pass
    else:
        raise AssertionError("den-aware + raw must be refused")


def test_engine_default_matches_rust():
    src = open(os.path.join(_ROOT, "ampzip/engine/src/poly.rs")).read()
    m = re.search(r"unwrap_or\((\d[\d_]*)\)", src)
    assert m and int(m.group(1).replace("_", "")) == limits.ENGINE_DEFAULT_MAX_TERMS
    assert limits.DEFAULT_GBFS_MAX_TERMS == limits.ENGINE_DEFAULT_MAX_TERMS


def main():
    for fn in (test_family, test_den_aware_default_off, test_engine_default_matches_rust):
        fn(); print(f"  ok {fn.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
