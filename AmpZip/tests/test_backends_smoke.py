"""Engine-only smoke tests for the runner backends.

The gbfs family needs only the engine, so it is the fast core of the suite. Each variant runs
on a 4-point Parke-Taylor input without error and tracks a non-inflating minimum. SFT and
CDS need checkpoints and are slow, so they run only with SIMP_TEST_NEURAL=1 and skip when the
weights are absent.
"""
import os
import pytest
import sympy as sp

import ampzip.simplifier._paths  # noqa: F401
from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.runner import run_all  # noqa: E402

LOCAL = {"ab": ab, "sb": sb}
# The 4-point Parke-Taylor MHV amplitude, which the engine alone simplifies.
EXPR = sp.sympify("ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))", locals=LOCAL)

GBFS = ["model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f"]


@pytest.mark.parametrize("algo", GBFS)
def test_gbfs_family_smoke(algo):
    (r,) = run_all([algo], EXPR, 4, gbfs_max_apps=2000, gbfs_max_seconds=15)
    assert (r.extra or {}).get("error") is None, r.extra
    assert r.bracket_count_start >= 0
    assert 0 <= r.bracket_count_min <= r.bracket_count_start  # non-inflating minimum


@pytest.mark.skipif(not os.environ.get("SIMP_TEST_NEURAL"),
                    reason="the neural baselines are slow and need weights, set SIMP_TEST_NEURAL=1 to run them")
@pytest.mark.parametrize("algo", ["sft_greedy", "cds_oneshot"])
def test_neural_smoke(algo):
    (r,) = run_all([algo], EXPR, 4)
    err = ((r.extra or {}).get("error") or "").lower()
    if any(k in err for k in ("load", "not found", "no such file", "checkpoint")):
        pytest.skip(f"{algo}: weights absent ({err[:60]})")
    assert (r.extra or {}).get("error") is None, r.extra
    assert r.bracket_count_min is not None


@pytest.mark.parametrize("algo", GBFS)
def test_gbfs_budget_not_capped_by_sample_timeout(algo):
    """`sample_timeout` applies to SFT and CDS. GBFS uses `gbfs_max_seconds` for its
    time limit. A small move budget keeps this test short."""
    (r,) = run_all([algo], EXPR, 4, gbfs_max_apps=50, gbfs_max_seconds=600, sample_timeout=5)
    assert (r.extra or {}).get("error") is None, r.extra
    assert r.extra.get("max_seconds") == 600
