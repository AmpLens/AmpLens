"""The verifier's engine path gives the same verdicts, on the same sampled points, as the sympy
path it replaces, and falls back to that path where the engine cannot parse.

Run:  python tests/test_verify_engine.py   (or via tests/run_tests.py)
"""
from __future__ import annotations

import glob
import os
import pickle
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import sympy as sp  # noqa: E402

from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.simplifier.parse import parse_expr  # noqa: E402
from ampzip.verify import equal, zero  # noqa: E402
from ampzip.verify.verifier import _engine_asts  # noqa: E402

LOCAL = {"ab": ab, "sb": sb, "I": sp.I, "opaque_den": sp.Function("opaque_den")}


def _pairs(limit=60):
    out = []
    for f in sorted(glob.glob(os.path.join(_ROOT, "data", "az_v2", "v2_*pt_k*_c*.pkl"))):
        n = int(os.path.basename(f).split("_")[1][0])
        with open(f, "rb") as fh:
            for r in pickle.load(fh)["records"][:2]:
                out.append((n, r["scrambled"], r["simple"]))
    random.Random(0).shuffle(out)
    return out[:limit]


def test_engine_path_is_taken():
    import ampzip_engine as E
    assert _engine_asts("ab(1,2)*sb(2,3)", "ab(1,2)") is not None                 # text
    assert _engine_asts(E.Expr("ab(1,2)*sb(2,3)"), sp.Integer(0)) is not None     # handle + number
    assert _engine_asts(parse_expr("ab(1,2)*sb(2,3)", LOCAL), "ab(1,2)") is None  # sympy object -> sympy path
    assert _engine_asts("I*ab(1,2)", "ab(1,2)") is None                           # outside the grammar


def test_same_verdicts_and_same_points():
    """Equal pairs, unequal pairs and zero tests agree between the two paths, and both consume
    the random stream identically (same end state => same points were sampled)."""
    for n, a, b in _pairs():
        A, B = parse_expr(a, LOCAL), parse_expr(b, LOCAL)
        # negatives: adding a constant to a homogeneous rational function never preserves it
        for X, Y, want in ((a, b, True), (a, f"({b}) + 1", False), (a, f"({a}) + 1", False)):
            r1, r2 = random.Random(7), random.Random(7)
            v_engine = equal(X, Y, n, rng=r1)
            v_sympy = equal(parse_expr(X, LOCAL), parse_expr(Y, LOCAL), n, rng=r2, engine=False)
            assert v_engine == v_sympy == want, (n, a[:60], want, v_engine, v_sympy)
            assert r1.getstate() == r2.getstate(), "the two paths sampled different points"
        r1, r2 = random.Random(3), random.Random(3)
        assert zero(f"({a}) - ({b})", n, rng=r1) is True and zero(A - B, n, rng=r2, engine=False) is True
        assert r1.getstate() == r2.getstate()


def test_opaque_den_and_negative_powers():
    # at five points the sum S = <12>[23] + <14>[43] does not vanish on shell (it does at four,
    # where momentum conservation makes it an identity, so a pole in it is degenerate)
    s = "ab(1,2)**(-2)*sb(3,4)*opaque_den(ab(1,2)*sb(2,3) + ab(1,4)*sb(4,3))"
    t = "sb(3,4)/(ab(1,2)**2*(ab(1,2)*sb(2,3) + ab(1,4)*sb(4,3)))"
    assert equal(s, t, 5, rng=random.Random(1)) is True
    assert equal(s, f"2*({t})", 5, rng=random.Random(1)) is False
    A, B = parse_expr(s, LOCAL, lenient=True), parse_expr(t, LOCAL)
    assert equal(A, B, 5, rng=random.Random(1), engine=False) is True


def test_schouten_fixture():
    lhs = ab(1, 2) * ab(3, 4) - ab(1, 3) * ab(2, 4)
    rhs = -ab(1, 4) * ab(2, 3)
    assert equal(lhs, rhs, 4, rng=random.Random(0)) is True
    assert zero(lhs - rhs, 4, rng=random.Random(0)) is True
    assert equal(ab(1, 2), ab(1, 3), 4, rng=random.Random(0)) is False


def test_schouten_symbolic_sign():
    """Check the paper's sign deterministically using independent spinor components."""
    spinors = [sp.symbols(f"x{i} y{i}") for i in range(4)]

    def bracket(i, j):
        x, y = spinors[i]
        u, v = spinors[j]
        return x * v - y * u

    lhs = bracket(0, 1) * bracket(2, 3)
    rhs = bracket(0, 2) * bracket(1, 3) - bracket(0, 3) * bracket(1, 2)
    old_rhs = bracket(0, 2) * bracket(1, 3) - bracket(0, 3) * bracket(2, 1)
    assert sp.expand(lhs - rhs) == 0
    assert sp.expand(lhs - old_rhs) != 0


def test_sampling_exhaustion_has_no_verdict():
    """Both evaluation paths must leave a failed sampling attempt inconclusive."""
    for engine in (True, False):
        try:
            equal("ab(1,2)", "ab(1,2)", 4, k=1, max_resample=0,
                  sampler=lambda n, p, rng: None, engine=engine)
        except RuntimeError as exc:
            assert "singular/degenerate" in str(exc)
        else:
            raise AssertionError("sampling exhaustion returned an equality verdict")


def main() -> int:
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ok  ] {name}")
            except AssertionError as e:
                fails += 1
                print(f"[FAIL] {name}: {e}")
    print("ALL VERIFY-ENGINE TESTS PASSED" if not fails else f"{fails} verify-engine test(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
