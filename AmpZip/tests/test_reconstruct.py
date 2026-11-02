"""Verifier-gated tests for the Reconstructor (simplifier/reconstruct.py).

Standalone runnable (no pytest):  python tests/test_reconstruct.py

Two properties:
  1. round-trip value preservation: equal(atomize(reconstruct(x)), x)   (exact regroup)
  2. priority s > ab/sb > Dirac is applied as specified.
"""
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ampzip.simplifier._paths  # noqa: F401,E402
import sympy as sp  # noqa: E402
from ampzip.atoms import ab, sb  # noqa: E402
from ampzip.verify import equal  # noqa: E402
from ampzip.simplifier.atomize import atomize  # noqa: E402
from ampzip.simplifier.reconstruct import reconstruct  # noqa: E402

_LOC = {"ab": ab, "sb": sb}
_fails = []


def _roundtrip(expr, n):
    s = str(expr)
    rec = reconstruct(s)
    back = sp.sympify(atomize(rec, n), locals=_LOC)
    ok = equal(sp.sympify(s, locals=_LOC), back, n, rng=random.Random(0)) is True
    print(f"  [{'ok ' if ok else 'FAIL'}] {s:46} -> {rec}")
    if not ok:
        _fails.append(s)
    return rec


def _expect(expr, n, must_contain):
    rec = _roundtrip(expr, n)
    if must_contain not in rec:
        print(f"       [FAIL] expected {must_contain!r} in {rec!r}")
        _fails.append(str(expr))


def main():
    print("Priority s > ab/sb > Dirac:")
    # bare canonical ab(i,j)*sb(i,j) = <ij>[ij] = -s(i,j) under textbook s=<ij>[ji]
    _expect(ab(1, 2) * sb(1, 2), 4, "-s(1, 2)")                 # pair -> -s
    _expect(ab(1, 2) * sb(2, 3), 4, "chain('as', 1, [2], 3)")   # a shared index folds to a chain
    # For a single-index chain reconstruct canonicalises the spelling to 'as' with reversed
    # legs, since chain('as',3,[2],1) and chain('sa',1,[2],3) are the same object.
    _expect(sb(1, 2) * ab(2, 3), 4, "chain('as', 3, [2], 1)")
    # s(1,2) folds, and the shared-index leftover ab(2,3)*sb(2,4) also folds to a chain. Under
    # the textbook s=<ij>[ji] the leading '-' is absorbed into the chain sign, and the whole
    # rendering is verified equal to the input.
    _expect(ab(1, 2) * sb(1, 2) * ab(2, 3) * sb(2, 4), 4, "s(1, 2)*chain('as', 3, [2], 4)")

    print("Multi-particle s / single-channel chain / opaque_den fold:")
    # a whole Add that IS a single multi-particle s or single-channel chain folds whole
    _expect(sp.sympify(atomize("s(1,2,3)"), locals=_LOC), 6, "s(1, 2, 3)")
    _expect(sp.sympify(atomize("chain('as',2,[6,1],5)"), locals=_LOC), 6,
            "chain('as', 2, [1, 6], 5)")
    # multi-particle s / single-channel chain in factor position (power base, opaque den)
    _expect(sp.sympify(atomize("chain('as',6,[1,2],3)**3"), locals=_LOC), 6,
            "chain('as', 6, [1, 2], 3)**3")
    _expect(sp.sympify(atomize("ab(1,2)/s(1,2,3)"), locals=_LOC), 6, "s(1, 2, 3)")
    _expect(sp.sympify(atomize("ab(1,2)/chain('as',2,[6,1],5)"), locals=_LOC), 6,
            "chain('as', 2, [1, 6], 5)")
    # a bare top-level sum is rendered term-wise
    _expect(ab(1, 2) * sb(1, 2) + ab(1, 3) * sb(1, 3), 6, "-s(1, 2) - s(1, 3)")
    # two reference six-point inputs: fully s()/chain() readable
    for ref in [
        "(chain('as',6,[1,2],3)**3)/(ab(6,1)*ab(1,2)*sb(3,4)*sb(4,5)*s(6,1,2)"
        "*chain('as',2,[6,1],5))+(chain('as',4,[5,6],1)**3)/(ab(2,3)*ab(3,4)*sb(5,6)"
        "*sb(6,1)*s(5,6,1)*chain('as',2,[6,1],5))",
        "(chain('as',6,[1,2],3)**2*ab(4,5)*sb(1,2))/(s(6,1)*s(1,2)*s(3,4)*s(4,5)"
        "*s(6,1,2)) + (chain('as',4,[2,3],1)**2*ab(5,6)*sb(2,3))/(s(2,3)*s(3,4)*s(5,6)"
        "*s(6,1)*s(5,6,1))",
    ]:
        rec = _roundtrip(sp.sympify(atomize(ref), locals=_LOC), 6)
        for bad in ("ab(1, 6)*sb(1, 3)", "opaque_den"):  # raw chain / opaque leak-through
            if bad in rec:
                print(f"       [FAIL] unfolded {bad!r} leaked into {rec!r}")
                _fails.append(zl)

    # A sum of fractions over a common pole A renders as a leading `-1/A*(...)` only with
    # prefactor=True, which gbfs_f sets. The default of every other backend keeps the two-story
    # `(...)/A`, which is what the test below checks. The split form gbfs_f recombines is built
    # with each raw term atomized separately so it keeps its own denominator.
    print("Prefactor opt-in, the gbfs_f display:")
    import ast as _ast  # noqa: E402
    _raw = ("(chain('as',6,[1,2],3)**3)/(ab(6,1)*ab(1,2)*sb(3,4)*sb(4,5)*s(6,1,2)"
            "*chain('as',2,[6,1],5))+(chain('as',4,[5,6],1)**3)/(ab(2,3)*ab(3,4)*sb(5,6)"
            "*sb(6,1)*s(5,6,1)*chain('as',2,[6,1],5))")
    _pf_loc = {"ab": ab, "sb": sb, "opaque_den": sp.Function("opaque_den")}
    _node = _ast.parse(_raw, mode="eval").body

    def _flat(nd):
        if isinstance(nd, _ast.BinOp) and isinstance(nd.op, (_ast.Add, _ast.Sub)):
            return _flat(nd.left) + [(type(nd.op).__name__, nd.right)]
        return [("Add", nd)]

    _pieces = []
    for _op, _sub in _flat(_node):
        _e = sp.sympify(atomize(_ast.unparse(_sub), 6), locals=_pf_loc)
        _pieces.append(_e if _op == "Add" else -_e)
    _pf_expr = sp.Add(*_pieces)
    _rec_on = reconstruct(_pf_expr, prefactor=True)
    _rec_off = reconstruct(_pf_expr, prefactor=False)
    for cond, msg in [
        (_rec_on.startswith("-1/chain('as', 2, [1, 6], 5)*("), "prefactor=True -> leading -1/A*("),
        (_rec_on.count("/chain('as', 2, [1, 6], 5)") == 1, "prefactor=True is single-story"),
        (_rec_off.endswith(")/chain('as', 2, [1, 6], 5)"), "prefactor=False keeps two-story (...)/A"),
        (not _rec_off.startswith("-1/chain"), "prefactor=False applies no prefactor (opt-in lock)"),
    ]:
        print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")
        if not cond:
            _fails.append(msg)

    print("Round-trip value preservation:")
    for e in [
        ab(1, 2) * sb(1, 2) * ab(1, 3),
        ab(1, 2) ** 2 * sb(1, 2) ** 2,
        ab(1, 3) * sb(2, 4) / (ab(1, 2) * sb(1, 2)),
        ab(1, 2) ** 3 / (ab(2, 3) * ab(3, 4) * ab(4, 1)),
        ab(1, 2) * sb(1, 2) + ab(1, 3) * sb(1, 3),
        -2 * ab(1, 2) * sb(1, 2),
        sp.Rational(1, 2) * ab(1, 2) * sb(2, 3),
    ]:
        _roundtrip(e, 6)

    # reconstruct of 0
    assert reconstruct("0") == "0"
    print("  [ok ] reconstruct('0') == '0'")

    print()
    if _fails:
        print(f"FAILED ({len(_fails)}): {_fails}")
        return 1
    print("ALL RECONSTRUCT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
