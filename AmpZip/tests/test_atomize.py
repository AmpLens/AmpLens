"""Verifier-gated tests for the Atomizer (simplifier/atomize.py).

Standalone runnable (no pytest):  python tests/test_atomize.py

Every atomization is checked value-equal to a hand-written atomic reference by the
finite-field test of ampzip.verify.equal. The monomial-denominator rule, degeneracies and
parser rejections are also checked.
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
from ampzip.verify import equal, zero  # noqa: E402
from ampzip.simplifier.atomize import atomize, AtomizeError, has_opaque_den  # noqa: E402

_LOC = {"ab": ab, "sb": sb}
_fails = []


def _eq(raw, n, ref):
    got = sp.sympify(atomize(raw, n), locals=_LOC)
    ok = equal(got, ref, n, rng=random.Random(0)) is True
    print(f"  [{'ok ' if ok else 'FAIL'}] {raw:42} == {ref}")
    if not ok:
        _fails.append(raw)


def _str_eq(raw, n, expect_str):
    got = atomize(raw, n)
    ok = got == expect_str
    print(f"  [{'ok ' if ok else 'FAIL'}] {raw:42} -> {got!r}")
    if not ok:
        _fails.append(raw)


def _is_zero(raw, n):
    got = sp.sympify(atomize(raw, n), locals=_LOC)
    ok = (got == 0) or (zero(got, n, rng=random.Random(0)) is True)
    print(f"  [{'ok ' if ok else 'FAIL'}] {raw:42} == 0")
    if not ok:
        _fails.append(raw)


def _rejects(raw, n, why):
    try:
        out = atomize(raw, n)
        print(f"  [FAIL] {raw:42} should reject ({why}) but gave {out!r}")
        _fails.append(raw)
    except AtomizeError:
        print(f"  [ok ] {raw:42} rejected ({why})")


def _accepts_opaque(raw, n, why):
    """A multi-particle pole in a denominator is (default allow_opaque_den=True) kept as
    a symbolic opaque_den(S) marker for the gbfs backend."""
    try:
        out = atomize(raw, n)
    except AtomizeError as e:
        print(f"  [FAIL] {raw:42} should accept as opaque_den ({why}) but rejected: {e}")
        _fails.append(raw)
        return
    ok = has_opaque_den(out)
    print(f"  [{'ok ' if ok else 'FAIL'}] {raw:42} -> opaque_den ({why})")
    if not ok:
        print(f"       [FAIL] expected an opaque_den(S) marker, got {out!r}")
        _fails.append(raw)


def main():
    print("Mandelstam (textbook s_ij = <ij>[ji] = ab(i,j)*sb(j,i) = -ab(i,j)*sb(i,j)):")
    _eq("s(1,2)", 4, ab(1, 2) * sb(2, 1))
    _eq("s(1,2,3)", 6, ab(1, 2) * sb(2, 1) + ab(1, 3) * sb(3, 1) + ab(2, 3) * sb(3, 2))
    _eq("s(1,2,6)", 6, ab(1, 2) * sb(2, 1) + ab(1, 6) * sb(6, 1) + ab(2, 6) * sb(6, 2))
    _eq("s(1,2)/3", 4, ab(1, 2) * sb(2, 1) / 3)
    _eq("1/2*s(1,2)", 4, sp.Rational(1, 2) * ab(1, 2) * sb(2, 1))  # exact rational, no float

    print("Mandelstam, independent of n, with no momentum conservation baked in:")
    _is_zero("s(1)", 4)                   # s of at most one leg is 0 by masslessness, at any n
    # s_I expands the same way regardless of n, with no complement swap and no zeroing of the
    # full set. These on-shell identities still hold by the verifier.
    _is_zero("s(1,2,3,4,5,6)", 6)         # the full set is on-shell zero by momentum conservation
    _eq("s(1,2,6)", 6, sp.sympify(atomize("s(3,4,5)", 6), locals=_LOC))  # s_I == s_complement on-shell
    # s(1,2) stays a bracket even at n=2, independent of n. The verifier needs n>=4, so the
    # string is checked directly. The parentheses come from the default allow_opaque_den
    # numerator wrapping, and the leading minus from the textbook s=<ij>[ji]=-ab*sb.
    _str_eq("s(1,2)", 2, "(-ab(1, 2)*sb(1, 2))")
    _str_eq("s(1,2)", 6, "(-ab(1, 2)*sb(1, 2))")

    print("Dirac chains:")
    _eq("chain('as',1,[3],4)", 4, ab(1, 3) * sb(3, 4))
    _eq("chain('as',3,[1,2],6)", 6, ab(3, 1) * sb(1, 6) + ab(3, 2) * sb(2, 6))
    _eq("chain('as',5,[1,6],2)", 6, ab(5, 1) * sb(1, 2) + ab(5, 6) * sb(6, 2))
    _eq("chain('sa',1,[2],3)", 4, sb(1, 2) * ab(2, 3))
    _eq("chain('aa',1,[2,3],[4,5],6)", 6,
        ab(1, 2) * sb(2, 4) * ab(4, 6) + ab(1, 2) * sb(2, 5) * ab(5, 6)
        + ab(1, 3) * sb(3, 4) * ab(4, 6) + ab(1, 3) * sb(3, 5) * ab(5, 6))
    _eq("chain('as',3,[3,4],6)", 6, ab(3, 4) * sb(4, 6))  # endpoint-in-channel term drops

    print("Chain internal consistency (momentum conservation):")
    _is_zero("chain('as',1,[1,2,3,4,5,6],2)", 6)   # <1|P_total|2] == 0

    print("Denominator poles (default allow_opaque_den=True: multi-particle -> opaque_den):")
    _accepts_opaque("ab(4,5)/s(1,2,3)", 6, "multi-particle s in denominator")
    _accepts_opaque("ab(3,4)/chain('as',5,[1,6],2)", 6, "multi-leg chain in denominator")
    _accepts_opaque("1/s(1,2,3)", 4, "3-particle pole in denominator (sum)")
    _rejects("1/s(1)", 4, "s(1)=0 -> 1/0 singular")  # a singularity, rejected
    # The strict path with allow_opaque_den=False keeps the monomial-denominator rule.
    try:
        atomize("1/s(1,2,3)", 4, allow_opaque_den=False)
        print(f"  [FAIL] {'1/s(1,2,3) [strict]':42} should reject in strict mode")
        _fails.append("1/s(1,2,3) [strict]")
    except AtomizeError:
        print(f"  [ok ] {'1/s(1,2,3) [strict]':42} rejected (allow_opaque_den=False)")
    # allowed in denominator (stay monomial):
    _eq("1/s(1,2)", 4, 1 / (ab(1, 2) * sb(2, 1)))
    _eq("ab(1,3)/chain('as',1,[2],4)", 4, ab(1, 3) / (ab(1, 2) * sb(2, 4)))

    print("Safe-parse rejections:")
    for junk, why in [("__import__('os')", "dunder"), ("ab(1,2).foo", "attribute"),
                      ("x+1", "unknown name"), ("chain('xx',1,[2],3)", "bad chain type"),
                      ("chain('aa',1,[2],3)", "chain parity mismatch")]:
        _rejects(junk, 4, why)

    print()
    if _fails:
        print(f"FAILED ({len(_fails)}): {_fails}")
        return 1
    print("ALL ATOMIZE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
