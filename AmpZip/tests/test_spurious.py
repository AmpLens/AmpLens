"""Tests for rational-function decomposition, spurious-pole detection and
cancellation certificates (`ampzip/search/spurious.py`).

Standalone runnable (no pytest):  python tests/test_spurious.py

The fixture is a six-point pair from 2026-08-18: a two-term BCFW-style expression
with the spurious pole <2|(6+1)|5] and a three-term local expression with only
physical poles. The tests require certification for BCFW -> local and rejection
for local -> BCFW. Controls at n=4 and n=5 cover removable and physical poles,
as well as degenerate pole factors that vanish identically on shell.

Needs the ampzip_engine extension, and is skipped when it cannot be imported.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ampzip.simplifier._paths  # noqa: F401,E402

N = 6

# The 3-term form with physical poles only, s_612 and s_561 over a monomial denominator.
LOCAL = (
    "(s(1,2,3)*sb(1,2)*sb(2,3)*ab(4,5)*ab(5,6)"
    "*chain('as',6,[1,2],3)*chain('as',4,[2,3],1))"
    "/(s(1,2)*s(2,3)*s(3,4)*s(4,5)*s(5,6)*s(6,1))"
    "+(chain('as',6,[1,2],3)**2*ab(4,5)**2*sb(1,2)**2)"
    "/(s(6,1)*s(1,2)*s(3,4)*s(4,5)*s(6,1,2))"
    "+(chain('as',4,[2,3],1)**2*ab(5,6)**2*sb(2,3)**2)"
    "/(s(2,3)*s(3,4)*s(5,6)*s(6,1)*s(5,6,1))"
)

# 2-term form: same value, one extra (spurious) pole chain('as',2,[6,1],5).
BCFW = (
    "(chain('as',6,[1,2],3)**3)"
    "/(ab(6,1)*ab(1,2)*sb(3,4)*sb(4,5)*s(6,1,2)*chain('as',2,[6,1],5))"
    "+(chain('as',4,[5,6],1)**3)"
    "/(ab(2,3)*ab(3,4)*sb(5,6)*sb(6,1)*s(5,6,1)*chain('as',2,[6,1],5))"
)

# The engine's canonical spelling of chain('as',2,[6,1],5).
SPURIOUS_POLE = "(-1)*ab(1,2)*sb(1,5) + (-1)*ab(2,6)*sb(5,6)"

_fails = []


def _check(cond, msg):
    print(f"  [{'ok ' if cond else 'FAIL'}] {msg}")
    if not cond:
        _fails.append(msg)


def test_decomposition():
    """The engine puts the two BCFW terms over one denominator, and the whole gap to the
    local form is a single extra sum-pole."""
    print("(1) rational-function decomposition")
    from ampzip.search.spurious import ratfn_parts
    b = ratfn_parts(BCFW, N)
    a = ratfn_parts(LOCAL, N)
    _check(b.n_terms == 24 and b.n_brackets == 288,
           f"BCFW numerator = 24 terms / 288 brackets (got {b.n_terms}/{b.n_brackets})")
    _check(a.n_terms == 126 and a.n_brackets == 1260,
           f"local numerator = 126 terms / 1260 brackets (got {a.n_terms}/{a.n_brackets})")
    _check(a.den_mono == b.den_mono, "identical monomial denominators")
    _check(set(b.opaque) - set(a.opaque) == {SPURIOUS_POLE},
           "the only pole difference is chain('as',2,[6,1],5)")


def test_blind_detection():
    """No target needed: the Dirac chain is removable, both s_ijk are real poles."""
    print("(2) blind spurious-pole detection")
    from ampzip.search.spurious import is_spurious, ratfn_parts
    b = ratfn_parts(BCFW, N)
    for s in b.opaque:
        verdict, stats = is_spurious(b.num, s, N, k=4)
        want = (s == SPURIOUS_POLE)
        _check(verdict is want,
               f"{'spurious' if want else 'physical '} {s[:44]:<44} "
               f"(vanished {stats['vanishing']}/{stats['points']})")


def test_certificate():
    """BCFW -> local certifies, and the reverse, which adds a pole, does not."""
    print("(3) cancellation certificate")
    from ampzip.search.spurious import certify_cancellation
    c = certify_cancellation(BCFW, LOCAL, N, k=4)
    _check(c.dropped == (SPURIOUS_POLE,), "dropped pole identified")
    _check(c.same_den_mono, "same monomial denominator")
    _check(c.quotient_ok is True, "exact division N_src == S * N_dst")
    _check(c.values_equal is True, "the two forms are equal as functions")
    _check(c.ok, "BCFW -> local certificate PASSES")

    r = certify_cancellation(LOCAL, BCFW, N, k=4)
    _check(r.dropped == () and r.quotient_ok is None and r.values_equal is True and not r.ok,
           "local -> BCFW (adds a pole) is left uncertified")


def test_generic_n():
    """Small controls at n=4 and n=5, beyond the 6-point fixture."""
    print("(4) generic n")
    from ampzip.search.spurious import certify_cancellation, is_spurious, ratfn_parts

    # n=5, spurious: momentum conservation makes chain('as',1,[2,3],4) == ab(1,5)*sb(4,5)
    # on-shell, so the numerator vanishes on the pole and the whole thing is 1/ab(1,2).
    # The divisibility holds only on shell.
    x5 = "ab(1,5)*sb(4,5)/(chain('as',1,[2,3],4)*ab(1,2))"
    r5 = ratfn_parts(x5, 5)
    v, st = is_spurious(r5.num, r5.opaque[0], 5, k=4)
    _check(v is True, f"n=5 chain('as',1,[2,3],4) spurious (vanished {st['vanishing']}/{st['points']})")
    _check(certify_cancellation(x5, "1/ab(1,2)", 5, k=4).ok,
           "n=5 certificate down to 1/ab(1,2)")

    # n=5, physical: s(1,2,3) is a real pole of a Parke-Taylor-like expression.
    r5p = ratfn_parts("ab(1,2)**4/(ab(1,2)*ab(2,3)*ab(3,4)*ab(4,5)*s(1,2,3))", 5)
    v, _ = is_spurious(r5p.num, r5p.opaque[0], 5, k=4)
    _check(v is False, "n=5 s(1,2,3) correctly reported physical")

    # n=4, degenerate: s(1,2,3) == s(4) == 0 identically on-shell, so there is no
    # surface to sample and no verdict is claimed.
    s123 = "ab(1,2)*sb(1,2)+ab(1,3)*sb(1,3)+ab(2,3)*sb(2,3)"
    v, st = is_spurious("ab(1,2)*sb(1,2)", s123, 4, k=4)
    _check(v is None and st["reason"] == "degenerate_pole",
           "n=4 s(1,2,3) flagged degenerate (identically 0 on-shell), no verdict claimed")


def main():
    try:
        import ampzip_engine  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[skip] Rust engine unavailable ({e!r})")
        return 0
    test_decomposition()
    test_blind_detection()
    test_certificate()
    test_generic_n()
    print()
    if _fails:
        print(f"FAILED ({len(_fails)}): {_fails}")
        return 1
    print("ALL SPURIOUS-POLE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
