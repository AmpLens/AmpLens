"""Finite-field spurious-pole checks and cancellation certificates.

GBFS rewrites numerator brackets and can cancel denominator factors when the
resulting numerator is divisible by them. This module checks spurious poles and
supplied cancellation candidates. Removing a denominator factor S requires the
combined numerator to be divisible by S on shell, and hence to vanish on S = 0.
The checks below test this vanishing condition and a proposed quotient.

`is_spurious(num, S, n)` samples on-shell points over F_p with S = 0 and checks whether
the supplied numerator vanishes there. No target expression is needed.
`certify_cancellation(src, dst, n)` checks that the monomial denominators agree, that dst
adds no opaque factors and drops at least one, and that each dropped factor passes the
spurious-pole check. It also tests `num(src) == product(dropped_factors) * num(dst)` and
`src == dst` on shell.

Arithmetic at each sampled point is exact over F_p. Agreement provides
probabilistic evidence for the tested relations, as in `ampzip.verify.equal`.

This module does not construct the quotient `num(src)/S` in bracket notation from `src`
alone. Recovering bracket notation from a quotient computed in a parametrised chart is the
separate reconstruction problem of `finiteflow.py`. Given a candidate `dst`, this module
checks the relation by finite-field evaluations.
"""
from __future__ import annotations

import random
from typing import NamedTuple, Optional

import sympy as sp

import ampzip_engine as E

from ampzip.verify.field import DegeneratePoint, eval_modp, modinv
from ampzip.verify.kinematics import vals_from_point
from ampzip.verify.verifier import DEFAULT_PRIMES, equal, zero

_T = sp.Symbol("_t")


# --------------------------------------------------------------------------- #
# rational-function decomposition, from an engine handle to bracket strings
# --------------------------------------------------------------------------- #

def _bracket(sq: bool, i: int, j: int, e: int = 1) -> str:
    s = f"{'sb' if sq else 'ab'}({i},{j})"
    return f"{s}**{e}" if e > 1 else s


def _monomial(brs) -> str:
    """`[(square, i, j, exp), ...]` as a bracket monomial string, '1' when empty."""
    return "*".join(_bracket(*b) for b in brs) or "1"


def _polynomial(terms) -> str:
    """`[(coeff, [(square,i,j,exp), ...]), ...]` as a bracket polynomial string."""
    if not terms:
        return "0"
    return " + ".join(f"({c})*{_monomial(brs)}" for c, brs in terms)


class RatFn(NamedTuple):
    """`num / (den_mono * prod(opaque))`, every field a bracket-string."""
    num: str
    den_mono: str
    opaque: tuple          # symbolic sum factors, as bracket-polynomial strings
    n_terms: int
    n_brackets: int
    handle: object

    def as_str(self) -> str:
        den = "*".join([self.den_mono] + [f"({s})" for s in self.opaque])
        return f"({self.num})/({den})"


def ratfn_parts(expr, n: int) -> RatFn:
    """Decompose `expr`, raw user notation, an atomized string or an `Expr`, into a `RatFn`.

    The engine has already put a sum of fractions over a common denominator, so `num` is the
    combined numerator, the object whose divisibility by a spurious pole is at stake.
    """
    if isinstance(expr, str):
        from ampzip.simplifier.atomize import atomize
        h = E.Expr(atomize(expr, n))
    else:
        h = expr
    return RatFn(
        num=_polynomial([(c, brs) for c, brs in h.term_powers()]),
        den_mono=_monomial(h.den_brackets()),
        opaque=tuple(_polynomial(o) for o in h.den_opaque_terms()),
        n_terms=h.num_terms(),
        n_brackets=h.bracket_count(),
        handle=h,
    )


# --------------------------------------------------------------------------- #
# on-shell points constrained to a surface S = 0
# --------------------------------------------------------------------------- #

def _gauge_point(lam, tfree, p: int):
    """Build an on-shell point from free angle spinors `lam` and the first n-2 square
    spinors `tfree`. Legs n-1 and n are solved from momentum conservation.

    This is `param6.phi` generalised to any n >= 4 and to arbitrary ungauged `lam`. Returns
    the verifier's point dict, or None if <n-1,n> == 0 mod p.
    """
    n = len(lam)
    lamt = [list(t) for t in tfree] + [None, None]
    tot = [[0, 0], [0, 0]]
    for i in range(n - 2):
        for a in (0, 1):
            for b in (0, 1):
                tot[a][b] = (tot[a][b] + lam[i][a] * lamt[i][b]) % p
    u, v = lam[n - 2], lam[n - 1]
    det = (u[0] * v[1] - u[1] * v[0]) % p
    if det == 0:
        return None
    inv = modinv(det, p)
    lu, lv = [0, 0], [0, 0]
    for b in (0, 1):
        r0, r1 = (-tot[0][b]) % p, (-tot[1][b]) % p
        lu[b] = ((r0 * v[1] - v[0] * r1) * inv) % p
        lv[b] = ((u[0] * r1 - r0 * u[1]) * inv) % p
    lamt[n - 2], lamt[n - 1] = lu, lv
    return {"n": n, "p": p, "lam": lam, "lamt": lamt}


def _eval_at(sym, atoms, point, p: int) -> int:
    return eval_modp(sym, vals_from_point(point, atoms, p), p) % p


def _roots_modp(coeffs, p: int):
    """Roots in F_p of the polynomial c[0] + c[1] t + ... with coefficients `coeffs`.

    Degrees 1 and 2 are solved in closed form. Every pole atom this package produces, a
    Mandelstam `s_I` or a Dirac chain `<a|P|b]`, is degree 2 in `t` along the lines drawn by
    `_surface_points`, since the angle spinors are frozen and each square spinor is linear
    in `t`. Higher degrees go to sympy over GF(p), and if that is unavailable the caller
    draws another line.
    """
    while coeffs and coeffs[-1] % p == 0:
        coeffs = coeffs[:-1]
    d = len(coeffs) - 1
    if d < 1:
        return []
    if d == 1:
        return [(-coeffs[0] * modinv(coeffs[1], p)) % p]
    if d == 2:
        c, b, a = coeffs[0] % p, coeffs[1] % p, coeffs[2] % p
        disc = (b * b - 4 * a * c) % p
        r = sp.sqrt_mod(disc, p)
        if r is None:
            return []
        inv2a = modinv((2 * a) % p, p)
        return sorted({((-b + int(r)) * inv2a) % p, ((-b - int(r)) * inv2a) % p})
    try:
        poly = sp.Poly(list(reversed(coeffs)), _T, modulus=p, symmetric=False)
        return sorted({int(r) % p for r in poly.ground_roots()})
    except Exception:
        return []


def _surface_points(S_sym, n: int, p: int, rng, want: int, max_lines: int = 256):
    """Yield up to `want` on-shell F_p points that lie on the surface `S_sym == 0`.

    The angle spinors are frozen, so `<n-1,n>` and every `ab` is constant, and the free
    square spinors move along a random affine line. Momentum conservation stays solved for
    every `t`, since the solved spinors are linear in the free ones, so the whole line is on
    shell, every `sb` is quadratic in `t`, and `S(t)` is a low-degree polynomial whose roots
    can be taken exactly.
    """
    atoms = S_sym.atoms(sp.Function)
    out = []
    for _ in range(max_lines):
        if len(out) >= want:
            break
        lam = [[rng.randrange(1, p), rng.randrange(1, p)] for _ in range(n)]
        t0 = [[rng.randrange(1, p), rng.randrange(1, p)] for _ in range(n - 2)]
        tv = [[rng.randrange(1, p), rng.randrange(1, p)] for _ in range(n - 2)]

        def at(t):
            free = [[(t0[i][b] + t * tv[i][b]) % p for b in (0, 1)] for i in range(n - 2)]
            return _gauge_point(lam, free, p)

        if at(0) is None:
            continue

        def f(t):
            pt = at(t)
            return 0 if pt is None else _eval_at(S_sym, atoms, pt, p)

        from ampzip.search.finiteflow import reconstruct_polynomial
        coeffs = reconstruct_polynomial(f, p, max_deg=12)
        if coeffs is None:
            continue
        for r in _roots_modp(list(coeffs), p):
            pt = at(r)
            if pt is None:
                continue
            if _eval_at(S_sym, atoms, pt, p) != 0:      # a bad interpolation
                continue
            out.append(pt)
            if len(out) >= want:
                break
    return out


# --------------------------------------------------------------------------- #
# blind spurious-pole detection
# --------------------------------------------------------------------------- #

def is_spurious(num, S, n: int, k: int = 8, primes=DEFAULT_PRIMES, rng=None):
    """Whether the pole `S` is spurious for the numerator `num`, with no target needed.

    On-shell points on the surface S = 0 are sampled and `num` is checked to vanish there
    in exact arithmetic over F_p. Agreement is probabilistic evidence for the vanishing
    condition.

    Returns `(verdict, stats)`. `verdict` is True or False, or None when no verdict could be
    reached, in which case `stats['reason']` says why. It is `'degenerate_pole'` when S
    itself is identically 0 on the on-shell variety, as `s(1,2,3)` is at n=4, where momentum
    conservation and masslessness force it to vanish, so the surface S=0 is the whole variety
    and the expression is ill-defined, and `'no_points'` when the surface exists but no
    usable point could be constructed.
    """
    if rng is None:
        rng = random.Random(20260818)
    num_sym = sp.sympify(num)
    S_sym = sp.sympify(S)
    if zero(S_sym, n):
        return None, {"points": 0, "vanishing": 0, "reason": "degenerate_pole"}
    n_atoms = num_sym.atoms(sp.Function)
    pts = zeros = 0
    for p in primes:
        for pt in _surface_points(S_sym, n, p, rng, want=k):
            try:
                v = _eval_at(num_sym, n_atoms, pt, p)
            except DegeneratePoint:
                continue
            pts += 1
            zeros += (v == 0)
    if pts == 0:
        return None, {"points": 0, "vanishing": 0, "reason": "no_points"}
    return zeros == pts, {"points": pts, "vanishing": zeros, "reason": "sampled"}


# --------------------------------------------------------------------------- #
# the cancellation certificate
# --------------------------------------------------------------------------- #

class Certificate(NamedTuple):
    ok: bool
    src: RatFn
    dst: RatFn
    dropped: tuple         # the opaque factors present in src and absent from dst
    same_den_mono: bool
    spurious: dict         # (verdict, stats) per S
    quotient_ok: Optional[bool]
    values_equal: Optional[bool]
    steps: tuple


def certify_cancellation(src, dst, n: int, k: int = 8, rng=None) -> Certificate:
    """Certify the representation change `src -> dst` as a spurious-pole cancellation.

    The checks run in order. Both sides must share the same monomial denominator, since
    otherwise the change is not a pure pole cancellation and the quotient identity would be
    off by that ratio. Every opaque factor of `dst` must also be one of `src`, the extra ones
    being the poles dropped. Each dropped pole must be spurious, so `num(src)` vanishes on
    its zero surface. The exact quotient identity `num(src) == (prod dropped) * num(dst)`
    must hold on shell. And `src == dst` must hold as functions, an independent end-to-end
    check kept although the others imply it.
    """
    a, b = ratfn_parts(src, n), ratfn_parts(dst, n)
    same_mono = sp.simplify(sp.sympify(a.den_mono) - sp.sympify(b.den_mono)) == 0
    keep = list(b.opaque)
    dropped = []
    for s in a.opaque:
        if s in keep:
            keep.remove(s)
        else:
            dropped.append(s)
    extra_in_dst = tuple(keep)

    spur = {}
    for s in dropped:
        spur[s] = is_spurious(a.num, s, n, k=k, rng=rng)

    quotient_ok = values_ok = None
    if dropped and not extra_in_dst and same_mono:
        rhs = "*".join([f"({s})" for s in dropped] + [f"({b.num})"])
        quotient_ok = equal(a.num, rhs, n)
    values_ok = equal(a.handle.to_str(), b.handle.to_str(), n)

    ok = bool(same_mono and dropped and not extra_in_dst
              and all(v is True for v, _ in spur.values())
              and quotient_ok and values_ok)

    steps = (
        f"1. common denominator: src = N_src / ({a.den_mono} * "
        + " * ".join(f"({s})" for s in a.opaque) + ")",
        "2. dropped pole(s): " + ", ".join(dropped),
        "3. regularity: N_src vanishes on every sampled on-shell point with the pole = 0",
        "4. exact division: N_src == (dropped) * N_dst on-shell",
        f"5. therefore src = N_dst / ({b.den_mono} * "
        + " * ".join(f"({s})" for s in b.opaque) + ") = dst",
    )
    return Certificate(ok, a, b, tuple(dropped), bool(same_mono), spur,
                       quotient_ok, values_ok, steps)
