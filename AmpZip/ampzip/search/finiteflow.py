"""Functional reconstruction over F_p in the manner of FiniteFlow.

Over a complete four-bracket denominator the numerator's little-group weight is high, so
a weight-matched monomial basis is astronomically large and a monomial fit is intractable.
Functional reconstruction treats the object as a black-box function of a few free
variables and rebuilds it by interpolation, whose cost scales with the answer's
per-variable degree.

The module implements univariate polynomial reconstruction by Newton interpolation and
univariate rational reconstruction by Thiele interpolation, both with automatic degree
detection, and dense and sparse multivariate polynomial interpolation. On-shell
parametrization and numerator reconstruction are separate subsequent steps.

Everything is exact over F_p. A final `equal` or `equal_twistor` test keeps it sound.
"""
from __future__ import annotations

import os
import sys

_SIMP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SIMP not in sys.path:
    sys.path.insert(0, _SIMP)

import numpy as np  # noqa: E402

from ampzip.verify.field import modinv  # noqa: E402


def newton_divided_diff(xs, ys, p):
    """Newton divided-difference coefficients c_k for the interpolating polynomial
    f(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ... over F_p."""
    n = len(xs)
    c = [y % p for y in ys]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            num = (c[i] - c[i - 1]) % p
            den = (xs[i] - xs[i - j]) % p
            c[i] = (num * modinv(den, p)) % p
    return c


def newton_to_monomial(xs, c, p):
    """Convert Newton form (nodes xs, divided-diff coeffs c) to monomial coefficients
    [a_0, a_1, ...] with f(x) = sum a_k x^k, over F_p."""
    n = len(c)
    coeffs = [0] * n
    # Horner-like expansion: poly = c[n-1]; poly = poly*(x - xs[k]) + c[k]
    acc = [0] * n
    acc[0] = c[n - 1] % p
    deg = 0
    for k in range(n - 2, -1, -1):
        # acc <- acc*(x - xs[k]) + c[k]
        new = [0] * n
        for i in range(deg + 1):
            new[i + 1] = (new[i + 1] + acc[i]) % p          # * x
            new[i] = (new[i] - acc[i] * xs[k]) % p           # * (-xs[k])
        deg += 1
        new[0] = (new[0] + c[k]) % p
        acc = new
    return [a % p for a in acc]


def reconstruct_polynomial(fn, p, max_deg=40, extra=3, x0=1):
    """Reconstruct a univariate polynomial fn: F_p -> F_p by Newton interpolation with
    automatic degree detection. It samples at x = x0, x0+1, ... and declares the degree
    found when `extra` further divided-difference coefficients are all zero. Returns the
    monomial coefficient list, or None if it does not stabilize by max_deg."""
    xs, ys = [], []
    x = x0
    while len(xs) < max_deg + extra + 2:
        xs.append(x % p)
        ys.append(fn(x % p) % p)
        x += 1
        c = newton_divided_diff(xs, ys, p)
        # trailing coeffs c[d+1..] all zero => degree d
        if len(c) >= extra + 1:
            tail = c[-extra:]
            if all(t == 0 for t in tail) and any(c[i] != 0 for i in range(len(c) - extra)):
                deg = max(i for i in range(len(c)) if c[i] != 0)
                return newton_to_monomial(xs[:deg + 1], c[:deg + 1], p)
            if all(t == 0 for t in c):  # identically zero
                return [0]
    return None


def thiele_rational(fn, p, max_order=40, x0=1):
    """Reconstruct a univariate rational function fn: F_p -> F_p by Thiele's continued
    fraction of inverse differences with automatic termination. Returns (num_coeffs,
    den_coeffs) as monomial coefficient lists, or None if it does not terminate.

    The continued fraction is grown until appending more sample points no longer changes
    the reconstructed rational, which is checked by re-evaluating at fresh points."""
    xs, ys = [], []
    x = x0

    def inv_diffs(xs, ys):
        # Thiele reciprocal differences, rho[k][i] = rho[k-2][i-1]
        #   + (x_i - x_{i-k}) / (rho[k-1][i] - rho[k-1][i-1]);  rho[-1]=0, rho[0]=ys.
        m = len(xs)
        rho = [[0] * m for _ in range(m)]
        for i in range(m):
            rho[0][i] = ys[i] % p
        for k in range(1, m):
            for i in range(k, m):
                d = (rho[k - 1][i] - rho[k - 1][i - 1]) % p
                if d == 0:
                    return None
                prev = 0 if k == 1 else rho[k - 2][i - 1]
                rho[k][i] = (prev + ((xs[i] - xs[i - k]) % p) * modinv(d, p)) % p
        return rho

    def cf_to_ratfn(a):
        # Convert Thiele coefficients to (num, den) coefficient lists by the recurrence
        # A_{-1}=1, A_0=a_0, B_{-1}=0, B_0=1 with h_k=(x-x_{k-1}).
        num_prev2, num_prev1 = [1], [a[0] % p]
        den_prev2, den_prev1 = [0], [1]

        def _strip(r):
            while len(r) > 1 and r[-1] == 0:
                r.pop()
            return r

        def polyadd(u, v):
            r = [0] * max(len(u), len(v))
            for i, x_ in enumerate(u):
                r[i] = (r[i] + x_) % p
            for i, x_ in enumerate(v):
                r[i] = (r[i] + x_) % p
            return _strip(r)

        def polymul(u, v):
            r = [0] * (len(u) + len(v) - 1)
            for i, ui in enumerate(u):
                if ui:
                    for j, vj in enumerate(v):
                        r[i + j] = (r[i + j] + ui * vj) % p
            return _strip(r)
        for k in range(1, len(a)):
            hk = [(-xs[k - 1]) % p, 1]  # (x - x_{k-1})
            num_k = polyadd(polymul([a[k] % p], num_prev1), polymul(hk, num_prev2))
            den_k = polyadd(polymul([a[k] % p], den_prev1), polymul(hk, den_prev2))
            num_prev2, num_prev1 = num_prev1, num_k
            den_prev2, den_prev1 = den_prev1, den_k
        return ([c % p for c in num_prev1], [c % p for c in den_prev1])

    def evalpoly(coef, xq):
        r = 0
        for cc in reversed(coef):
            r = (r * xq + cc) % p
        return r

    def predicts(num, den, x_test):
        # The reconstructed rational reproduces fn at fresh points.
        for xt in x_test:
            dv = evalpoly(den, xt)
            if dv == 0:
                return False
            if (evalpoly(num, xt) * modinv(dv, p)) % p != fn(xt) % p:
                return False
        return True

    while len(xs) < max_order + 6:
        xs.append(x % p)
        ys.append(fn(x % p) % p)
        x += 1
        if len(xs) < 2:
            continue
        rho = inv_diffs(xs, ys)
        if rho is None:
            continue
        # Thiele continued-fraction coefficients b_0=rho_0, b_1=rho_1, b_k = rho_k - rho_{k-2} for k>=2.
        m = len(xs)
        b = [rho[k][k] for k in range(m)]
        for k in range(2, m):
            b[k] = (rho[k][k] - rho[k - 2][k - 2]) % p
        try:
            num, den = cf_to_ratfn(b)
        except Exception:
            continue
        while len(num) > 1 and num[-1] == 0:
            num.pop()
        while len(den) > 1 and den[-1] == 0:
            den.pop()
        # Accept only if it reproduces fn at fresh points.
        x_test = [(x + 50 + j) % p for j in range(4)]
        if predicts(num, den, x_test):
            return num, den
    return None


def reconstruct_poly_multivar(evalf, nvars, p, max_deg=30, bases=3):
    """Reconstruct a multivariate polynomial `evalf(x_0, ..., x_{nvars-1})` over F_p
    as a dictionary of exponent tuples and coefficients. First estimate the
    degree in the last variable using `bases` choices of the other variables. At
    each interpolation value of the last variable, recursively reconstruct a
    polynomial in the remaining variables. Then interpolate each coefficient in
    the last variable. This is dense interpolation, with sample counts growing
    approximately as the product of the per-variable degrees. `zippel_interpolate` is the
    sparse alternative."""
    if nvars == 1:
        c = reconstruct_polynomial(lambda x: evalf((x,)), p, max_deg)
        if c is None:
            return None
        return {(e,): c[e] % p for e in range(len(c)) if c[e] % p}
    # The degree in the last variable is the maximum over a few random bases for the other variables.
    dlast = 0
    for bi in range(bases):
        base = tuple(((17 * (bi + 1) + 3 * i) % p) for i in range(nvars - 1))
        cser = reconstruct_polynomial(lambda t: evalf(base + (t,)), p, max_deg)
        if cser is None:
            return None
        dlast = max(dlast, len(cser) - 1)
    tpts = [i + 1 for i in range(dlast + 1)]
    slices = []
    for t in tpts:
        sub = reconstruct_poly_multivar(lambda rest: evalf(tuple(rest) + (t,)),
                                        nvars - 1, p, max_deg, bases)
        if sub is None:
            return None
        slices.append(sub)
    monos = set()
    for s in slices:
        monos |= set(s.keys())
    result = {}
    for mono in monos:
        ys = [s.get(mono, 0) for s in slices]
        cc = newton_to_monomial(tpts, newton_divided_diff(tpts, ys, p), p)
        for e, coef in enumerate(cc):
            if coef % p:
                result[mono + (e,)] = coef % p
    return result


def _mono_eval(expo, pt, p):
    """Evaluate a monomial with exponent tuple `expo` at point `pt`, both of length k, mod p."""
    r = 1
    for e, v in zip(expo, pt):
        if e:
            r = (r * pow(v % p, e, p)) % p
    return r


def _solve_square_modp(A, b, p):
    """Solve the square system A x = b mod p by numpy-vectorized Gauss-Jordan.
    Returns x as a list, or None if singular. Products stay below 2^63 for p ~ 2^31 and fit int64."""
    n = len(A)
    M = np.array([list(A[i]) + [b[i]] for i in range(n)], dtype=np.int64) % p
    for c in range(n):
        nz = np.nonzero(M[c:, c] % p)[0]
        if nz.size == 0:
            return None
        pr = c + int(nz[0])
        if pr != c:
            M[[c, pr]] = M[[pr, c]]
        M[c] = (M[c] * pow(int(M[c, c]), p - 2, p)) % p
        f = M[:, c].copy()
        f[c] = 0
        M = (M - np.outer(f, M[c])) % p
    return [int(M[r, n]) % p for r in range(n)]


def zippel_interpolate(evalf, nvars, p, max_deg=30, seed=1, max_pts_factor=3, verbose=False):
    """Zippel sparse multivariate interpolation over F_p. Reconstructs the sparse polynomial
    evalf(x_0..x_{v-1}) as a monomial dict {exponent_tuple: coeff}.

    It proceeds variable by variable. The monomial skeleton over the variables already
    processed is kept, under Zippel's probabilistic sparsity assumption that a generic anchor
    makes the support at the anchor equal the full support. To add x_j it solves a T x T
    linear system, T being the number of skeleton terms, at each of deg_j+1 sample values of
    x_j, then interpolates each skeleton coefficient in x_j. The cost is about
    sum_j (deg_j+1) * T_j * solve and scales with the number of nonzero terms. The method is
    probabilistic, and a downstream `equal` test turns any wrong skeleton into a miss."""
    import random
    rng = random.Random(seed)
    anchor = [rng.randrange(1, p) for _ in range(nvars)]

    # First the univariate polynomial in x_0 at the anchor for x_1..x_{v-1}.
    c0 = reconstruct_polynomial(lambda x: evalf(tuple([x] + anchor[1:])), p, max_deg)
    if c0 is None:
        return None
    current = {(e,): c0[e] % p for e in range(len(c0)) if c0[e] % p}

    for j in range(1, nvars):
        # The degree in x_j, probed at a random base for the other variables.
        dj = 0
        for bi in range(3):
            base = [rng.randrange(1, p) for _ in range(nvars)]
            cj = reconstruct_polynomial(
                lambda t: evalf(tuple(base[:j] + [t] + base[j + 1:])), p, max_deg)
            if cj is None:
                return None
            dj = max(dj, len(cj) - 1)
        skel = list(current.keys())
        T = len(skel)
        vs = [l + 1 for l in range(dj + 1)]
        coeffs_per_v = []
        for vl in vs:
            # The skeleton coefficients at x_j=vl with the later variables at anchor[j+1:].
            rows, rhs = [], []
            tries = 0
            while len(rows) < T:
                tries += 1
                if tries > max_pts_factor * T + 50:
                    return None  # persistent singular system -> give up (sound: a miss)
                b = [rng.randrange(1, p) for _ in range(j)]
                row = [_mono_eval(m, b, p) for m in skel]
                fullpt = tuple(b + [vl] + anchor[j + 1:])
                rows.append(row)
                rhs.append(evalf(fullpt) % p)
                sol = _solve_square_modp(rows, rhs, p) if len(rows) == T else None
                if len(rows) == T and sol is None:
                    rows.pop()
                    rhs.pop()  # last point made it singular; resample
            coeffs_per_v.append(dict(zip(skel, _solve_square_modp(rows, rhs, p))))
        new_current = {}
        for m in skel:
            ys = [coeffs_per_v[l][m] for l in range(len(vs))]
            cc = newton_to_monomial(vs, newton_divided_diff(vs, ys, p), p)
            for e, coef in enumerate(cc):
                if coef % p:
                    new_current[m + (e,)] = coef % p
        current = new_current
        if verbose:
            print(f"  [zippel] var {j}: deg={dj}, skeleton terms={len(current)}", flush=True)
    return current


if __name__ == "__main__":
    import traceback
    p = 2_147_483_647
    tests = []

    def check(name, cond):
        tests.append((name, cond))

    # Polynomial reconstruction.
    poly = [7, 0, 3, 5]  # 7 + 3x^2 + 5x^3
    def fp(x): return (7 + 3 * x * x + 5 * x * x * x) % p
    got = reconstruct_polynomial(fp, p)
    check("poly deg3 exact", got is not None and got[:4] == [c % p for c in poly] and all(g == 0 for g in got[4:]))

    def fp0(x): return 0
    check("poly zero", reconstruct_polynomial(fp0, p) == [0])

    # Rational reconstruction of f(x) = (3x^2 + 5) / (x + 2).
    def fr(x):
        num = (3 * x * x + 5) % p
        den = (x + 2) % p
        return num * modinv(den, p) % p
    r = thiele_rational(fr, p)
    ok_r = False
    if r is not None:
        num, den = r
        # Verify at fresh points.
        def ev(coef, xq):
            v = 0
            for cc in reversed(coef):
                v = (v * xq + cc) % p
            return v
        ok_r = all((ev(num, x) * modinv(ev(den, x), p)) % p == fr(x) for x in (100, 101, 102, 103))
    check("rational (3x^2+5)/(x+2) exact", ok_r)

    passed = sum(1 for _, c in tests if c)
    for name, c in tests:
        print(f"{'PASS' if c else 'FAIL'} {name}")
    print(f"\n{passed}/{len(tests)} passed")
