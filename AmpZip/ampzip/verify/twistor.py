"""Momentum-twistor kinematics over F_p.

Momentum twistors Z_i = (lambda_i, mu_i) are unconstrained. Sampled freely, momentum
conservation and masslessness hold automatically through Hodges' construction of the
square spinor. This gives an on-shell sampler the verifier accepts as
`equal(..., sampler=sample_twistor_modp)`, and through the twistors' mu components it
evaluates the momentum-twistor 4-brackets ⟨ijkl⟩ = det(Z_i,Z_j,Z_k,Z_l).

In twistor variables the complete set of invariants is finite, the 2-brackets ⟨ij⟩, the
existing `ab`, and the 4-brackets ⟨ijkl⟩, a new `fb` atom. Every pole an n-point massless
amplitude can have, physical or spurious, is a 4-bracket, so the pole library for
reconstruction is this finite, complete set.

The convention is the standard Hodges one.
    ⟨i j⟩       = lambda_i x lambda_j                       (2x2 minor)
    lambdatilde_i = ( ⟨i,i+1⟩ mu_{i-1} + ⟨i+1,i-1⟩ mu_i
                      + ⟨i-1,i⟩ mu_{i+1} ) / ( ⟨i-1,i⟩⟨i,i+1⟩ )   (cyclic)
    [i j]       = lambdatilde_i x lambdatilde_j
    ⟨i j k l⟩   = det( Z_i, Z_j, Z_k, Z_l ),  Z_i = (lambda_i, mu_i)
Momentum conservation Sum_i lambda_i lambdatilde_i == 0 is automatic, and the tests verify it.
"""
from __future__ import annotations

import random

import sympy as sp

from .field import DegeneratePoint, modinv
from .kinematics import sample_onshell_modp  # noqa: F401  re-exported

DEFAULT_PRIMES = (2_147_483_647, 2_147_483_629)
DEFAULT_K = 16

# The 1-indexed 4-bracket atom, fb(1,2,3,4) = <1 2 3 4>.
fb = sp.Function("fb")


def _minor(u, v, p):
    return (u[0] * v[1] - u[1] * v[0]) % p


def _det4_modp(rows, p):
    """The determinant of a 4x4 matrix, a list of 4 rows, mod p by Gaussian elimination."""
    m = [r[:] for r in rows]
    det = 1
    for c in range(4):
        piv = next((r for r in range(c, 4) if m[r][c] % p != 0), None)
        if piv is None:
            return 0
        if piv != c:
            m[c], m[piv] = m[piv], m[c]
            det = (-det) % p
        det = (det * m[c][c]) % p
        inv = modinv(m[c][c], p)
        for r in range(c + 1, 4):
            f = (m[r][c] * inv) % p
            if f:
                m[r] = [(m[r][k] - f * m[c][k]) % p for k in range(4)]
    return det % p


def sample_twistor_modp(n, p, rng):
    """Sample one on-shell, momentum-conserving point via free momentum twistors.

    Returns {'n','p','lam','lamt','Z'}, the dict shape of sample_onshell_modp, which
    equal(..., sampler=...) and vals_from_point accept, plus 'Z', the length-n list of
    4-vectors for 4-bracket evaluation. Returns None if a Hodges denominator
    ⟨i-1,i⟩⟨i,i+1⟩ vanishes mod p, and the caller resamples."""
    lam = [[rng.randrange(p), rng.randrange(p)] for _ in range(n)]
    mu = [[rng.randrange(p), rng.randrange(p)] for _ in range(n)]
    lamt = [None] * n
    for i in range(n):
        im1, ip1 = (i - 1) % n, (i + 1) % n
        a_im1_i = _minor(lam[im1], lam[i], p)      # ⟨i-1 i⟩
        a_i_ip1 = _minor(lam[i], lam[ip1], p)      # ⟨i i+1⟩
        a_ip1_im1 = _minor(lam[ip1], lam[im1], p)  # ⟨i+1 i-1⟩
        denom = (a_im1_i * a_i_ip1) % p
        if denom == 0:
            return None
        inv = modinv(denom, p)
        lamt[i] = [((a_i_ip1 * mu[im1][b] + a_ip1_im1 * mu[i][b]
                     + a_im1_i * mu[ip1][b]) % p) * inv % p for b in (0, 1)]
    Z = [[lam[i][0], lam[i][1], mu[i][0], mu[i][1]] for i in range(n)]
    return {"n": n, "p": p, "lam": lam, "lamt": lamt, "Z": Z}


def sample_twistor_fb_zero(n, p, rng, quad):
    """Sample a twistor point on the surface ⟨ijkl⟩ = 0, quad being the 1-indexed (i,j,k,l).
    Z_l is set to a*Z_i + b*Z_j + c*Z_k with random a, b, c so the four twistors are
    dependent while the other invariants stay generic. Returns the dict shape of
    sample_twistor_modp, or None if the Hodges denominator degenerates."""
    i, j, k, l = (q - 1 for q in quad)
    lam = [[rng.randrange(p), rng.randrange(p)] for _ in range(n)]
    mu = [[rng.randrange(p), rng.randrange(p)] for _ in range(n)]
    Z = [[lam[t][0], lam[t][1], mu[t][0], mu[t][1]] for t in range(n)]
    a, b, c = rng.randrange(p), rng.randrange(p), rng.randrange(p)
    Z[l] = [(a * Z[i][t] + b * Z[j][t] + c * Z[k][t]) % p for t in range(4)]
    lam[l] = [Z[l][0], Z[l][1]]
    mu[l] = [Z[l][2], Z[l][3]]
    lamt = [None] * n
    for t in range(n):
        tm1, tp1 = (t - 1) % n, (t + 1) % n
        a1 = _minor(lam[tm1], lam[t], p)
        a2 = _minor(lam[t], lam[tp1], p)
        a3 = _minor(lam[tp1], lam[tm1], p)
        d = (a1 * a2) % p
        if d == 0:
            return None
        inv = modinv(d, p)
        lamt[t] = [((a2 * mu[tm1][bb] + a3 * mu[t][bb] + a1 * mu[tp1][bb]) % p) * inv % p
                   for bb in (0, 1)]
    return {"n": n, "p": p, "lam": lam, "lamt": lamt, "Z": Z}


def vals_from_twistor_point(point, atoms, p):
    """Map the ab, sb and fb Function atoms to F_p values at a twistor point. ab and sb are
    the 2x2 minors of lam and lamt, as in kinematics.vals_from_point, and fb(i,j,k,l) is the
    4x4 twistor determinant ⟨ijkl⟩. Labels are 1-indexed."""
    lam, lamt, Z = point["lam"], point["lamt"], point["Z"]
    vals = {}
    for a in atoms:
        nm = a.func.__name__
        if nm == "ab":
            i, j = int(a.args[0]) - 1, int(a.args[1]) - 1
            vals[a] = _minor(lam[i], lam[j], p)
        elif nm == "sb":
            i, j = int(a.args[0]) - 1, int(a.args[1]) - 1
            vals[a] = _minor(lamt[i], lamt[j], p)
        elif nm == "fb":
            idx = [int(x) - 1 for x in a.args]
            vals[a] = _det4_modp([Z[t] for t in idx], p)
        else:
            raise ValueError(f"vals_from_twistor_point: unsupported atom {nm}")
    return vals


def equal_twistor(A, B, n, k=DEFAULT_K, primes=DEFAULT_PRIMES, rng=None, max_resample=128):
    """verify.equal over twistor points, so expressions may contain the 4-bracket fb atoms
    besides ab and sb. It cross-multiplies and never divides."""
    from .field import eval_modp
    from .verifier import _num_den
    if rng is None:
        rng = random.Random()
    numA, denA = _num_den(A)
    numB, denB = _num_den(B)
    atoms = (numA.atoms(sp.Function) | denA.atoms(sp.Function)
             | numB.atoms(sp.Function) | denB.atoms(sp.Function))
    for p in primes:
        got = tries = 0
        while got < k:
            tries += 1
            if tries > k + max_resample:
                raise RuntimeError(f"too many singular twistor points (n={n}, p={p})")
            pt = sample_twistor_modp(n, p, rng)
            if pt is None:
                continue
            vals = vals_from_twistor_point(pt, atoms, p)
            try:
                lhs = (eval_modp(numA, vals, p) * eval_modp(denB, vals, p)) % p
                rhs = (eval_modp(numB, vals, p) * eval_modp(denA, vals, p)) % p
            except DegeneratePoint:
                continue
            if lhs != rhs:
                return False
            got += 1
    return True
