"""On-shell massless kinematics over F_p.

Sample 2-component spinors lambda_i, lambdatilde_i in F_p subject to momentum
conservation ``sum_i lambda_i^a lambdatilde_i^b == 0 (mod p)``, then read off the
brackets as 2x2 minors:

    ab(i,j) = <i j> = lam[i][0]*lam[j][1]  - lam[i][1]*lam[j][0]   (mod p)
    sb(i,j) = [i j] = lamt[i][0]*lamt[j][1] - lamt[i][1]*lamt[j][0] (mod p)

Because the spinors are rank 2 by construction, Schouten holds identically as the Plücker
relation, and because momentum conservation is solved, the momentum and momentum-squared
identities hold too. Hence on-shell-equal expressions evaluate equal at admissible points
where reduction is defined. A verifier that exhausts its sampling budget has no conclusion
about equality.
"""
from __future__ import annotations

from .field import modinv


def sample_onshell_modp(n: int, p: int, rng):
    """Sample one on-shell, momentum-conserving massless point over F_p.

    Returns ``{'n', 'p', 'lam', 'lamt'}`` with 0-indexed spinor lists of length n, or
    ``None`` if the 2x2 momentum-conservation solve is singular mod p, in which case the
    caller resamples. ``n == 3`` is a structural degeneracy and is not supported.
    """
    if n < 4:
        if n == 3:
            raise NotImplementedError("n=3 needs a special-kinematics sampler")
        raise ValueError(f"n must be >= 4, got {n}")

    lam = [[rng.randrange(p), rng.randrange(p)] for _ in range(n)]
    lamt = [[rng.randrange(p), rng.randrange(p)] for _ in range(n - 2)]

    # S[a][b] = sum_{i=0}^{n-3} lam[i][a] * lamt[i][b]
    S = [[0, 0], [0, 0]]
    for i in range(n - 2):
        for a in (0, 1):
            la = lam[i][a]
            for b in (0, 1):
                S[a][b] = (S[a][b] + la * lamt[i][b]) % p

    A = lam[n - 2]
    B = lam[n - 1]
    det = (A[0] * B[1] - A[1] * B[0]) % p
    if det == 0:
        return None
    inv = modinv(det, p)

    ltn2 = [0, 0]
    ltn1 = [0, 0]
    for b in (0, 1):
        r0 = (-S[0][b]) % p
        r1 = (-S[1][b]) % p
        # Cramer's rule on [[A0,B0],[A1,B1]] [x;y] = [r0;r1]
        ltn2[b] = ((r0 * B[1] - B[0] * r1) * inv) % p   # lamt[n-2][b]
        ltn1[b] = ((A[0] * r1 - r0 * A[1]) * inv) % p   # lamt[n-1][b]

    lamt_full = lamt + [ltn2, ltn1]
    return {"n": n, "p": p, "lam": lam, "lamt": lamt_full}


def vals_from_point(point, atoms, p: int) -> dict:
    """Map each ab/sb Function atom, with 1-indexed labels, to its F_p minor.

    Any argument order works, since antisymmetry falls out of the minor."""
    lam = point["lam"]
    lamt = point["lamt"]
    vals = {}
    for a in atoms:
        name = a.func.__name__
        if name == "ab":
            v = lam
        elif name == "sb":
            v = lamt
        else:
            raise ValueError(f"vals_from_point: unsupported atom {name} "
                             f"(massive atoms need a massive sampler)")
        i = int(a.args[0]) - 1
        j = int(a.args[1]) - 1
        vals[a] = (v[i][0] * v[j][1] - v[i][1] * v[j][0]) % p
    return vals
