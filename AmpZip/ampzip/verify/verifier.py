"""Randomized finite-field equality / zero checks with exact modular arithmetic.

``equal(A, B, n)`` returns True when all sampled cross-multiplied comparisons
(numA*denB == numB*denA) agree, and False when a modular mismatch is found.
Agreement is probabilistic evidence for rational equality on the massless on-shell
variety. Exhausting the resampling budget raises RuntimeError: the comparison is
inconclusive, not a mismatch. ``zero(A, n)`` checks against zero in the same way.

For independent samples the conditional false-pass bound is the product over primes
of min(1, d_p / (alpha_p * p))**k. Here d_p is the total degree after substituting the
Cramer solution and clearing denominators, alpha_p is the fraction of proposals
accepted by the sampler/evaluator, and the resulting polynomial must remain nonzero
modulo each prime. Two fixed primes do not certify this last assumption. We do not
compute degree, acceptance-fraction or coefficient-height certificates and quote no
numerical error bound. True identities agree at admissible points where reduction
is defined; failure to obtain enough such points gives no conclusion.

Evaluation runs in the engine when both sides arrive as text (or engine handles, or sympy
numbers): each is parsed once into a resident syntax tree (`ampzip_engine.parse_ast`, no
expansion) and evaluated at every point as a fraction pair with no division
(`Ast.eval_pair`), then cross-multiplied here. The sampler, the random stream, the primes
and `k` are the same as on the sympy path. A general sympy object takes the sympy path,
unchanged: printing it would cost more than that path does. The sympy path is also the
fallback for text the engine cannot parse (an `I` coefficient, a float, an atom outside
ab/sb). The engine path resamples points at which either denominator vanishes; the
sympy path resamples undefined modular evaluations of the cross-multiplied products.
"""
from __future__ import annotations

import random

import sympy as sp

from .field import eval_modp, DegeneratePoint
from .kinematics import sample_onshell_modp, vals_from_point

# 2^31-1 (Mersenne) and the next prime below it; nonzero reduction is an assumption.
DEFAULT_PRIMES = (2_147_483_647, 2_147_483_629)
DEFAULT_K = 16


def _strip_opaque_den(e):
    """Rewrite the engine's `opaque_den(S)` marker (its symbolic 1/S for a polynomial-sum
    denominator factor) into an ordinary reciprocal `1/S`, so the sum lands in the real
    denominator where `together` / cross-multiply handle it. No-op when the marker is absent."""
    return e.replace(
        lambda x: x.is_Function and x.func.__name__ == "opaque_den" and len(x.args) == 1,
        lambda x: 1 / x.args[0],
    )


def _num_den(expr):
    """Combine into a single fraction and split into (numerator, denominator)."""
    return sp.fraction(sp.together(_strip_opaque_den(sp.sympify(expr))))


def _engine_source(x):
    """The text the engine can parse, when producing it is free: a string, an engine handle
    (`to_str` runs in Rust), or a sympy number. A general sympy object returns None — printing
    one is slower than the sympy path itself, so that path keeps it."""
    if isinstance(x, str):
        return x
    if isinstance(x, sp.Number):
        return str(x)
    if not isinstance(x, sp.Basic) and hasattr(x, "to_str"):
        return x.to_str()
    return None


def _engine_asts(A, B):
    """Resident engine syntax trees for both sides, or None when the engine path does not apply."""
    sa, sb_ = _engine_source(A), _engine_source(B)
    if sa is None or sb_ is None:
        return None
    try:
        import ampzip_engine as E
        if not hasattr(E, "parse_ast"):
            return None
        return E.parse_ast(sa), E.parse_ast(sb_)
    except Exception:  # noqa: BLE001  (engine absent, or a string outside its grammar)
        return None


def _equal_engine(astA, astB, n, k, primes, sampler, rng, max_resample) -> bool:
    for p in primes:
        got = 0
        tries = 0
        while got < k:
            tries += 1
            if tries > k + max_resample:
                raise RuntimeError(
                    f"too many singular/degenerate points (n={n}, p={p})")
            pt = sampler(n, p, rng)
            if pt is None:
                continue
            nA, dA = astA.eval_pair(pt["lam"], pt["lamt"], p)
            nB, dB = astB.eval_pair(pt["lam"], pt["lamt"], p)
            if dA == 0 or dB == 0:
                continue
            if (nA * dB) % p != (nB * dA) % p:
                return False
            got += 1
    return True


def equal(A, B, n, k=DEFAULT_K, primes=DEFAULT_PRIMES,
          sampler=sample_onshell_modp, rng=None, max_resample=64, engine=True) -> bool:
    """Check agreement at random on-shell points for n legs using exact arithmetic.

    True means every sampled comparison passed; False means a modular mismatch was
    found. Agreement is not a deterministic proof. RuntimeError on resampling
    exhaustion means inconclusive. Unsupported inputs and evaluation failures raise
    their own errors rather than returning an equality verdict.

    `engine=False` forces the sympy evaluation path (the differential-test reference).
    """
    if rng is None:
        rng = random.Random()
    if engine:
        asts = _engine_asts(A, B)
        if asts is not None:
            return _equal_engine(asts[0], asts[1], n, k, primes, sampler, rng, max_resample)
    numA, denA = _num_den(A)
    numB, denB = _num_den(B)
    atoms = (numA.atoms(sp.Function) | denA.atoms(sp.Function)
             | numB.atoms(sp.Function) | denB.atoms(sp.Function))

    for p in primes:
        got = 0
        tries = 0
        while got < k:
            tries += 1
            if tries > k + max_resample:
                raise RuntimeError(
                    f"too many singular/degenerate points (n={n}, p={p})")
            pt = sampler(n, p, rng)
            if pt is None:
                continue
            vals = vals_from_point(pt, atoms, p)
            try:
                lhs = (eval_modp(numA, vals, p) * eval_modp(denB, vals, p)) % p
                rhs = (eval_modp(numB, vals, p) * eval_modp(denA, vals, p)) % p
            except DegeneratePoint:
                continue
            if lhs != rhs:
                return False
            got += 1
    return True


def zero(A, n, k=DEFAULT_K, primes=DEFAULT_PRIMES,
         sampler=sample_onshell_modp, rng=None, max_resample=64, engine=True) -> bool:
    """Check A against zero with the same probabilistic/inconclusive semantics as equal."""
    return equal(A, sp.Integer(0), n, k=k, primes=primes,
                 sampler=sampler, rng=rng, max_resample=max_resample, engine=engine)
