"""Randomized finite-field equality / zero checks with exact modular arithmetic.

``equal(A, B, n)`` returns True when all sampled cross-multiplied comparisons
(numA*denB == numB*denA) agree, and False when a modular mismatch is found.
Agreement is probabilistic evidence for rational equality on the massless on-shell
variety. Exhausting the resampling budget raises RuntimeError: the comparison is
inconclusive. ``zero(A, n)`` checks against zero in the same way.

For independent samples the conditional false-pass bound is the product over primes
of min(1, d_p / (alpha_p * p))**k. Here d_p is the total degree after substituting the
Cramer solution and clearing denominators, alpha_p is the fraction of proposals
accepted by the sampler/evaluator, and the resulting polynomial must remain nonzero
modulo each prime. Two fixed primes do not certify this last assumption. We do not
compute degree, acceptance-fraction or coefficient-height certificates and quote no
numerical error bound. True identities agree at admissible points where reduction
is defined. Failure to obtain enough such points gives no conclusion.

Evaluation runs in the engine when both sides arrive as text (or engine handles, or sympy
numbers): each is parsed once into a resident syntax tree (`ampzip_engine.parse_ast`, no
expansion) and evaluated at every point as a fraction pair with no division
(`Ast.eval_pair`), then cross-multiplied here. The sampler, the random stream, the primes
and `k` are the same as on the sympy path. A general sympy object takes the sympy path,
unchanged: printing it would cost more than that path does. The sympy path is also the
fallback for text the engine cannot parse (an `I` coefficient, a float, an atom outside
ab/sb). Both paths resample a point at which either denominator vanishes. A denominator that
vanishes everywhere exhausts the budget and the comparison is inconclusive. The sympy path
also resamples points where a modular evaluation is undefined.
"""
from __future__ import annotations

import random

import sympy as sp

from .field import eval_modp, DegeneratePoint
from .kinematics import sample_onshell_modp, vals_from_point

# The Mersenne prime 2^31-1 and the next prime below it. Nonzero reduction is an assumption.
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
    """The text the engine can parse, when producing it is free, from a string, an engine
    handle, whose `to_str` runs in Rust, or a sympy number. A general sympy object returns
    None, since printing one is slower than the sympy path itself, which keeps it."""
    if isinstance(x, str):
        return x
    if isinstance(x, sp.Number):
        return str(x)
    if not isinstance(x, sp.Basic) and hasattr(x, "to_str"):
        return x.to_str()
    return None


def _engine_asts(A, B, params=()):
    """Engine syntax trees for both sides, or None when the engine path does not apply.
    `params` lists the identifiers admitted as named parameters."""
    sa, sb_ = _engine_source(A), _engine_source(B)
    if sa is None or sb_ is None:
        return None
    try:
        import ampzip_engine as E
        if not hasattr(E, "parse_ast"):
            return None
        names = list(params)
        if names:                      # older engines take no params argument
            return E.parse_ast(sa, names), E.parse_ast(sb_, names)
        return E.parse_ast(sa), E.parse_ast(sb_)
    except Exception:  # noqa: BLE001  (engine absent, or a string outside its grammar)
        return None


def _is_prime(n: int) -> bool:
    """Miller-Rabin with bases that make it exact below 3.3e24."""
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes_for(needs, count=len(DEFAULT_PRIMES), below=2 ** 31):
    """The largest `count` primes below `below` that suit `needs` from `Ast.needs()`, meaning
    p = 1 (mod 4) when I occurs and every prime under a square root a quadratic residue mod p.
    Without such needs this is DEFAULT_PRIMES."""
    imag, roots, _names = needs
    if not imag and not roots:
        return DEFAULT_PRIMES
    out = []
    p = below - 1
    while len(out) < count:
        if _is_prime(p) and (not imag or p % 4 == 1) and \
                all(pow(q % p, (p - 1) // 2, p) == 1 for q in roots):
            out.append(p)
        p -= 2
    return tuple(out)


def _merge_needs(astA, astB):
    if not hasattr(astA, "needs"):      # older engine: integer coefficients only
        return (False, [], [])
    ia, ra, na = astA.needs()
    ib, rb, nb = astB.needs()
    return (ia or ib, sorted(set(ra) | set(rb)), sorted(set(na) | set(nb)))


def _equal_engine(astA, astB, n, k, primes, sampler, rng, max_resample, names=()) -> bool:
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
            # One random value per parameter and point, used by both sides.
            extra = ({name: rng.randrange(1, p) for name in names},) if names else ()
            nA, dA = astA.eval_pair(pt["lam"], pt["lamt"], p, *extra)
            nB, dB = astB.eval_pair(pt["lam"], pt["lamt"], p, *extra)
            if dA == 0 or dB == 0:
                continue
            if (nA * dB) % p != (nB * dA) % p:
                return False
            got += 1
    return True


def compare(A, B, n, k=DEFAULT_K, primes=DEFAULT_PRIMES, sampler=sample_onshell_modp,
            rng=None, max_resample=64, params=()):
    """`equal` on the engine path, returning `(verdict, info)`: True, False, or None when
    sampling was exhausted. `info` records the primes, the point count and the needs (I,
    square roots, parameters). Both sides must be text the engine parses. With the default
    `primes`, `primes_for` chooses primes that suit the needs."""
    asts = _engine_asts(A, B, params)
    if asts is None:
        raise ValueError("compare needs two expressions in the engine's grammar")
    if rng is None:
        rng = random.Random()
    needs = _merge_needs(*asts)
    if primes is DEFAULT_PRIMES:
        primes = primes_for(needs, len(DEFAULT_PRIMES))
    info = {"k": k, "primes": list(primes), "has_I": needs[0], "sqrt_primes": needs[1],
            "parameters": needs[2]}
    try:
        return _equal_engine(asts[0], asts[1], n, k, primes, sampler, rng, max_resample, needs[2]), info
    except RuntimeError:
        return None, info


def equal(A, B, n, k=DEFAULT_K, primes=DEFAULT_PRIMES,
          sampler=sample_onshell_modp, rng=None, max_resample=64, engine=True, params=()) -> bool:
    """Check agreement at random on-shell points for n legs using exact arithmetic.

    True means every sampled comparison passed, and False means a modular mismatch was
    found. Agreement is probabilistic evidence. RuntimeError on resampling
    exhaustion means inconclusive. Unsupported inputs and evaluation failures raise
    their own errors.

    `engine=False` forces the sympy evaluation path (the differential-test reference).
    `params` lists identifiers the engine path admits as named parameters, drawn at random per
    point for both sides. On that path `I`, `sqrt(n)` and decimals are exact, with primes from
    `primes_for` when the defaults do not suit them.
    """
    if rng is None:
        rng = random.Random()
    if engine:
        asts = _engine_asts(A, B, params)
        if asts is not None:
            needs = _merge_needs(*asts)
            if primes is DEFAULT_PRIMES:
                primes = primes_for(needs, len(DEFAULT_PRIMES))
            return _equal_engine(asts[0], asts[1], n, k, primes, sampler, rng, max_resample, needs[2])
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
                dA = eval_modp(denA, vals, p)
                dB = eval_modp(denB, vals, p)
                if dA == 0 or dB == 0:
                    continue
                lhs = (eval_modp(numA, vals, p) * dB) % p
                rhs = (eval_modp(numB, vals, p) * dA) % p
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
