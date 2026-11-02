"""Exact evaluation of a sympy expression over a prime field F_p.

A pure modular tree walk, with no floats and no ``sympy.N``. The evaluator is generic over
the atom set. Every leaf that is a ``sympy.Function``, ``ab(1,2)`` and ``sb(2,3)`` today
and ``ap``, ``sp``, ``ms`` and ``xf`` for massive kinematics later, is looked up in a
``vals`` dict that maps the atom object to its int value in [0, p). A new atom therefore
needs only a new kinematics sampler.

The supported nodes are Integer, Rational, Add, Mul, Pow with an integer exponent, a
negative one meaning a modular inverse, and Function atoms. Any other node raises, so a
silent mis-evaluation is impossible.
"""
from __future__ import annotations

import sympy as sp


class DegeneratePoint(Exception):
    """A denominator, or the base of a negative power, is 0 mod p at this point.

    The caller discards this kinematic point and resamples. The verifier's default
    ``equal`` path cross-multiplies and never divides, so only ``eval_modp`` on expressions
    that still contain negative powers raises this, raw fractions passed directly or
    ``zero`` on a numerator that itself carries a pole."""


def modinv(a: int, p: int) -> int:
    """The modular inverse by Fermat's little theorem, for prime p."""
    a %= p
    if a == 0:
        raise DegeneratePoint("inverse of 0 mod p")
    return pow(a, p - 2, p)


def eval_modp(expr, vals: dict, p: int) -> int:
    """Evaluate ``expr`` over F_p given ``vals`` mapping Function atoms -> int.

    Raises ``DegeneratePoint`` if a denominator vanishes, ``KeyError`` if an atom is
    missing from ``vals``, and ``ValueError`` or ``AssertionError`` on an unsupported node
    such as a non-integer exponent, a Float or a Symbol."""
    # Integer comes first, since is_Rational is also True for integers.
    if expr.is_Integer:
        return int(expr) % p
    if expr.is_Rational:
        return (int(expr.p) * modinv(int(expr.q), p)) % p
    if expr.is_Add:
        s = 0
        for t in expr.args:
            s = (s + eval_modp(t, vals, p)) % p
        return s
    if expr.is_Mul:
        s = 1
        for t in expr.args:
            s = (s * eval_modp(t, vals, p)) % p
            if s == 0:
                return 0
        return s
    if expr.is_Pow:
        base, e = expr.args
        if not e.is_Integer:
            raise AssertionError(f"non-integer exponent unsupported: {expr}")
        bb = eval_modp(base, vals, p)
        e = int(e)
        if e < 0:
            return pow(modinv(bb, p), -e, p)
        return pow(bb, e, p)
    if isinstance(expr, sp.Function):
        v = vals.get(expr)
        if v is None:
            raise KeyError(f"atom {expr} not in vals (sampler missing this atom?)")
        return v % p
    raise ValueError(f"eval_modp: unhandled node {type(expr).__name__}: {expr}")
