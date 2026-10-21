"""Exact evaluation of a sympy expression over a prime field F_p.

A pure modular tree-walk. No floats, no ``sympy.N``. The evaluator is *generic
over the atom set*: every leaf that is a ``sympy.Function`` (e.g. ``ab(1,2)``,
``sb(2,3)`` today; ``ap``/``sp``/``ms``/``xf`` for massive later) is looked up in
a ``vals`` dict that maps the atom object -> its int value in [0, p). Adding new
atoms therefore needs only a new kinematics sampler, never a change here.

Supported nodes: Integer, Rational, Add, Mul, Pow (integer exponent, incl.
negative -> modular inverse), and Function atoms. Any other node raises, so a
silent mis-evaluation is impossible.
"""
from __future__ import annotations

import sympy as sp


class DegeneratePoint(Exception):
    """A denominator (or the base of a negative power) is 0 mod p at this point.

    The caller should discard this kinematic point and resample. Note: the
    verifier's default ``equal`` path cross-multiplies and never divides, so this
    is only hit by ``eval_modp`` on expressions that still contain negative
    powers (e.g. raw fractions passed directly, or ``zero`` on a numerator that
    itself carries a pole)."""


def modinv(a: int, p: int) -> int:
    """Modular inverse via Fermat's little theorem (p must be prime)."""
    a %= p
    if a == 0:
        raise DegeneratePoint("inverse of 0 mod p")
    return pow(a, p - 2, p)


def eval_modp(expr, vals: dict, p: int) -> int:
    """Evaluate ``expr`` over F_p given ``vals`` mapping Function atoms -> int.

    Raises ``DegeneratePoint`` if a denominator vanishes, ``KeyError`` if an atom
    is missing from ``vals``, and ``ValueError``/``AssertionError`` on an
    unsupported node (non-integer exponent, Float, Symbol, ...)."""
    # Integer first (is_Rational is also True for integers).
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
                return 0  # short-circuit
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
