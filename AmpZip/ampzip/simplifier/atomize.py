"""The atomizer, which turns rich amplitude notation into atomic ab/sb.

It is a string-to-string transformation applied before `validate()`. The output uses
`ab(i,j)` for <i j>, `sb(i,j)` for [i j] and `opaque_den(S)` for supported denominator
sums, which keeps the s() and chain() expansion rules out of validation and search.

Besides native ab/sb, the input may use

    s(i, j, ...)                      the Mandelstam invariant s_{ij...} = (p_i+...)^2
    chain(type, a, M_1, ..., M_k, b)  a Dirac chain with type one of 'aa', 'as', 'sa' and
                                      'ss', integer endpoints a and b, and each M_m a
                                      channel, a list of integer leg labels such as [1,6]
                                      for p1+p6.

The notation follows the textbook convention of Elvang and Huang and of Dixon, `ab(i,j) =
<ij>` and `sb(i,j) = [ij]` with matching argument order. Mandelstam invariants and Dirac
chains are

    s_ij = <ij>[ji] = ab(i,j) * sb(j,i)
    chain('as',a,P,b) = <a|P|b] = CHAIN_SIGN * sum_{i in P} ab(a,i)*sb(i,b)

The square-bracket arguments of s_ij are reversed, so it canonicalises to
-ab(i,j)*sb(i,j). With `sb(i,j) = [ij]`, the Dirac chain's adjacent contraction
ab(a,i)*sb(i,b) is the textbook <ai>[ib], so `CHAIN_SIGN = +1` follows the textbook and
chains need no sign. The index reversal of the textbook Mandelstam lives in `s()` alone.

With `allow_opaque_den=True`, denominator sums with positive integer powers are
retained as `opaque_den(S)` factors. Other powers of denominator sums are rejected.
Setting `allow_opaque_den=False` rejects denominator sums. Two-particle `s(i,j)` and
single-momentum chains expand to products of brackets and remain allowed in
denominators in either mode.

The Rust engine keeps polynomial denominator factors separate so it can cancel
a factor after a rewrite makes the numerator divisible by it.
"""
from __future__ import annotations

import ast
import itertools
from typing import Optional, Set

import sympy as sp

from .parse import parse_tree, eval_tree, ParseError

# The chain sign convention. With sb(i,j)=[ij], the adjacent contraction ab(a,i)*sb(i,b)
# equals the textbook <a|p_i|b] = <ai>[ib], so +1 follows the textbook. The Mandelstam index
# reversal is carried by s() alone.
CHAIN_SIGN = 1

_ALLOWED_CALLS = frozenset({"s", "chain", "ab", "sb"})
_ALLOWED_NAMES = frozenset({"s", "chain", "ab", "sb", "I"})
_CHAIN_TYPES = frozenset({"aa", "as", "sa", "ss"})


class AtomizeError(ValueError):
    """Raised when the input notation is malformed or violates a constraint, such as
    a denominator sum when `allow_opaque_den=False`."""


def _get_bracket_classes():
    from ..atoms import bracket_classes  # noqa: WPS433
    return bracket_classes()


# The AST whitelist and the integer wrapper live in parse.py, shared with the input gate.


def _infer_n(tree: ast.AST) -> Optional[int]:
    """The largest integer leg label among the arguments of s, chain, ab and sb, which the
    parser records. Pow exponents and the chain type string are not call arguments."""
    legs = getattr(tree, "call_ints", set())
    return max(legs) if legs else None


# --------------------------------------------------------------------------- #
# The notation callables, closures over n and the bracket classes
# --------------------------------------------------------------------------- #
def _make_mandelstam(ab, sb, n: Optional[int]):
    # s_I = (sum_{i in I} p_i)^2 = sum_{i<j in I} <ij>[ji] is an algebraic identity, independent
    # of n. The n-dependent on-shell relations, s_I = 0 for the full set and s_I = s_complement
    # by momentum conservation, are left to the verifier and the backends. The atomizer is a
    # faithful translator, and baking in momentum conservation would surprise the user, for
    # example with s(1,2) -> 0 when n is inferred as 2. Only s of at most one leg is 0, which is
    # masslessness, p_i^2 = 0, and holds for any n.
    #
    # In the textbook convention of Elvang and Huang and of Dixon, s_ij = <ij>[ji], the square
    # bracket's indices reversed with respect to the angle bracket's. Since sb(i,j) = [ij] here,
    # [ji] = sb(j,i), so each pair is ab(i,j)*sb(j,i), which is -ab(i,j)*sb(i,j) after
    # canonicalization. ab(i,j)*sb(i,j) would be <ij>[ij] = -s_ij, the opposite sign.
    # reorder_expr() in atomize() then flips sb(j,i) to -sb(i,j).
    def _s(*legs):
        I = sorted({int(x) for x in legs})
        m = len(I)
        if m <= 1:
            return sp.Integer(0)
        return sp.Add(*[ab(I[a], I[b]) * sb(I[b], I[a])
                        for a in range(m) for b in range(a + 1, m)])
    return _s


def _make_chain(ab, sb):
    def _chain(typ, a, *rest):
        if not isinstance(typ, str) or typ not in _CHAIN_TYPES:
            raise AtomizeError(f"chain type must be one of {sorted(_CHAIN_TYPES)}, got {typ!r}")
        if len(rest) < 1:
            raise AtomizeError("chain needs at least the endpoints: chain(type, a, ..., b)")
        a = int(a)
        b = int(rest[-1])
        channels = [[int(x) for x in M] for M in rest[:-1]]
        k = len(channels)
        # The parity. The starting bracket is typ[0], it flips once per momentum, and the last must be typ[1].
        right = typ[0] if k % 2 == 0 else ("s" if typ[0] == "a" else "a")
        if right != typ[1]:
            need = "even" if typ[0] == typ[1] else "odd"
            raise AtomizeError(
                f"chain parity mismatch: type '{typ}' requires an {need} number of "
                f"middle momenta, got {k}")
        total = sp.Integer(0)
        for choice in itertools.product(*channels) if channels else [()]:
            pts = [a, *choice, b]
            term = sp.Integer(1)
            ok = True
            for mth in range(k + 1):
                x, y = pts[mth], pts[mth + 1]
                if x == y:            # ab(x,x) and sb(x,x) are 0
                    ok = False
                    break
                angle = (typ[0] == "a") == (mth % 2 == 0)
                term *= ab(x, y) if angle else sb(x, y)
            if ok:
                total += term
        return CHAIN_SIGN * total
    return _chain


# --------------------------------------------------------------------------- #
# The denominator must be a monomial
# --------------------------------------------------------------------------- #
def _reject_sum_denominator(expr: sp.Expr) -> None:
    den = sp.together(expr).as_numer_denom()[1]
    for fac in sp.Mul.make_args(den):
        base = fac.base if fac.is_Pow else fac
        if base.is_Add:
            raise AtomizeError(
                "denominator sums are rejected when allow_opaque_den=False: "
                f"1/({base}). Two-particle s(i,j) and single-momentum chains are "
                "allowed in denominators. Set allow_opaque_den=True to retain "
                "the sum as opaque_den(S) for the GBFS backends.")


def _rewrite_with_opaque_den(expr: sp.Expr) -> str:
    """The allow_opaque_den path, which emits a string in which every Add factor in the
    denominator becomes an `opaque_den(S)` marker, the engine's symbolic 1/S kept unexpanded
    through the search, while monomial factors remain an ordinary product of brackets.

    The output is rebuilt from the numerator and denominator extracted by
    `sp.fraction(sp.together(expr))`, even when the denominator has no sum factors.
    """
    from ..atoms import canonical_order as reorder_expr  # noqa: WPS433
    num, den = sp.fraction(sp.together(expr))
    mono_factors = []
    opaque = []  # (S_expr, multiplicity) pairs
    for fac in sp.Mul.make_args(den):
        base, exp = (fac.base, fac.exp) if fac.is_Pow else (fac, sp.Integer(1))
        if base.is_Add:
            if not (exp.is_Integer and int(exp) >= 1):
                raise AtomizeError(
                    f"opaque denominator factor with unsupported power 1/({base})**{exp}")
            opaque.append((base, int(exp)))
        else:
            mono_factors.append(fac)
    # Rebuild from the together'd num/den. When sympy cancels the sum away
    # algebraically, as (S*c)/S -> c leaves `den` monomial and `opaque` empty, str(expr) would
    # still carry the uncancelled `1/(sum)` and the engine would reject it. Rebuilding emits
    # the cancelled monomial form, and the opaque_den(S) marker only when the sum survives,
    # which means it cancels only on shell and is left to the search.
    d_mono = sp.Mul(*mono_factors) if mono_factors else sp.Integer(1)
    s = f"({reorder_expr(num)})"
    if d_mono != sp.Integer(1):
        s = f"{s}/({reorder_expr(d_mono)})"
    for sfac, mult in opaque:
        s += f"*opaque_den({reorder_expr(sfac)})" * mult
    return s


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def has_opaque_den(atomized: str) -> bool:
    """True iff an atomized string carries a sum-denominator marker. Callers route these
    inputs to the gbfs family because the neural baselines cannot consume `opaque_den`."""
    return "opaque_den(" in atomized


def atomize(raw: str, n_hint: Optional[int] = None, allow_opaque_den: bool = True) -> str:
    """Expand `s()` and `chain()` notation into `ab`/`sb` notation.

    Returns an expression string with canonical bracket order for `validate()`.
    Malformed notation, singular inputs and unsupported denominator factors raise
    `AtomizeError`.

    With `allow_opaque_den=True`, the default, denominator sums with positive integer powers
    are represented by `opaque_den(S)` factors for the gbfs family. Other powers of
    denominator sums are rejected. `allow_opaque_den=False` rejects denominator sums.
    """
    if not raw or not raw.strip():
        raise AtomizeError("empty expression")
    # Normalise line breaks to spaces. A newline is a token separator for Python's tokenizer,
    # so `A + \n B` outside brackets is a SyntaxError. Users often paste expressions wrapped
    # across lines. A space replaces the newline, since "" would fuse adjacent tokens, and
    # `12\n34` becomes `12 34`.
    try:
        tree = parse_tree(raw, calls=_ALLOWED_CALLS, names=_ALLOWED_NAMES, allow_str=True)
    except ParseError as e:
        raise AtomizeError(str(e)) from e

    n = n_hint if n_hint is not None else _infer_n(tree)
    ab, sb = _get_bracket_classes()
    ns = {"s": _make_mandelstam(ab, sb, n), "chain": _make_chain(ab, sb),
          "ab": ab, "sb": sb, "I": sp.I}
    try:
        expr = eval_tree(tree, ns)
    except AtomizeError:
        raise
    except ParseError as e:
        # eval_tree wraps the notation callables' own AtomizeErrors. Those are surfaced verbatim.
        cause = e.__cause__
        raise (cause if isinstance(cause, AtomizeError) else AtomizeError(str(e))) from e
    if expr.has(sp.zoo, sp.oo, sp.nan, -sp.oo):
        raise AtomizeError("expression is singular or undefined")
    from ..atoms import canonical_order as reorder_expr  # noqa: WPS433
    expr = reorder_expr(expr)
    if allow_opaque_den:
        return _rewrite_with_opaque_den(expr)
    _reject_sum_denominator(expr)
    return str(expr)
