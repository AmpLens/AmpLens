"""The bracket atoms and their canonical ordering.

`ab(i, j)` is the angle bracket ⟨i j⟩ and `sb(i, j)` the square bracket [i j], both as
uninterpreted sympy functions of two leg labels. Every expression the package handles is
built from these two atoms (plus `opaque_den`, the engine's marker for a polynomial
denominator factor, which is created where it is needed).

`canonical_order(expr)` rewrites every bracket to `i < j`, pushing the antisymmetry sign into
the coefficient: `ab(2, 1) -> -ab(1, 2)`. The engine, the atomizer and the display layer all
rely on this spelling, so a Mandelstam `s_ij = ⟨ij⟩[ji]` is stored as `-ab(i,j)*sb(i,j)`.

`ATOMS` is the registry the input gate consults: name -> class. A new atom kind is added by
registering it here together with its canonicalisation rule; nothing else needs to know.
"""
from __future__ import annotations

from typing import Dict, Type

import sympy as sp


def _leg_labels(printer, args):
    """Leg labels for display: an integer prints as itself; a symbol such as `p_{3}` prints
    as its bare label."""
    return [printer._print(a).replace("p_{", "").replace("}", "") for a in args]


class _Bracket(sp.Function):
    """A two-leg bracket; subclasses only choose the delimiters."""

    _delims = ("", "")

    def _latex(self, printer, exp=1):
        i, j = _leg_labels(printer, self.args)
        left, right = self._delims
        body = f"{left}{i} {j}{right}"
        return body if exp == 1 else f"{body}^{{{exp}}}"


class ab(_Bracket):
    """Angle bracket ⟨i j⟩ of legs i and j."""

    _delims = (r"\langle ", r" \rangle")


class sb(_Bracket):
    """Square bracket [i j] of legs i and j."""

    _delims = (r"\left[ ", r" \right]")


ATOMS: Dict[str, Type[sp.Function]] = {"ab": ab, "sb": sb}


def bracket_classes():
    """The two bracket classes, in the order every caller unpacks them: `ab, sb`."""
    return ab, sb


def canonical_order(expr: sp.Expr) -> sp.Expr:
    """Rewrite each two-leg bracket with its legs ascending, times -1 when they were swapped.

    Applies to every two-argument function atom in `expr`, which is what the bracket
    antisymmetry needs; one-argument atoms such as `opaque_den` are left alone.
    """
    replace = {}
    for fun in expr.atoms(sp.Function):
        if len(fun.args) == 2 and fun.args[0] > fun.args[1]:
            replace[fun] = fun.func(fun.args[1], fun.args[0]) * (-1)
    return expr.subs(replace)
