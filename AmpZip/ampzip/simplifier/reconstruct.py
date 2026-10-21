"""Reconstructor — fold atomic ab/sb backend output back into readable notation.

Display-only inverse of the Atomizer: a pure function `reconstruct(expr_or_str)`
that rewrites a value-equal string using `s(...)` (Mandelstam) and `chain(...)`
(Dirac) notation. Never re-runs a backend.

Matching priority (user directive): **s first, plain ab/sb second, Dirac last**.

Two folding mechanisms, applied at different tree positions:

  1. *Loose-pool fold* (`_fold`): within a single product term, the multiset of bare
     ab/sb factors is grouped — `ab(i,j)*sb(i,j)` -> `s(i,j)`, a shared-index
     `ab(a,m)*sb(m,b)` -> `chain('as',a,[m],b)`, leftovers stay bare. This is the
     two-particle / single-momentum case and was the whole of v1. Under the textbook
     convention s(i,j)=<ij>[ji]=ab(i,j)*sb(j,i), a canonical bare pair ab(i,j)*sb(i,j)
     equals -s(i,j), so each two-particle s fold flips the term's coefficient sign
     (`_render_term` compensates); chains (CHAIN_SIGN=+1) carry no such sign.

  2. *Sub-expression fold* (`_fold_add`): an
     intact `Add` sitting in *factor position* (the base of a power `(...)**k`, a
     summed numerator factor inside a product, or the argument of an `opaque_den(S)`
     denominator marker) is recognized as a *multi-particle* `s(i,j,k,...)` or a
     *single-channel* Dirac chain `chain('as',a,[m1,m2,...],b)` and folded whole.

Recognition is by **candidate-generation + symbolic equality** against the atomizer's
own `_make_mandelstam` / `_make_chain` expansions (so the `CHAIN_SIGN` convention and
leg ordering stay byte-identical to atomization, and every fold is value-preserving by
construction — it re-atomizes to exactly the sub-expression it replaced, sign included).

Scope boundary (Scope A). `_fold_add` runs ONLY on Add nodes in factor position and on
a whole top-level Add that is itself a single s/chain. It is deliberately NOT run on the
individual terms of a top-level sum — a bare sum the user typed is rendered term-wise
(each term still loose-pool-folded), never force-collapsed. Recognizing a multi-particle
s / multi-leg chain hidden inside a fully *expanded* flat sum (what a gbfs rollout emits)
is the non-unique inverse-of-expansion problem and remains out of scope; such output
renders term-wise with per-term loose-pool folding, the best available without regrouping.
Multi-*channel* chains (`<a|P Q|b>`, k>=2 momenta) are also out of scope for v2.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import sympy as sp


_Factor = Tuple[str, int, int]  # ('ab'|'sb', i, j)

# Bilinear-Add fold guards: skip candidate-generation on Adds too large to be a single
# s/chain (a k-leg Mandelstam has C(k,2) terms; C(8,2)=28), keeping cost bounded on the
# large flat sums a gbfs rollout can emit.
_MAX_FOLD_TERMS = 30
_MAX_FOLD_LEGS = 8


def _get_bracket_classes():
    from ..atoms import bracket_classes  # noqa: WPS433
    return bracket_classes()


def _make_ctx():
    """Build the per-call context: bracket classes + atomizer expansion callables
    (reused verbatim so folds match atomization's sign/ordering exactly)."""
    from ..atoms import canonical_order as reorder_expr  # noqa: WPS433

    from .atomize import _make_chain, _make_mandelstam  # noqa: WPS433
    ab_cls, sb_cls = _get_bracket_classes()
    return {
        "ab": ab_cls,
        "sb": sb_cls,
        "mand": _make_mandelstam(ab_cls, sb_cls, None),
        "chain": _make_chain(ab_cls, sb_cls),
        "reorder": reorder_expr,
    }


def _is_opaque(base) -> bool:
    return base.is_Function and getattr(base.func, "__name__", "") == "opaque_den"


# --------------------------------------------------------------------------- #
# Sub-expression fold: recognize a whole Add as s(...) or a single-channel chain
# --------------------------------------------------------------------------- #
def _bilinear_legs(add_expr: sp.Expr, ab_cls, sb_cls) -> Optional[List[int]]:
    """If every term of `add_expr` is (coeff)*ab(i,j)*sb(k,l) — exactly one angle and
    one square bracket, each to the first power, nothing else — return the sorted leg
    set. Otherwise None. This cheaply rejects anything a multi-particle s / single-channel
    chain can never be (higher powers, products of >2 brackets, foreign factors) BEFORE
    any expensive expand()."""
    legs: set = set()
    for term in add_expr.args:
        _coeff, rest = term.as_coeff_Mul()
        n_ab = n_sb = 0
        for fac in sp.Mul.make_args(rest):
            if fac.is_Pow:
                return None  # a squared bracket is never part of a bilinear sum
            if isinstance(fac, ab_cls):
                n_ab += 1
                legs |= {int(fac.args[0]), int(fac.args[1])}
            elif isinstance(fac, sb_cls):
                n_sb += 1
                legs |= {int(fac.args[0]), int(fac.args[1])}
            else:
                return None  # foreign factor (number>coeff, opaque_den, ...) — not bilinear
        if n_ab != 1 or n_sb != 1:
            return None
    return sorted(legs)


def _fold_add(add_expr: sp.Expr, ctx: Dict) -> Optional[Tuple[str, int]]:
    """Try to recognize an Add as a multi-particle Mandelstam or single-channel Dirac
    chain. Returns (token, sign) with sign in {+1,-1} such that sign*token_value ==
    add_expr, or None. `s(...)` is tried first (priority), then chains."""
    # Peel a +-1 scalar wrapping the sum: factor_terms can leave a hoisted pole as
    # `opaque_den(-(a+b))` i.e. Mul(-1, Add), which is not `.is_Add`. Fold on the inner
    # Add and carry the sign, so it collapses to a clean token (no redundant parens).
    outer_sign = 1
    if add_expr.is_Mul:
        c, r = add_expr.as_coeff_Mul()
        if r.is_Add and c in (sp.Integer(1), sp.Integer(-1)):
            outer_sign = int(c)
            add_expr = r
    if not add_expr.is_Add or len(add_expr.args) > _MAX_FOLD_TERMS:
        return None
    ab_cls, sb_cls = ctx["ab"], ctx["sb"]
    legs = _bilinear_legs(add_expr, ab_cls, sb_cls)
    if legs is None or not (2 <= len(legs) <= _MAX_FOLD_LEGS):
        return None

    # Bracket classes do NOT apply antisymmetry under sp.expand (ab(2,1) stays ab(2,1)),
    # so `_make_chain` candidates carry non-canonical brackets. Compare via the atomizer's
    # `reorder_expr` canonicalizer, which folds ab(j,i) -> -ab(i,j) etc.
    reorder = ctx["reorder"]

    def _matches(sgn: int, cand) -> bool:
        return sp.expand(reorder(add_expr - sgn * cand)) == 0

    # --- s(i,j,...) : the full leg set, C(m,2) terms ---
    cand = ctx["mand"](*legs)
    for sgn in (1, -1):
        if _matches(sgn, cand):
            return f"s({', '.join(map(str, legs))})", sgn * outer_sign

    # --- single-channel chain('as'|'sa', a, [P], b) : endpoints a,b; middle P=rest ---
    # Prefer 'as' (angle-first, the common textbook <a|P|b] spelling) over the value-equal
    # 'sa' form, so folded chains match how users typically write them.
    for typ in ("as", "sa"):
        for a in legs:
            for b in legs:
                if a == b:
                    continue
                P = [x for x in legs if x != a and x != b]
                if not P:
                    continue
                cand = ctx["chain"](typ, a, P, b)
                for sgn in (1, -1):
                    if _matches(sgn, cand):
                        return f"chain('{typ}', {a}, {P}, {b})", sgn * outer_sign
    return None


# --------------------------------------------------------------------------- #
# Loose-pool fold: group the bare ab/sb multiset of one product term
# --------------------------------------------------------------------------- #
def _fold(factors: List[_Factor]) -> Tuple[List[Tuple[int, str]], int]:
    """Group a factor multiset into display tokens, returning (tokens, sign):
    each token is (priority, text) with priority 0 = s, 1 = plain ab/sb, 2 = chain
    (order within a priority preserved); `sign` in {+1,-1} is the antisymmetry sign the
    chain folds pull out (see PASS 2), to be multiplied onto the term coefficient by the
    caller. The two-particle s sign is NOT included here — the caller compensates it by
    counting priority-0 tokens (`n_s_folds`), as before.

    The chain test mirrors `objective_atoms.fold_counts` (symmetric "share exactly one
    leg") so the atom score and this display fold agree on which brackets form a chain."""
    abs_ = [(i, j) for (k, i, j) in factors if k == "ab"]
    sbs = [(i, j) for (k, i, j) in factors if k == "sb"]
    used_a = [False] * len(abs_)
    used_s = [False] * len(sbs)
    tokens: List[Tuple[int, str]] = []
    sign = 1

    # PASS 1 — Mandelstam: ab(i,j) * sb(i,j) with identical args
    for ia, (i, j) in enumerate(abs_):
        for js, (k, l) in enumerate(sbs):
            if not used_a[ia] and not used_s[js] and (i, j) == (k, l):
                used_a[ia] = used_s[js] = True
                tokens.append((0, f"s({i}, {j})"))
                break

    # PASS 2 — single-momentum Dirac chains: ab and sb sharing EXACTLY one leg.
    # Symmetric leg test (matches objective_atoms.fold_counts, so the atom score and this
    # display agree on which pairs are chains). Written angle-first: a chain reads
    # <a|m|b] = chain('as', a, [m], b) = ab(a,m)*sb(m,b), with `m` the shared momentum and
    # a,b the non-shared endpoints. The bracket pair may carry `m` in either slot; bracket
    # antisymmetry (ab(j,i)=-ab(i,j), sb(j,i)=-sb(i,j)) then contributes a ±1 we fold into
    # `sign` so the rewrite stays value-exact (the old code only handled the sign-free
    # ab-2nd / sb-1st position and missed every other canonical placement).
    for ia, (ai, aj) in enumerate(abs_):
        if used_a[ia]:
            continue
        for js, (si, sj) in enumerate(sbs):
            if used_s[js]:
                continue
            if len({ai, aj} & {si, sj}) != 1:
                continue
            m = next(iter({ai, aj} & {si, sj}))
            a = aj if ai == m else ai          # non-shared endpoint of the angle bracket
            b = si if sj == m else sj          # non-shared endpoint of the square bracket
            # ab(ai,aj) = s_ab * ab(a,m):  +1 iff m is already the 2nd slot, else swap → -1
            s_ab = 1 if aj == m else -1
            # sb(si,sj) = s_sb * sb(m,b):  +1 iff m is already the 1st slot, else swap → -1
            s_sb = 1 if si == m else -1
            sign *= s_ab * s_sb
            used_a[ia] = used_s[js] = True
            tokens.append((2, f"chain('as', {a}, [{m}], {b})"))
            break

    # PASS 3 — leftovers stay as plain brackets
    for ia, (i, j) in enumerate(abs_):
        if not used_a[ia]:
            tokens.append((1, f"ab({i}, {j})"))
    for js, (i, j) in enumerate(sbs):
        if not used_s[js]:
            tokens.append((1, f"sb({i}, {j})"))

    return tokens, sign


def _collapse_pool(tokens: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Collapse repeated identical tokens into `tok**k`, keeping (priority, text)."""
    counts: dict = {}
    order: List[Tuple[int, str]] = []
    for pr, text in tokens:
        if (pr, text) not in counts:
            counts[(pr, text)] = 0
            order.append((pr, text))
        counts[(pr, text)] += 1
    out: List[Tuple[int, str]] = []
    for key in order:
        k = counts[key]
        out.append((key[0], key[1] if k == 1 else f"{key[1]}**{k}"))
    return out


# --------------------------------------------------------------------------- #
# Recursive renderer
# --------------------------------------------------------------------------- #
def _pow_text(text: str, mult: int) -> str:
    return text if mult == 1 else f"{text}**{mult}"


def _render_term(term: sp.Expr, ctx: Dict) -> str:
    """Render one product term: coeff * (num factors) / (den factors), folding brackets
    (loose-pool) and Add/opaque_den sub-expressions (`_fold_add`)."""
    ab_cls, sb_cls = ctx["ab"], ctx["sb"]
    coeff, rest = term.as_coeff_Mul()

    num_pool: List[_Factor] = []
    den_pool: List[_Factor] = []
    num_groups: List[Tuple[int, str]] = []  # (priority, text) — s(0), chain(2), other(3)
    den_groups: List[Tuple[int, str]] = []
    # Prefactor mode (ctx["prefactor"], gbfs_f only): if this term is a sum-of-fractions numerator
    # over a common denominator A, remember the rendered numerator so a DEN factor can be hoisted
    # to a leading `-1/A*(...)` prefactor instead of the two-story `(...)/A`. `num_frac_sign`
    # carries the overall sign we pulled out of the inner sum (so the inner renders all-`+`).
    num_frac_sum: Optional[str] = None
    num_frac_sign: int = 1

    for fac in sp.Mul.make_args(rest):
        base, exp = (fac.base, fac.exp) if fac.is_Pow else (fac, sp.Integer(1))
        if not exp.is_Integer:
            # non-integer power (rare post-atomize): keep raw, undissected
            num_groups.append((3, f"({base})**{exp}"))
            continue
        e = int(exp)
        to_den = e < 0
        mult = abs(e)
        groups = den_groups if to_den else num_groups
        pool = den_pool if to_den else num_pool

        if isinstance(base, (ab_cls, sb_cls)):
            kind = "ab" if isinstance(base, ab_cls) else "sb"
            i, j = int(base.args[0]), int(base.args[1])
            for _ in range(mult):
                pool.append((kind, i, j))
        elif _is_opaque(base):
            # opaque_den(S) is a symbolic 1/S pole -> always denominator
            folded = _fold_add(base.args[0], ctx)
            if folded is not None:
                tok, sgn = folded
                coeff *= sgn ** mult
                pr = 0 if tok.startswith("s(") else 2
                den_groups.append((pr, _pow_text(tok, mult)))
            else:
                den_groups.append((3, _pow_text(f"({_render(base.args[0], ctx)})", mult)))
        elif base.is_Add:
            folded = _fold_add(base, ctx)
            if folded is not None:
                tok, sgn = folded
                coeff *= sgn ** mult
                pr = 0 if tok.startswith("s(") else 2
                groups.append((pr, _pow_text(tok, mult)))
            else:
                # Render the sum term-wise so we can detect an all-negative sum at the DISPLAY
                # level. The per-term minus is a RENDERING artifact of the fold sign conventions
                # (s(i,j) = -ab(i,j)*sb(i,j), chain antisymmetry), NOT the sympy coefficient — so
                # base.args' sympy signs are the wrong signal (measured on eq 5.28: the Add and its
                # args all report could_extract_minus_sign()==False, yet every term renders `-`).
                term_pieces = ([_render_term(t, ctx) for t in sp.Add.make_args(base)]
                               if base.is_Add else [_render(base, ctx)])
                frac_sign = 1
                if (ctx.get("prefactor") and not to_den and mult == 1
                        and len(term_pieces) > 1
                        and all(p.startswith("-") for p in term_pieces)):
                    # In prefactor mode, pull the overall minus out of an all-negative sum so the
                    # inner renders all-`+` and the sign rides on the leading -1/A prefactor
                    # (-1/A*(B/C + D/E)). Only fires when EVERY term is negative — unambiguous,
                    # never introduces a spurious leading minus on a mixed-sign sum.
                    stripped = [p[1:] for p in term_pieces]
                    inner = " + ".join(stripped)
                    frac_sign = -1
                else:
                    # Normal join — byte-identical to `_render`'s Add branch (which we've already
                    # tried to fold above), so non-prefactor output is unchanged.
                    inner = term_pieces[0]
                    for p in term_pieces[1:]:
                        inner += (" - " + p[1:]) if p.startswith("-") else (" + " + p)
                text = _pow_text(f"({inner})", mult)
                groups.append((3, text))
                # Record only a genuine sum-of-FRACTIONS numerator (an inner `/`), so a plain
                # `(a+b)` numerator is untouched and H3 can't misfire on it.
                if ctx.get("prefactor") and not to_den and mult == 1 and "/" in inner:
                    num_frac_sum, num_frac_sign = text, frac_sign
        else:
            # a bare number or unrecognized atom — absorb into the coefficient (value-safe)
            coeff = coeff * fac

    num_folded, num_chain_sign = _fold(num_pool)
    den_folded, den_chain_sign = _fold(den_pool)
    # Chain folds may pull a bracket-antisymmetry sign (PASS 2 in `_fold`). Fold it onto
    # the coefficient — for the denominator too, since sign in {+1,-1} gives 1/(s*T)=s/T.
    coeff *= num_chain_sign * den_chain_sign
    # Textbook convention s(i,j)=<ij>[ji]=ab(i,j)*sb(j,i): a bare canonical pair
    # ab(i,j)*sb(i,j) is <ij>[ij] = -s(i,j), so every two-particle s fold (priority 0
    # in the loose pool) flips the sign. Compensate on the coefficient — for both num
    # and den, since 1/(ab*sb) = -1/s. (Multi-particle s / chains handled via _fold_add
    # already carry their own sign in `coeff` above.)
    n_s_folds = (sum(1 for pr, _ in num_folded if pr == 0)
                 + sum(1 for pr, _ in den_folded if pr == 0))
    if n_s_folds % 2:
        coeff = -coeff
    num_tokens = _collapse_pool(num_folded) + num_groups
    den_tokens = _collapse_pool(den_folded) + den_groups
    num_tokens.sort(key=lambda t: t[0])
    den_tokens.sort(key=lambda t: t[0])
    num_texts = [t[1] for t in num_tokens]
    den_texts = [t[1] for t in den_tokens]

    sign = "-" if coeff.is_number and coeff < 0 else ""
    mag = -coeff if sign else coeff

    num_parts: List[str] = []
    if mag != 1 or not num_texts:
        num_parts.append(str(mag))
    num_parts.extend(num_texts)
    body = "*".join(num_parts) if num_parts else "1"

    if den_texts:
        den_body = "*".join(den_texts)
        den_body = den_body if len(den_texts) == 1 else f"({den_body})"
        if num_frac_sum is not None and body == num_frac_sum:
            # A pole common to every term of the sum, hoisted out front:
            # "-1/A*(B/C + D/E)" instead of the two-story "(-B/C - D/E)/A".
            body = f"1/{den_body}*{body}"
            if num_frac_sign < 0:
                sign = "" if sign == "-" else "-"   # the pulled-out minus flips the term sign
        else:
            body = f"{body}/{den_body}"
    return sign + body


def _render(expr: sp.Expr, ctx: Dict) -> str:
    """Render an expression. A top-level Add is first tried as a single foldable
    s/chain (so a standalone `s(1,2,3)` folds whole); otherwise it is rendered
    term-wise, each term loose-pool-folded (bare user sums are never force-collapsed)."""
    if expr == 0:
        return "0"
    if expr.is_Add:
        whole = _fold_add(expr, ctx)
        if whole is not None:
            tok, sgn = whole
            return tok if sgn > 0 else f"-{tok}"
        terms = sp.Add.make_args(expr)
        pieces = [_render_term(t, ctx) for t in terms]
        out = pieces[0]
        for p in pieces[1:]:
            out += (" - " + p[1:]) if p.startswith("-") else (" + " + p)
        return out
    return _render_term(expr, ctx)


def reconstruct(expr_or_str, prefactor: bool = False) -> str:
    """Fold an atomic ab/sb expression into s()/chain() display notation.

    Recognizes two-particle AND multi-particle `s(...)`, single-momentum AND
    single-channel `chain(...)`, and folds `opaque_den(S)` denominator markers back to
    their s/chain form. Display-only and value-preserving (re-atomizes identically).

    `prefactor` (default False, OPT-IN — only gbfs_f passes True): render a sum-of-fractions
    numerator over a common denominator A as a leading `-1/A*(B/C + D/E)` prefactor instead of the
    two-story `(-B/C - D/E)/A`. Default False keeps EVERY other caller byte-identical — the
    behavior is gated on this flag, never sniffed from the expression shape (a shape sniff cannot
    tell gbfs_f's output from gbfs_e's, which is the same sum-of-fractions class).
    """
    ctx = _make_ctx()
    ctx["prefactor"] = prefactor
    if isinstance(expr_or_str, str):
        from .parse import parse_expr  # noqa: WPS433
        expr = parse_expr(
            expr_or_str,
            {"ab": ctx["ab"], "sb": ctx["sb"], "I": sp.I,
             "opaque_den": sp.Function("opaque_den")},
            lenient=True,
        )
    else:
        expr = expr_or_str
    # Hoist factors common to every term of a sum out of the sum (value-preserving regroup).
    # A backend that distributes shared poles across a big sum — e.g. gbfs replicating the
    # three opaque_den(S) markers into all 24 numerator terms — would otherwise make every
    # term carry the same denominator (`.../(s126*s156*<2|1+6|5])`) with a *second* outer
    # denominator on top. factor_terms pulls the shared factors out once, so the whole
    # expression renders over a single combined denominator.
    expr = sp.factor_terms(expr)
    return _render(expr, ctx)
