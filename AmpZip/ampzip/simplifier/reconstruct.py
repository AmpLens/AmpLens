"""Fold atomic ab/sb backend output back into readable notation.

This is the display-only inverse of the atomizer, a pure function `reconstruct(expr_or_str)`
that rewrites a value-equal string in `s(...)` and `chain(...)` notation. Matching tries s
first, plain ab/sb second and Dirac chains last.

Two folds apply at different tree positions. The loose-pool fold in `_fold` groups the
multiset of bare ab/sb factors within a single product term. `ab(i,j)*sb(i,j)` becomes
`s(i,j)`, a shared-index `ab(a,m)*sb(m,b)` becomes `chain('as',a,[m],b)`, and leftovers
stay bare. This covers the two-particle and single-momentum case. Under the textbook
convention s(i,j)=<ij>[ji]=ab(i,j)*sb(j,i), a canonical bare pair ab(i,j)*sb(i,j) equals
-s(i,j), so each two-particle s fold flips the term's coefficient sign, which
`_render_term` compensates. Chains carry no such sign, CHAIN_SIGN being +1.

The sub-expression fold in `_fold_add` takes an intact `Add` in factor position, the base
of a power `(...)**k`, a summed numerator factor inside a product or the argument of an
`opaque_den(S)` denominator marker, and recognises it as a multi-particle `s(i,j,k,...)`
or a single-channel Dirac chain `chain('as',a,[m1,m2,...],b)`, folding it whole.

Recognition generates candidates and tests symbolic equality against the atomizer's own
`_make_mandelstam` and `_make_chain` expansions, so the `CHAIN_SIGN` convention and the
leg ordering stay byte-identical to atomization and every fold is value-preserving by
construction. It re-atomizes to exactly the sub-expression it replaced, sign included.

`_fold_add` runs on Add nodes in factor position and on a whole top-level Add that is
itself a single s or chain. It does not run on the individual terms of a top-level sum. A
bare sum the user typed is rendered term-wise, each term still loose-pool-folded, and is
never collapsed. Recognising a multi-particle s or multi-leg chain hidden inside a fully
expanded flat sum, which a gbfs rollout emits, is the non-unique inverse of expansion and
remains out of scope. Such output renders term-wise with per-term loose-pool folding.
Multi-channel chains `<a|P Q|b>` with two or more momenta are out of scope too.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import sympy as sp


_Factor = Tuple[str, int, int]  # ('ab'|'sb', i, j)

# Guards for the bilinear-Add fold. Candidate generation is skipped on Adds too large to be
# a single s or chain, a k-leg Mandelstam having C(k,2) terms and C(8,2) being 28, which
# keeps the cost bounded on the large flat sums a gbfs rollout can emit.
_MAX_FOLD_TERMS = 30
_MAX_FOLD_LEGS = 8


def _get_bracket_classes():
    from ..atoms import bracket_classes  # noqa: WPS433
    return bracket_classes()


def _make_ctx():
    """Build the per-call context, the bracket classes and the atomizer's expansion
    callables, reused verbatim so folds match atomization's sign and ordering exactly."""
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
# Sub-expression fold, which recognises a whole Add as s(...) or a single-channel chain
# --------------------------------------------------------------------------- #
def _bilinear_legs(add_expr: sp.Expr, ab_cls, sb_cls) -> Optional[List[int]]:
    """If every term of `add_expr` is (coeff)*ab(i,j)*sb(k,l), exactly one angle and one
    square bracket, each to the first power and nothing else, return the sorted leg set.
    Otherwise None. This cheaply rejects anything a multi-particle s or single-channel
    chain can never be, higher powers, products of more than two brackets or foreign
    factors, before any expensive expand()."""
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
                return None  # a foreign factor such as a number or opaque_den, so the term is not bilinear
        if n_ab != 1 or n_sb != 1:
            return None
    return sorted(legs)


def _fold_add(add_expr: sp.Expr, ctx: Dict) -> Optional[Tuple[str, int]]:
    """Try to recognize an Add as a multi-particle Mandelstam or single-channel Dirac
    chain. Returns (token, sign) with sign in {+1,-1} such that sign*token_value ==
    add_expr, or None. `s(...)` is tried first, then chains."""
    # Peel a +-1 scalar wrapping the sum. factor_terms can leave a hoisted pole as
    # `opaque_den(-(a+b))`, a Mul(-1, Add), which is not `.is_Add`. Fold on the inner Add
    # and carry the sign, so it collapses to a clean token without redundant parens.
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

    # Bracket classes do not apply antisymmetry under sp.expand (ab(2,1) stays ab(2,1)),
    # so `_make_chain` candidates carry non-canonical brackets. Compare via the atomizer's
    # `reorder_expr` canonicalizer, which folds ab(j,i) -> -ab(i,j) etc.
    reorder = ctx["reorder"]

    def _matches(sgn: int, cand) -> bool:
        return sp.expand(reorder(add_expr - sgn * cand)) == 0

    # --- s(i,j,...) over the full leg set, C(m,2) terms ---
    cand = ctx["mand"](*legs)
    for sgn in (1, -1):
        if _matches(sgn, cand):
            return f"s({', '.join(map(str, legs))})", sgn * outer_sign

    # --- single-channel chain('as'|'sa', a, [P], b) with endpoints a, b and P the rest ---
    # 'as', the angle-first textbook spelling <a|P|b], is preferred to the value-equal 'sa'
    # form, so folded chains match how users write them.
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
# Loose-pool fold, which groups the bare ab/sb multiset of one product term
# --------------------------------------------------------------------------- #
def _fold(factors: List[_Factor]) -> Tuple[List[Tuple[int, str]], int]:
    """Group a factor multiset into display tokens, returning (tokens, sign). Each token is
    (priority, text) with priority 0 for s, 1 for plain ab/sb and 2 for a chain, the order
    within a priority preserved. `sign` in {+1,-1} is the antisymmetry sign the chain folds
    pull out in the second pass, which the caller multiplies onto the term coefficient. The
    two-particle s sign is left out here and the caller compensates it by counting
    priority-0 tokens in `n_s_folds`.

    The chain test mirrors `objective_atoms.fold_counts`, which asks for exactly one shared
    leg, so the atom score and this display fold agree on which brackets form a chain."""
    abs_ = [(i, j) for (k, i, j) in factors if k == "ab"]
    sbs = [(i, j) for (k, i, j) in factors if k == "sb"]
    used_a = [False] * len(abs_)
    used_s = [False] * len(sbs)
    tokens: List[Tuple[int, str]] = []
    sign = 1

    # First pass, Mandelstams, ab(i,j) * sb(i,j) with identical args.
    for ia, (i, j) in enumerate(abs_):
        for js, (k, l) in enumerate(sbs):
            if not used_a[ia] and not used_s[js] and (i, j) == (k, l):
                used_a[ia] = used_s[js] = True
                tokens.append((0, f"s({i}, {j})"))
                break

    # Second pass, single-momentum Dirac chains, an ab and an sb sharing exactly one leg.
    # The leg test is symmetric and matches objective_atoms.fold_counts, so the atom score
    # and this display agree on which pairs are chains. Written angle-first, a chain reads
    # <a|m|b] = chain('as', a, [m], b) = ab(a,m)*sb(m,b), with `m` the shared momentum and
    # a, b the endpoints. The bracket pair may carry `m` in either slot. Bracket
    # antisymmetry, ab(j,i)=-ab(i,j) and sb(j,i)=-sb(i,j), then contributes a ±1 folded
    # into `sign` so the rewrite stays value-exact.
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
            # ab(ai,aj) = s_ab * ab(a,m), +1 iff m is already the second slot, else -1 for the swap.
            s_ab = 1 if aj == m else -1
            # sb(si,sj) = s_sb * sb(m,b), +1 iff m is already the first slot, else -1 for the swap.
            s_sb = 1 if si == m else -1
            sign *= s_ab * s_sb
            used_a[ia] = used_s[js] = True
            tokens.append((2, f"chain('as', {a}, [{m}], {b})"))
            break

    # Third pass, leftovers stay as plain brackets.
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
    """Render one product term, coeff * numerator factors / denominator factors, folding
    brackets through the loose pool and Add and opaque_den sub-expressions through
    `_fold_add`."""
    ab_cls, sb_cls = ctx["ab"], ctx["sb"]
    coeff, rest = term.as_coeff_Mul()

    num_pool: List[_Factor] = []
    den_pool: List[_Factor] = []
    num_groups: List[Tuple[int, str]] = []  # (priority, text), with priority 0 for s, 2 for chain and 3 for the rest
    den_groups: List[Tuple[int, str]] = []
    # In prefactor mode, ctx["prefactor"] set by gbfs_f only, a sum-of-fractions numerator
    # over a common denominator A is remembered as rendered, so a denominator factor can be
    # hoisted to a leading `-1/A*(...)` prefactor. `num_frac_sign` carries the overall sign
    # pulled out of the inner sum, so the inner part renders with all plus signs.
    num_frac_sum: Optional[str] = None
    num_frac_sign: int = 1

    for fac in sp.Mul.make_args(rest):
        base, exp = (fac.base, fac.exp) if fac.is_Pow else (fac, sp.Integer(1))
        if not exp.is_Integer:
            # A non-integer power, rare after atomization, is kept raw.
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
            # opaque_den(S) is a symbolic 1/S pole and always goes to the denominator.
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
                # Render the sum term-wise so an all-negative sum is detected at the display
                # level. The per-term minus is a rendering artifact of the fold sign conventions,
                # s(i,j) = -ab(i,j)*sb(i,j) and chain antisymmetry, so base.args' sympy signs are
                # the wrong signal. On eq 5.28 the Add and its args all report
                # could_extract_minus_sign() False, yet every term renders with a minus.
                term_pieces = ([_render_term(t, ctx) for t in sp.Add.make_args(base)]
                               if base.is_Add else [_render(base, ctx)])
                frac_sign = 1
                if (ctx.get("prefactor") and not to_den and mult == 1
                        and len(term_pieces) > 1
                        and all(p.startswith("-") for p in term_pieces)):
                    # In prefactor mode, pull the overall minus out of an all-negative sum so the
                    # inner part renders with plus signs and the sign rides on the leading -1/A
                    # prefactor, as in -1/A*(B/C + D/E). This fires only when every term is
                    # negative, so a mixed-sign sum never gets a spurious leading minus.
                    stripped = [p[1:] for p in term_pieces]
                    inner = " + ".join(stripped)
                    frac_sign = -1
                else:
                    # The normal join, byte-identical to `_render`'s Add branch, which was
                    # already tried as a fold above, so output without a prefactor is unchanged.
                    inner = term_pieces[0]
                    for p in term_pieces[1:]:
                        inner += (" - " + p[1:]) if p.startswith("-") else (" + " + p)
                text = _pow_text(f"({inner})", mult)
                groups.append((3, text))
                # Record only a sum-of-fractions numerator, one with an inner `/`, so a
                # plain `(a+b)` numerator is untouched.
                if ctx.get("prefactor") and not to_den and mult == 1 and "/" in inner:
                    num_frac_sum, num_frac_sign = text, frac_sign
        else:
            # A bare number or unrecognised atom is absorbed into the coefficient.
            coeff = coeff * fac

    num_folded, num_chain_sign = _fold(num_pool)
    den_folded, den_chain_sign = _fold(den_pool)
    # Chain folds may pull out a bracket-antisymmetry sign in the second pass of `_fold`. It
    # is folded onto the coefficient, for the denominator too, since a sign in {+1,-1} gives
    # 1/(s*T)=s/T.
    coeff *= num_chain_sign * den_chain_sign
    # Under the textbook convention s(i,j)=<ij>[ji]=ab(i,j)*sb(j,i), a bare canonical pair
    # ab(i,j)*sb(i,j) is <ij>[ij] = -s(i,j), so every two-particle s fold, priority 0 in the
    # loose pool, flips the sign. The coefficient compensates, for the numerator and the
    # denominator alike, since 1/(ab*sb) = -1/s. Multi-particle s and chains handled through
    # _fold_add already carry their sign in `coeff` above.
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
            # A pole common to every term of the sum, hoisted out front as "-1/A*(B/C + D/E)".
            body = f"1/{den_body}*{body}"
            if num_frac_sign < 0:
                sign = "" if sign == "-" else "-"   # the pulled-out minus flips the term sign
        else:
            body = f"{body}/{den_body}"
    return sign + body


def _render(expr: sp.Expr, ctx: Dict) -> str:
    """Render an expression. A top-level Add is first tried as a single foldable s or chain,
    so a standalone `s(1,2,3)` folds whole. Otherwise it is rendered term-wise with each term
    loose-pool-folded, and a bare user sum is never collapsed."""
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

    Recognises two-particle and multi-particle `s(...)`, single-momentum and single-channel
    `chain(...)`, and folds `opaque_den(S)` denominator markers back to their s or chain
    form. It is display-only and value-preserving, re-atomizing identically.

    `prefactor`, False by default and True only from gbfs_f, renders a sum-of-fractions
    numerator over a common denominator A as a leading `-1/A*(B/C + D/E)` prefactor rather
    than the two-story `(-B/C - D/E)/A`. The flag decides, since the expression shape alone
    cannot tell gbfs_f's output from gbfs_e's, the same sum-of-fractions class.
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
    # Hoist factors common to every term of a sum out of the sum, a value-preserving regroup.
    # A backend that distributes shared poles across a big sum, for example gbfs replicating
    # three opaque_den(S) markers into all 24 numerator terms, would otherwise make every term
    # carry the same denominator with a second outer denominator on top. factor_terms pulls
    # the shared factors out once, so the whole expression renders over a single combined
    # denominator.
    expr = sp.factor_terms(expr)
    return _render(expr, ctx)
