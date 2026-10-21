"""GBFS version D ("version 4") objective — the "physical-atom" count.

The version-D simplicity objective, "minimize the sum of brackets, Dirac chains, and S
symbols": partition each term's spinor-bracket multiset into
three **disjoint** kinds of physical atoms and count the atoms, NOT the raw brackets:

  * **S symbol** (Mandelstam)  : `ab(i,j) * sb(i,j)`   — same two legs           -> 1 atom
  * **Dirac chain**            : `ab(a,m) * sb(m,b)`    — share exactly one leg m -> 1 atom
  * **bare bracket**           : whatever is left over                            -> 1 atom each

Because every bracket belongs to exactly ONE atom, the three counts are a partition — there
is no double counting (that redundancy is precisely why the earlier
`w_b*brackets + w_c*chains + w_s*Sij` weighted-sum was rejected: it re-counted the same
brackets). An S symbol / chain folds two brackets into one atom, so `atom_count <=
bracket_count`, with the gap = (#S + #chains). Minimizing atoms therefore rewards reaching an
equivalent form whose brackets pair up into physical invariants — a *different* notion of
simple from the raw/factored bracket count that plain `model_free_gbfs` optimises (hence a
separate algorithm, not a replacement).

Matching note: we use the **symmetric** leg-sharing test (`|{i,j} ∩ {k,l}| == 1` for a chain,
equal unordered pairs for an S symbol), which is robust to the engine's `i<j` bracket
canonicalisation. The display-side `reconstruct._fold` was aligned to this test (same symmetric
leg check + greedy order) on 2026-07-14, so the atom score and the rendered `s()/chain()` form
now agree on which brackets are a chain (`_fold` carries the bracket-antisymmetry sign so the
rewrite stays value-exact). Multi-channel chains `<a|P Q|b]` remain unfolded on both sides.

Fold priority is fixed (`S > chain > bare`) so the count is canonical/deterministic. Greedy
matching (not max-cardinality) — deterministic and identical to the display fold's greedy
order; a globally-minimal matching is a possible future refinement.
"""
from __future__ import annotations

from typing import List, Tuple


def fold_counts(ab_pairs: List[Tuple[int, int]],
                sb_pairs: List[Tuple[int, int]]) -> Tuple[int, int, int]:
    """Greedily fold one term's brackets into (n_s, n_chain, n_bare).

    `ab_pairs` / `sb_pairs` are one canonical `(i, j)` leg pair per occurrence (exponents
    already expanded). S symbols are matched first (same unordered pair), then single-leg
    Dirac chains (share exactly one leg), then the remainder counts as bare brackets.
    """
    used_a = [False] * len(ab_pairs)
    used_s = [False] * len(sb_pairs)
    n_s = 0
    n_chain = 0

    # PASS 1 — S symbol (Mandelstam): ab(i,j) * sb(i,j), identical legs.
    for ia, (i, j) in enumerate(ab_pairs):
        for js, (k, l) in enumerate(sb_pairs):
            if not used_s[js] and (i, j) == (k, l):
                used_a[ia] = used_s[js] = True
                n_s += 1
                break

    # PASS 2 — single-leg Dirac chain: ab and sb sharing EXACTLY one leg.
    for ia, (i, j) in enumerate(ab_pairs):
        if used_a[ia]:
            continue
        legs = {i, j}
        for js, (k, l) in enumerate(sb_pairs):
            if used_s[js]:
                continue
            if len(legs & {k, l}) == 1:
                used_a[ia] = used_s[js] = True
                n_chain += 1
                break

    # PASS 3 — leftovers stay bare.
    n_bare = used_a.count(False) + used_s.count(False)
    return n_s, n_chain, n_bare


def _term_pairs(brackets) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split one `term_powers` bracket list into (ab_pairs, sb_pairs), exponents expanded.

    Engine encoding (`Expr.term_powers`): each entry is `(is_square, i, j, exp)` with
    `is_square` True for `sb` (square) and False for `ab` (angle), legs already canonical
    (`i < j`).
    """
    ab_pairs: List[Tuple[int, int]] = []
    sb_pairs: List[Tuple[int, int]] = []
    for (is_square, i, j, exp) in brackets:
        pair = (i, j)
        bucket = sb_pairs if is_square else ab_pairs
        bucket.extend([pair] * int(exp))
    return ab_pairs, sb_pairs


def atom_count(handle, w_s: float = 1.0, w_chain: float = 1.0, w_bare: float = 1.0) -> float:
    """Total physical-atom count of `handle`'s numerator, summed over terms.

    Weights default to 1 (the plain "sum of brackets, Dirac chains, S symbols"
    objective); they are parameters only so a weighted variant is a one-line change later —
    the CLI/runner does NOT expose them for version D.

    Denominator is excluded, matching the engine's `bracket_count` / `factored_bracket_count`
    convention (the numerator is where the simplification search actually reshapes structure).
    """
    total = 0.0
    for _coeff, brackets in handle.term_powers():
        ab_pairs, sb_pairs = _term_pairs(brackets)
        n_s, n_chain, n_bare = fold_counts(ab_pairs, sb_pairs)
        total += w_s * n_s + w_chain * n_chain + w_bare * n_bare
    return total


def atom_heuristic(handle):
    """version-D search objective: `(atom_count, num_terms)` — smaller is simpler, terms as tie-break.

    Same tuple shape as `gbfs.heuristic` so it drops into `gbfs.simplify(objective_fn=...)`
    for both frontier ordering and best-node tracking.
    """
    return (atom_count(handle), handle.num_terms())
