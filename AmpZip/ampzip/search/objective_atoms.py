"""The gbfs_d objective, the physical-atom count.

gbfs_d minimises the sum of brackets, Dirac chains and S symbols. Each term's spinor-bracket
multiset is partitioned into three kinds of physical atom and the atoms are counted. An S
symbol, the Mandelstam `ab(i,j) * sb(i,j)` with the same two legs, is one atom. A Dirac
chain `ab(a,m) * sb(m,b)`, sharing exactly one leg m, is one atom. Every leftover bracket is
one atom.

Every bracket belongs to exactly one atom, so the three counts are a partition with no
double counting. An earlier weighted sum `w_b*brackets + w_c*chains + w_s*Sij` was dropped
because it re-counted the same brackets. An S symbol or chain folds two brackets into one
atom, so `atom_count <= bracket_count`, the gap being the number of S symbols and chains.
Minimising atoms therefore rewards an equivalent form whose brackets pair up into physical
invariants, a notion of simplicity different from the bracket count that `model_free_gbfs`
optimises, hence a separate algorithm.

The leg-sharing test is symmetric, `|{i,j} ∩ {k,l}| == 1` for a chain and equal unordered
pairs for an S symbol, which is robust to the engine's `i<j` bracket canonicalisation. The
display-side `reconstruct._fold` uses the same test and greedy order, so the atom score and
the rendered `s()` and `chain()` form agree on which brackets form a chain. `_fold` carries
the bracket-antisymmetry sign so the rewrite stays value-exact. Multi-channel chains
`<a|P Q|b]` remain unfolded on both sides.

The fold priority is S, then chain, then bare, so the count is canonical. Matching is greedy,
identical to the display fold's order. A globally minimal matching is a possible refinement.
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

    # First pass, S symbols, ab(i,j) * sb(i,j) with identical legs.
    for ia, (i, j) in enumerate(ab_pairs):
        for js, (k, l) in enumerate(sb_pairs):
            if not used_s[js] and (i, j) == (k, l):
                used_a[ia] = used_s[js] = True
                n_s += 1
                break

    # Second pass, single-leg Dirac chains, an ab and an sb sharing exactly one leg.
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

    # Third pass, leftovers stay bare.
    n_bare = used_a.count(False) + used_s.count(False)
    return n_s, n_chain, n_bare


def _term_pairs(brackets) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split one `term_powers` bracket list into (ab_pairs, sb_pairs), exponents expanded.

    In the engine's `Expr.term_powers` encoding each entry is `(is_square, i, j, exp)`,
    `is_square` being True for `sb` and False for `ab`, with the legs already canonical,
    `i < j`.
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

    The weights default to 1, the plain sum of brackets, Dirac chains and S symbols. They
    are parameters so a weighted variant is a one-line change, and the CLI and runner do not
    expose them for gbfs_d.

    The denominator is excluded, as in the engine's `bracket_count` and
    `factored_bracket_count`, since the numerator is where the search reshapes structure.
    """
    total = 0.0
    for _coeff, brackets in handle.term_powers():
        ab_pairs, sb_pairs = _term_pairs(brackets)
        n_s, n_chain, n_bare = fold_counts(ab_pairs, sb_pairs)
        total += w_s * n_s + w_chain * n_chain + w_bare * n_bare
    return total


def atom_heuristic(handle):
    """The gbfs_d objective `(atom_count, num_terms)`, smaller being simpler, with terms as the tie-break.

    Same tuple shape as `gbfs.heuristic` so it drops into `gbfs.simplify(objective_fn=...)`
    for both frontier ordering and best-node tracking.
    """
    return (atom_count(handle), handle.num_terms())
