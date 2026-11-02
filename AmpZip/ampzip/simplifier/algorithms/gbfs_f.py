"""The gbfs_f variant, per-term decomposition with a no-inflation guard.

It serves the local notion of simplicity, breaking an expression into simpler components
with the same denominator. Each additive term stays its own local object over its own
denominator, and the pole common to every term is displayed once, as the leading prefactor
in -1/A * (B/C + D/E).

gbfs_f is gbfs_e plus one rule at recombination, so the two are a controlled comparison. At
recombination it chooses the smallest, by the internal whole-expression bracket count, of
four candidates that are all already computed, with no extra search. They are the per-piece
keep-the-better mix, the running recombination, the sum of the un-searched pieces, the sum
of the searched pieces, which is exactly what gbfs_e returns, and the atomized input.

The selected candidate therefore has an internal bracket count no greater than the input's
or the all-searched sum's. This count includes denominator brackets and is computed by
`_post.count_brackets`. The guarantee does not apply to the factored numerator counts
reported to the user.

The guard covers the whole sum because `_post.count_brackets` runs `sp.factor_terms` over
the whole sum first, so the count is not additive over pieces. On eq 5.28, cb(o0) and
cb(o1) are both 26 but cb(o0+o1) is 48. A locally larger searched piece can share a factor
with the other pieces that the original does not, and rejecting it per piece destroys a
global hoist. In measured counterexamples every per-piece decision was correct yet the
result came out worse than the input and worse than gbfs_e, 9 against 6. Including the
all-searched sum and the input as whole-result candidates rules both out.

The guard, and no objective, is the lever because the engine represents an expanded
numerator over a monomial denominator with optional polynomial factors. It cannot treat
`chain('as',6,[1,2],3)**3` as a unit and re-serialises it as 4 expanded monomials, 26
brackets becoming 38 per piece. Every objective scores only the engine's numerator inside
that same expanded representation, so the fix has to be a decision made after the engine
returns, in sympy. On eq 5.28 the input counts 68, the default search and gbfs_d 312,
gbfs_e 72 and gbfs_f 48.

Two limits of that 48 count. On eq 5.28 both pieces are search-rejected, so gbfs_f's answer
is byte-identically the input re-split. The 48 is 26 + 26 - 4, the shared pole <2|1+6|5]
counted once after factor_terms. `extra["guard_candidates"]` reports all four counts, and
`solved` is False and `n_steps` 0 in this case, no identity having been applied. And gbfs_f
is blind to cross-term cancellation, vanishing or not. On
ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4) it reaches 5 where model_free_gbfs
reaches 1. Splitting makes each piece a monomial, and no per-piece search can recover the
cross-term cancellation. gbfs_f therefore runs in a best-of alongside model_free_gbfs,
which owns that class. Alone, it is unsafe on cancelling inputs.
"""
from __future__ import annotations

import sympy as sp

from ._post import count_brackets as _count_brackets
from .gbfs_e import GBFSVersionE


class GBFSVersionF(GBFSVersionE):
    name = "gbfs_f"
    # Inherited from gbfs_e, per_term_objective_fn=None, use_den_aware=False and wants_raw_input=True.

    def _search_piece(self, piece, n_point, piece_budget, simplify_mode,
                      cancel_event, loc, gbfs_simplify, ab_cls, sb_cls, piece_apps=None):
        best_piece, searched, row, res = super()._search_piece(
            piece, n_point, piece_budget, simplify_mode,
            cancel_event, loc, gbfs_simplify, ab_cls, sb_cls, piece_apps=piece_apps)
        # Compare both pieces using the internal whole-expression bracket count.
        # res["best_brackets"] uses the engine's numerator count.
        orig_bk = _count_brackets(piece)
        new_bk = _count_brackets(best_piece)
        row["orig_brackets"] = orig_bk
        row["searched_brackets"] = new_bk
        row["kept_original"] = bool(new_bk >= orig_bk)   # informational, the decision being in _recombine
        # "searched" stays True. The search ran, and only its result may be rejected here.
        chosen = piece if new_bk >= orig_bk else best_piece
        # Return `searched`, the best piece, unchanged so run()'s searched_pieces list keeps the
        # engine's result for the all-searched candidate even when it is rejected per piece.
        return chosen, searched, row, res

    def _recombine(self, mixed, orig_pieces, searched_pieces, expr):
        cands = [
            ("mix", mixed),                              # the per-piece keep-the-better mix
            ("split_baseline", sp.Add(*orig_pieces)),    # the un-searched split baseline
            ("all_searched", sp.Add(*searched_pieces)),  # gbfs_e's answer, which makes f <= e a theorem
            ("input", expr),                             # which makes f <= start a theorem
        ]
        scored = [(lbl, e, _count_brackets(e)) for lbl, e in cands]
        # min() is stable, so on a tie the earliest candidate in `cands` wins, mix before split before all before input.
        label, best, _bk = min(scored, key=lambda c: c[2])
        # run() derives `guard_no_progress` from guard_winner and the per_term kept_original
        # flags, since a mix whose pieces were all kept original applied no identity either,
        # then zeros n_steps and refuses `solved` on the bracket delta alone, as
        # GBFSVersionE.run() does.
        return best, {
            "guard_winner": label,
            "guard_candidates": {c[0]: c[2] for c in scored},
        }
