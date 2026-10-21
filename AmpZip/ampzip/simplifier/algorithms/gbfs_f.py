"""GBFS version F — local-factorized-simple: per-term decomposition + a no-inflation guard.

Serves the "local-factorized-simple / local objects" notion of simplicity — break an
expression into simpler components with the same denominator: keep each additive term as its OWN
local object over its OWN denominator, and display the pole common to every term once, as a
leading prefactor —  -1/A * (B/C + D/E).

(f) = (e) + ONE recombination-time rule, so (e) vs (f) is a controlled comparison. At
recombination, choose the SMALLEST (by the internal whole-expression bracket count) of four candidates
that are ALL already computed — no extra search:

    cand_mix   = the per-piece keep-the-better mix   (the running recombination)
    cand_orig  = sum of the ORIGINAL pieces          (the un-searched split baseline)
    cand_all   = sum of the SEARCHED pieces           (== exactly what (e) returns)
    cand_input = the atomized input

At recombination, the selected candidate has an internal bracket count no greater
than either cand_input or cand_all. This count includes denominator brackets and
is computed by `_post.count_brackets`. The guarantee does not apply to the
factored numerator counts reported to the user.

### Why a per-PIECE guard is NOT enough (measured — do not "simplify" it back to per-piece)

`_post.count_brackets` runs `sp.factor_terms` over the WHOLE sum first, so the metric is NOT
additive over pieces:  cb(o0)=26, cb(o1)=26, but cb(o0+o1)=48  (eq 5.28). A locally-LARGER
searched piece can share a factor with the OTHER pieces that the original does not; rejecting it
per-piece destroys a global hoist. Measured counterexamples where every per-piece decision was
individually correct yet the RESULT came out worse than the input (9 vs 6) and worse than (e)
(9 vs 6). Including cand_all and cand_input as whole-result candidates makes both impossible.

### Why the guard is the lever at all (not the objective)

The Rust engine represents an expanded numerator over a monomial denominator with optional
polynomial factors. It cannot treat `chain('as',6,[1,2],3)**3` as a unit, so it
re-serializes it as 4 expanded monomials — per piece 26 brackets -> 38.
The engine LOSES here, and NO objective_fn can fix it: every objective
scores only the engine's numerator, inside that same expanded representation. The fix has to be a
decision made AFTER the engine returns, in sympy. Measured, eq 5.28:
    input = 68,  default and (d) = 312,  (e) = 72,  (f) = 48.

### HONEST SCOPE — read before quoting the 48

  * On eq 5.28 BOTH pieces are search-rejected, so (f)'s answer is byte-identically the input
    re-split. The 48 = 26 + 26 - 4 (the shared pole <2|1+6|5] counted once after factor_terms),
    NOT search progress. `extra["guard_candidates"]` reports all four counts; `solved` is False
    and `n_steps` is 0 in this case (no identity was applied).
  * (f) is BLIND to cross-term cancellation, vanishing or not. Measured:
    ab(1,2)*ab(3,4)/ab(1,4) - ab(1,3)*ab(2,4)/ab(1,4)  ->  (f) = 5,  model_free_gbfs = 1.
    Splitting makes each piece a monomial; no per-piece search can recover the cross-term cancel.
    => (f) MUST be run in best-of alongside gbfs_a (a/model_free_gbfs), which owns that class.
    Alone, (f) is unsafe on cancelling inputs.
"""
from __future__ import annotations

import sympy as sp

from ._post import count_brackets as _count_brackets
from .gbfs_e import GBFSVersionE


class GBFSVersionF(GBFSVersionE):
    name = "gbfs_f"
    # Inherited from (e): per_term_objective_fn=None, use_den_aware=False, wants_raw_input=True.

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
        row["kept_original"] = bool(new_bk >= orig_bk)   # informational; the DECISION is in _recombine
        # "searched": True stays — the search DID run; only its result may be rejected here.
        chosen = piece if new_bk >= orig_bk else best_piece
        # Return `searched` (== best_piece) unchanged so run()'s searched_pieces list keeps the
        # engine's result for the all-searched candidate, even when we reject it per-piece.
        return chosen, searched, row, res

    def _recombine(self, mixed, orig_pieces, searched_pieces, expr):
        cands = [
            ("mix", mixed),                              # the per-piece keep-the-better mix
            ("split_baseline", sp.Add(*orig_pieces)),    # the un-searched split baseline
            ("all_searched", sp.Add(*searched_pieces)),  # == (e); makes f <= e a theorem
            ("input", expr),                             # makes f <= start a theorem
        ]
        scored = [(lbl, e, _count_brackets(e)) for lbl, e in cands]
        # min() is stable: on a tie the earliest candidate in `cands` wins (mix > split > all > input).
        label, best, _bk = min(scored, key=lambda c: c[2])
        # run() derives `guard_no_progress` from guard_winner + the per_term kept_original flags
        # (a "mix" whose pieces were all kept-original applied no identity either), then zeros
        # n_steps and refuses `solved` on the bracket delta alone. See GBFSVersionE.run().
        return best, {
            "guard_winner": label,
            "guard_candidates": {c[0]: c[2] for c in scored},
        }
