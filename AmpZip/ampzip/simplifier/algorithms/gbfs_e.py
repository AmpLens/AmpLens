"""The gbfs_e variant, a per-term driver.

As `docs/GBFS_VARIANTS.md` describes, gbfs_e splits the numerator into its additive terms,
simplifies each term as an independent sub-problem and recombines. This decomposition is
orthogonal to the objective each per-term sub-search uses, and it drives the same engine as
the default search and gbfs_d.

It serves the local notion of simplicity, breaking an expression into simpler components
with the same denominator. Each additive term is reduced on its own, giving a per-term-local
form. Independent term searches cannot use cancellations that require joint rewrites of
different terms, which can make them less effective than the default search and gbfs_d for
expressions that vanish identically or for spurious-pole cancellations. On compact chain and
sum-denominator inputs, keeping the terms separate skips the `sp.cancel` over a common
denominator that inflates a monolithic backend's display, a three-term compact form being
able to cancel into more than 1000 brackets. gbfs_e keeps each term compact.

The split happens before the common denominator. The atomizer combines a summed input into
one fraction over a common denominator through `sp.together` in
`atomize._rewrite_with_opaque_den`, so by the time a sympy `expr` reaches an algorithm the
additive structure is gone and a three-term input arrives as a single `Mul`. gbfs_e therefore
splits the raw input string from before atomization at its top-level `+` and `-` through the
AST, atomizes each term separately so each keeps its own denominator, possibly a sum, and
runs the engine per term. `runner.run_all` threads the raw string in as the `raw_input`
kwarg, from the CLI's `record.raw_input` or the web's `body.expr`.

The fallbacks preserve the value. Without `raw_input`, when
called directly or when the raw text does not parse, the numerator of `sp.together(expr)` is
split into its `Add.make_args` terms over the common denominator without `expand`, since
expanding a compact numerator once blew a three-term input up into more than 100 terms and
collapsed gbfs_e into a whole-expression search. With a single additive term, a pure monomial
or product, or with more than `MAX_SPLIT_TERMS` terms, one whole-expression search runs. The
reason the raw split was skipped is surfaced in `extra["raw_split_skipped"]` as one of
no_raw_input, unparseable, single_term, cap and atomize_error, or None when the raw split
succeeded. `extra["split_capped"]` only fires on the expression-level fallback, so on a
sum-denominator input the cap shows up as `raw_split_skipped == "cap"` with `split_capped`
still False.

Recombination skips `together` and `cancel` on the sum, which would re-merge the terms and
reintroduce the blow-up. Each per-term best is optionally post-processed on its own, then the
pieces are summed and the recombined result is checked against the input by the finite-field
test.

Whether the per-term objective should default to the bracket count, as now, or to gbfs_d's
atom count is open. It is one class attribute, `per_term_objective_fn`.
"""
from __future__ import annotations

import ast
import time
from typing import List, Optional, Tuple

import sympy as sp

from ..result import AlgorithmResult
from ._post import count_brackets as _count_brackets, maybe_simplify
from .gbfs import GBFS, verify_equal, withheld
from ._post import judge_solved, post_process
from ..limits import DEFAULT_GBFS_SPLIT_MAX_TERMS

# The split cap is the constructor parameter `split_max_terms`, the CLI's
# `--gbfs-split-max-terms`, with the default limits.DEFAULT_GBFS_SPLIT_MAX_TERMS. Above it the
# input is searched whole and extra["split_capped"] and extra["raw_split_skipped"] say so.


class GBFSVersionE(GBFS):
    name = "gbfs_e"

    # The objective each per-term sub-search optimises. None means the engine's factored-bracket
    # `heuristic`. `ampzip.search.objective_atoms.atom_heuristic` gives a per-term gbfs_d. The
    # denominator-aware frontier bias is off, since the pieces are small and the per-term answer
    # should be deterministic and order-independent.
    per_term_objective_fn = None
    use_den_aware = False

    # This driver needs the input string from before atomization, to split at the top-level + and
    # - before atomize's sp.together merges the terms. runner.run_all threads it into run() as
    # raw_input for any algorithm whose class sets this True. gbfs_f inherits it.
    wants_raw_input = True

    def __init__(self, *args, split_max_terms: int = DEFAULT_GBFS_SPLIT_MAX_TERMS, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_max_terms = int(split_max_terms)   # 0 means no cap

    def _over_split_cap(self, k: int) -> bool:
        return self.split_max_terms > 0 and k > self.split_max_terms

    # -- splitting -------------------------------------------------------------------------- #
    def _split_raw(self, raw_input: str, n_point: int,
                   loc: dict) -> Tuple[Optional[List[sp.Expr]], Optional[str]]:
        """Split the raw input at its top-level + and - through the AST and atomize each term
        separately, so each additive term keeps the denominator it had before the common
        denominator. Returns `(pieces, None)` on success, or `(None, reason)` to request the
        expression-level split, with reason one of "unparseable", "single_term", "cap" and
        "atomize_error", which the caller surfaces in extra["raw_split_skipped"]."""
        from ..atomize import atomize, AtomizeError
        try:
            node = ast.parse(raw_input, mode="eval").body
        except SyntaxError:
            return None, "unparseable"

        def flatten(nd):
            if isinstance(nd, ast.BinOp) and isinstance(nd.op, (ast.Add, ast.Sub)):
                return flatten(nd.left) + [(type(nd.op).__name__, nd.right)]
            return [("Add", nd)]

        terms = flatten(node)
        if len(terms) <= 1:
            return None, "single_term"
        if self._over_split_cap(len(terms)):
            return None, "cap"  # too many terms, so the caller searches the whole expression
        pieces: List[sp.Expr] = []
        for op, sub in terms:
            try:
                e = sp.sympify(atomize(ast.unparse(sub), n_hint=n_point), locals=loc)
            except (AtomizeError, Exception):  # noqa: BLE001
                return None, "atomize_error"  # a term that fails abandons the raw split
            pieces.append(e if op == "Add" else -e)
        return pieces, None

    def _split_expr(self, expr: sp.Expr) -> Tuple[List[sp.Expr], int, bool]:
        """The fallback split without raw input, the additive terms of the common-denominator
        numerator without expand. Returns (pieces, n_terms, capped)."""
        together = sp.together(expr)
        num, den = sp.fraction(together)
        terms = sp.Add.make_args(num)
        n = len(terms)
        if n <= 1:
            return [together], n, False
        if self._over_split_cap(n):
            return [together], n, True
        return [t / den for t in terms], n, False

    # -- per-piece search and recombination seam -------------------------------------------- #
    # These two hooks are the seam gbfs_f overrides. gbfs_e searches every piece and accepts the
    # sum of the searched pieces verbatim.
    def _search_piece(self, piece, n_point, piece_budget, simplify_mode,
                      cancel_event, loc, gbfs_simplify, ab_cls, sb_cls, piece_apps=None):
        """Search one additive piece and return (chosen, searched, row, res). `chosen` is the
        piece to accumulate into the running recombination, which for gbfs_e is `searched`, the
        engine's post-processed result, kept distinct so gbfs_f can offer the all-searched
        candidate even when it rejects the searched piece. `row` is the per_term telemetry dict
        and `res` the raw engine result. The accumulators stay in run(), out of every override.
        """
        res = gbfs_simplify(
            str(piece), n_point, max_apps=(self.max_apps if piece_apps is None else piece_apps),
            goal_brackets=None, max_seconds=piece_budget,
            value_fn=self._value_fn(), objective_fn=self.per_term_objective_fn,
            should_stop=(cancel_event.is_set if cancel_event is not None else None),
            max_terms=self.max_terms, threads=self.threads,
        )
        # Post-process each piece on its own, bounded, with no cross-term re-merge.
        post = post_process(res["best_expr"], simplify_mode, timeout_s=self.post_timeout_s)
        best_str = post.expr_str if post.bracket_count >= 0 else res["best_expr"]
        best_piece = sp.sympify(best_str, locals=loc)
        row = {"piece": str(piece), "best": best_str,
               "searched": True, "brackets": res["best_brackets"],
               "depth": res["depth"], "apps": res["apps"],
               "max_apps": (self.max_apps if piece_apps is None else piece_apps),
               "max_seconds": piece_budget, "stop_reason": res["stop_reason"],
               "term_limit_skips": res["term_limit_skips"],
               "overflow_skips": res["overflow_skips"],
               "engine_error_skips": res["engine_error_skips"],
               "post_process_timed_out": post.timed_out}
        return best_piece, best_piece, row, res

    def _recombine(self, mixed, orig_pieces, searched_pieces, expr):
        """Choose the final expression from the accumulated pieces. gbfs_e accepts the per-piece
        mix unconditionally and adds nothing to extra. Returns (expr, guard_extra)."""
        return mixed, {}

    # -- run -------------------------------------------------------------------------------- #
    def run(
        self,
        expr: sp.Expr,
        n_point: int,
        target: Optional[sp.Expr] = None,
        max_steps: int = 50,                 # ignored, since gbfs has no step budget
        per_step_timeout: float = 30.0,      # ignored
        sample_timeout: float = 120.0,       # ignored, the SFT and CDS rollout budget
        capture_derivation: bool = False,    # the per-term driver emits a per-term summary
        simplify_mode: str = "none",
        sft_patience: int = 0,               # ignored
        cancel_event=None,                   # a threading.Event checked between terms and during search
        raw_input: Optional[str] = None,     # the input before atomization, for the split
        **kwargs,                            # absorb any other runner kwargs
    ) -> AlgorithmResult:
        start_brackets = _count_brackets(expr)
        t0 = time.time()
        # The wall-clock budget is max_seconds alone. sample_timeout is the SFT/CDS rollout
        # budget. Deployments bound this through
        # the limit on gbfs_max_seconds, WEBSVC_MAX_GBFS_MAX_SECONDS on the web.
        budget_s = self.max_seconds

        try:
            from ampzip.search.gbfs import simplify as gbfs_simplify
            from ampzip.atoms import ab as _ab, sb as _sb
            loc = {"ab": _ab, "sb": _sb, "opaque_den": sp.Function("opaque_den")}

            if raw_input:
                pieces, raw_skip_reason = self._split_raw(raw_input, n_point, loc)
            else:
                pieces, raw_skip_reason = None, "no_raw_input"
            if pieces is not None:
                split_mode, n_terms, capped = "raw_terms", len(pieces), False
            else:
                pieces, n_terms, capped = self._split_expr(expr)
                split_mode = "numerator_terms"

            recombined = sp.Integer(0)
            per_term = []
            orig_pieces = []       # the un-searched pieces, in order
            searched_pieces = []   # the engine's per-piece results, aligned with them
            total_apps = total_expanded = 0
            depth_sum = 0
            stopped = False
            piece_reasons = []
            skips = {"term_limit_skips": 0, "overflow_skips": 0, "engine_error_skips": 0}
            for idx, piece in enumerate(pieces):
                elapsed = time.time() - t0
                remaining = (budget_s - elapsed) if budget_s is not None else None
                remaining_apps = self.max_apps - total_apps
                cancelled = cancel_event is not None and cancel_event.is_set()
                # Out of time or moves, or cancelled. The remaining pieces stay unsimplified.
                if cancelled or (remaining is not None and remaining <= 0) or remaining_apps <= 0:
                    recombined += piece
                    orig_pieces.append(piece)
                    searched_pieces.append(piece)   # no search, so searched equals original
                    stopped = stopped or cancelled
                    why = "cancelled" if cancelled else ("budget" if remaining_apps <= 0 else "time")
                    piece_reasons.append(why)
                    per_term.append({"piece": str(piece), "best": str(piece),
                                     "searched": False, "brackets": _count_brackets(piece),
                                     "apps": 0, "stop_reason": why})
                    continue
                n_left = len(pieces) - idx
                # The variant-level budgets bound all pieces together. What is left is shared evenly over
                # the pieces still to run, and a piece that stops early hands its unused share on.
                piece_budget = (remaining / n_left) if remaining is not None else None
                piece_apps = remaining_apps // n_left if n_left > 1 else remaining_apps
                chosen, searched, row, res = self._search_piece(
                    piece, n_point, piece_budget, simplify_mode,
                    cancel_event, loc, gbfs_simplify, _ab, _sb, piece_apps=piece_apps)
                piece_reasons.append(res["stop_reason"])
                for k in skips:
                    skips[k] += res[k]
                recombined += chosen
                orig_pieces.append(piece)
                searched_pieces.append(searched)
                total_apps += res["apps"]
                total_expanded += res["expanded"]
                depth_sum += res["depth"]
                stopped = stopped or res.get("stopped", False)
                per_term.append(row)

            # The recombination seam. gbfs_f overrides it and gbfs_e returns the mix unchanged.
            recombined, guard_extra = self._recombine(recombined, orig_pieces,
                                                      searched_pieces, expr)
            # No progress means the winning candidate applied no net identity, so a smaller bracket
            # count reflects a different spelling, the raw input, the un-searched split baseline, or
            # the per-piece mix when every searched piece was rejected. gbfs_e never populates
            # guard_winner, so this stays False for it.
            _winner = guard_extra.get("guard_winner")
            guard_no_progress = (
                _winner in ("split_baseline", "input")
                or (_winner == "mix" and not any(
                    p.get("searched") and not p.get("kept_original", True) for p in per_term))
            )
        except Exception as e:  # noqa: BLE001
            return AlgorithmResult(
                algorithm=self.name, final_expr_str=str(expr),
                bracket_count_start=start_brackets, bracket_count_final=start_brackets,
                bracket_count_min=start_brackets, n_steps=0, solved=False, timed_out=False,
                duration_s=time.time() - t0,
                extra={"error": f"{type(e).__name__}: {e}", "stop_reason": "error", **self._params()},
            )

        duration = time.time() - t0
        # Aggregate stop reason over the pieces, in priority order.
        reason = "frontier_empty"
        for cand in ("cancelled", "time", "budget", "patience"):
            if cand in piece_reasons:
                reason = cand
                break
        if reason == "frontier_empty" and "goal" in piece_reasons:
            reason = "goal"
        # Recombine without together or cancel, which would re-merge the terms and reintroduce the
        # common-denominator blow-up. The sum of the per-term-simplified pieces is the answer.
        final_str = str(recombined)
        bk_final = _count_brackets(recombined)

        # gbfs_f can choose a candidate, the split baseline or the raw input, that applied no
        # identity, so a smaller bracket count reflects a different spelling. In that case the step
        # count is zero and `solved` is not reported on the bracket delta alone. gbfs_e never sets
        # this flag.
        if guard_no_progress:
            depth_sum = 0

        # --- the recombined answer must equal the input on shell ---
        sound, note = verify_equal(final_str, str(expr), n_point)
        if sound is not True:
            return withheld(self.name, expr, start_brackets, duration,
                            (f"the finite-field test failed, the {self.name} recombined output differing from the input on shell"
                             if sound is False else f"the finite-field test could not be completed, {note}")
                            + ", so the result is withheld", final_str, stop_reason=reason, **self._params())
        target_equal, target_counts = None, {}
        if target is None:
            solved = (not guard_no_progress) and (bk_final < start_brackets)
        else:
            target_equal, tnote = verify_equal(final_str, str(target), n_point)
            if self.target_criterion == "none":
                solved = target_equal is True
            else:
                by_count, target_counts = judge_solved(final_str, str(target), self.target_criterion,
                                                      max_terms=self.max_terms)
                solved = bool(by_count) and target_equal is True
            if tnote:
                target_counts["target_check_note"] = tnote

        return AlgorithmResult(
            algorithm=self.name,
            final_expr_str=final_str,
            min_expr_str=final_str,
            bracket_count_start=start_brackets,
            bracket_count_final=bk_final,
            bracket_count_min=bk_final,
            n_steps=depth_sum,
            solved=solved,
            timed_out=(reason == "time"),
            duration_s=duration,
            derivation=None,   # k independent searches give no single chain
            extra={
                "strategy": "per_term_decomposition",
                "split_mode": split_mode,        # "raw_terms" or "numerator_terms"
                "n_terms": n_terms,
                "n_pieces_searched": sum(1 for p in per_term if p["searched"]),
                "split_capped": capped,
                "raw_split_skipped": raw_skip_reason,   # None, no_raw_input, unparseable, single_term, cap or atomize_error
                "max_split_terms": self.split_max_terms,
                "per_term": per_term,
                "apps": total_apps,
                "expanded": total_expanded,
                **self._params(),
                "split_max_terms": self.split_max_terms,
                "stopped": stopped,
                "stop_reason": reason,
                **skips,
                "soundness_verified": True,
                "post_process_timed_out": any(p.get("post_process_timed_out") for p in per_term),
                "post_process_bounded": True,
                "target_equal": target_equal,
                **target_counts,
                "raw_final_expr": str(recombined),
                "simplify_mode": simplify_mode,
                **guard_extra,   # guard_winner and guard_candidates from gbfs_f, empty for gbfs_e
            },
        )
