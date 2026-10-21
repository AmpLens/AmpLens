"""GBFS version E ("version E") — per-term simplification DRIVER (decomposition strategy).

Version (e) (see `docs/GBFS_VARIANTS.md`): **split the numerator into its additive terms and simplify each term as an independent
sub-problem, then recombine.** This is a *decomposition strategy* (search-axis 3), NOT a new
simplicity objective — it is orthogonal to which objective each per-term sub-search uses, and it
wraps the same model-free engine that the default search and (d) drive.

Why it is a different notion of "simple", not a strictly-better one:
  * It serves the "local-factorized-simple / local objects" notion of simplicity — break an
    expression into simpler components with the same denominator: each additive term is reduced
    on its own, giving a per-term-local form rather than the globally-smallest form the default search and (d) chase.
  * Independent term searches cannot use cancellations that require joint rewrites
    of different terms. This can make them less effective than the default search
    and variant (d) for expressions that vanish identically or for spurious-pole
    cancellations that require such rewrites.
  * The *win*: on compact chain/opaque inputs, keeping the terms separate AVOIDS the
    `sp.cancel`-over-a-common-denominator blow-up that inflates a monolithic backend's display
    (a 3-term compact form can `cancel` into 1000+ brackets). (e) keeps each term compact.

### Where the split happens — BEFORE the common denominator (fixed 2026-07-14)

The atomizer combines a summed input into ONE fraction over a common denominator (`sp.together`
inside `atomize._rewrite_with_opaque_den`), so by the time a sympy `expr` reaches an algorithm the
original additive structure is already gone (a 3-term input arrives as a single `Mul`). To honor
"split by the ORIGINAL additive terms", (e) therefore splits the **raw pre-atomize input string**
at its top-level `+`/`-` (via the AST), atomizes each term *separately* (each keeps its own —
possibly opaque — denominator), and runs the engine per term. The raw string is threaded in as the
`raw_input` kwarg by `runner.run_all` (from the CLI's `record.raw_input` / the web's original
`body.expr`).

Fallbacks, all value-preserving and soundness-gated:
  * **No raw_input** (called directly, or raw unparseable): fall back to splitting the numerator of
    `sp.together(expr)` into `Add.make_args` terms over the common denominator — WITHOUT `expand`
    (expanding a compact numerator is what used to explode a 3-term input into 100+ terms and made
    (e) collapse to a whole-expression search).
  * **A single additive term** (`k == 1`, a pure monomial/product) or **more than `MAX_SPLIT_TERMS`
    terms**: run one whole-expression search. The reason the raw split was skipped is surfaced in
    `extra["raw_split_skipped"]` (one of no_raw_input / unparseable / single_term / cap /
    atomize_error, or None when the raw split succeeded); note `extra["split_capped"]` only ever
    fires on the expr-level fallback, so on an OPAQUE input the cap shows up as
    `raw_split_skipped == "cap"`, not `split_capped == True`.

Recombination deliberately does NOT re-`together`/`cancel` the sum (that would re-merge the terms
and reintroduce the blow-up); each per-term best is optionally post-processed on its own, then the
pieces are summed and the recombined result is mod-p soundness-gated against the input.

Open question: whether the per-term objective
should default to the baseline bracket count (current) or the version-D atom count. One-line
class-attribute swap (`per_term_objective_fn`). See `docs/GBFS_VARIANTS.md`.
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

# The split cap is a constructor parameter (`split_max_terms`, CLI `--gbfs-split-max-terms`,
# default limits.DEFAULT_GBFS_SPLIT_MAX_TERMS); above it the input is searched whole and the
# result says so (extra["split_capped"], extra["raw_split_skipped"]).


class GBFSVersionE(GBFS):
    name = "gbfs_e"

    # Objective each per-term sub-search optimises. None -> engine's factored-bracket `heuristic`
    # (the baseline "normal simplifier"). Swap to `ampzip.search.objective_atoms.atom_heuristic`
    # for a per-term version-D. The den-aware frontier bias is off (per-term pieces are small and
    # we want a deterministic, order-independent per-term answer).
    per_term_objective_fn = None
    use_den_aware = False

    # This driver needs the ORIGINAL pre-atomize string (to split at the top-level +/- before
    # atomize's sp.together merges the terms). runner.run_all threads it into run() as raw_input
    # for any algorithm whose class sets this True (see runner.py). gbfs_f inherits it.
    wants_raw_input = True

    def __init__(self, *args, split_max_terms: int = DEFAULT_GBFS_SPLIT_MAX_TERMS, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_max_terms = int(split_max_terms)   # 0 = no cap

    def _over_split_cap(self, k: int) -> bool:
        return self.split_max_terms > 0 and k > self.split_max_terms

    # -- splitting -------------------------------------------------------------------------- #
    def _split_raw(self, raw_input: str, n_point: int,
                   loc: dict) -> Tuple[Optional[List[sp.Expr]], Optional[str]]:
        """Split the ORIGINAL raw input at its top-level +/- (AST) and atomize each term
        separately, so each additive term keeps its own (pre-common-denominator) denominator.
        Returns `(pieces, None)` on success, or `(None, reason)` to signal 'fall back to the
        expr-level split', where reason is one of {"unparseable", "single_term", "cap",
        "atomize_error"} so the caller can surface WHY the raw split was skipped (never silent —
        see extra["raw_split_skipped"])."""
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
            return None, "cap"  # too many -> caller takes the capped whole-expression path
        pieces: List[sp.Expr] = []
        for op, sub in terms:
            try:
                e = sp.sympify(atomize(ast.unparse(sub), n_hint=n_point), locals=loc)
            except (AtomizeError, Exception):  # noqa: BLE001
                return None, "atomize_error"  # any term fails -> abandon raw split
            pieces.append(e if op == "Add" else -e)
        return pieces, None

    def _split_expr(self, expr: sp.Expr) -> Tuple[List[sp.Expr], int, bool]:
        """Fallback split (no raw input): the additive terms of the common-denominator numerator,
        WITHOUT expand. Returns (pieces, n_terms, capped)."""
        together = sp.together(expr)
        num, den = sp.fraction(together)
        terms = sp.Add.make_args(num)
        n = len(terms)
        if n <= 1:
            return [together], n, False
        if self._over_split_cap(n):
            return [together], n, True
        return [t / den for t in terms], n, False

    # -- per-piece search + recombination seam ---------------------------------------------- #
    # These two hooks are the ONLY seam gbfs_f overrides. (e)'s behavior is unchanged: it searches
    # every piece and accepts the sum of searched pieces verbatim.
    def _search_piece(self, piece, n_point, piece_budget, simplify_mode,
                      cancel_event, loc, gbfs_simplify, ab_cls, sb_cls, piece_apps=None):
        """Search ONE additive piece. Returns (chosen, searched, row, res):
          chosen   — the piece to accumulate into the running recombination (for (e) == searched);
          searched — the engine's post-processed result (kept distinct so gbfs_f can offer the
                     all-searched candidate even when it rejects the searched piece per-piece);
          row      — the per_term telemetry dict;
          res      — the raw engine result (accumulators stay in run(), out of every override).
        """
        res = gbfs_simplify(
            str(piece), n_point, max_apps=(self.max_apps if piece_apps is None else piece_apps),
            goal_brackets=None, max_seconds=piece_budget,
            value_fn=self._value_fn(), objective_fn=self.per_term_objective_fn,
            should_stop=(cancel_event.is_set if cancel_event is not None else None),
            max_terms=self.max_terms, threads=self.threads,
        )
        # Post-process each piece on its own (safe: no cross-term re-merge), bounded.
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
        """Hook: choose the final expression from the accumulated pieces. (e) accepts the
        per-piece mix unconditionally and adds nothing to extra. Returns (expr, guard_extra)."""
        return mixed, {}

    # -- run -------------------------------------------------------------------------------- #
    def run(
        self,
        expr: sp.Expr,
        n_point: int,
        target: Optional[sp.Expr] = None,
        max_steps: int = 50,                 # ignored (gbfs is not step-budgeted)
        per_step_timeout: float = 30.0,      # ignored
        sample_timeout: float = 120.0,       # ignored (SFT/CDS rollout budget; gbfs is bounded by max_seconds only)
        capture_derivation: bool = False,    # per-term driver emits a per-term summary, not a chain
        simplify_mode: str = "none",
        sft_patience: int = 0,               # ignored
        cancel_event=None,                   # threading.Event; checked between terms and during search
        raw_input: Optional[str] = None,     # ORIGINAL pre-atomize input, for the pre-common-den split
        **kwargs,                            # absorb any other runner kwargs
    ) -> AlgorithmResult:
        start_brackets = _count_brackets(expr)
        t0 = time.time()
        # The wall-clock budget is max_seconds alone. sample_timeout is the SFT/CDS rollout
        # budget and is NOT folded in (it used to be, silently capping every gbfs run at
        # the 120 s default). Deployments bound this through
        # the ceiling applied to gbfs_max_seconds (web: WEBSVC_MAX_GBFS_MAX_SECONDS).
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
            orig_pieces = []       # the original (un-searched) pieces, in order
            searched_pieces = []   # the engine's per-piece results, in order (kept aligned)
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
                # Out of time, apps or cancelled: keep the remaining pieces unsimplified (value-safe).
                if cancelled or (remaining is not None and remaining <= 0) or remaining_apps <= 0:
                    recombined += piece
                    orig_pieces.append(piece)
                    searched_pieces.append(piece)   # no search -> searched == original
                    stopped = stopped or cancelled
                    why = "cancelled" if cancelled else ("budget" if remaining_apps <= 0 else "time")
                    piece_reasons.append(why)
                    per_term.append({"piece": str(piece), "best": str(piece),
                                     "searched": False, "brackets": _count_brackets(piece),
                                     "apps": 0, "stop_reason": why})
                    continue
                n_left = len(pieces) - idx
                # The variant-level budgets are true ceilings: what is left is shared evenly over
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

            # Recombination seam (gbfs_f overrides; (e) returns the mix unchanged, guard_extra={}).
            recombined, guard_extra = self._recombine(recombined, orig_pieces,
                                                      searched_pieces, expr)
            # "No progress" = the winning candidate applied no net identity, so a smaller bracket
            # count reflects a different SPELLING, not search: the raw input, the un-searched split
            # baseline, OR the per-piece mix when every searched piece was rejected (kept_original).
            # (e) never populates guard_winner, so this stays False for (e) — behavior unchanged.
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
        # Recombine WITHOUT re-together/cancel (that would re-merge the terms and reintroduce the
        # common-denominator blow-up). The sum of per-term-simplified pieces IS the (e) answer.
        final_str = str(recombined)
        bk_final = _count_brackets(recombined)

        # gbfs_f can choose a candidate (the split baseline or the raw input) that applied NO
        # identity — a smaller bracket count then reflects a different SPELLING, not search
        # progress. In that case zero out the step count and refuse to report `solved` on the
        # bracket delta alone. (e) never sets this flag, so its behavior is unchanged.
        if guard_no_progress:
            depth_sum = 0

        # --- soundness gate: the recombined answer must be on-shell equal to the input ---
        sound, note = verify_equal(final_str, str(expr), n_point)
        if sound is not True:
            return withheld(self.name, expr, start_brackets, duration,
                            (f"soundness check FAILED ({self.name} recombined output not on-shell equal to input)"
                             if sound is False else f"soundness check could not be completed ({note})")
                            + " — result withheld", final_str, stop_reason=reason, **self._params())
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
            derivation=None,   # per-term driver: k independent searches, no single linear chain
            extra={
                "strategy": "per_term_decomposition",
                "split_mode": split_mode,        # "raw_terms" (pre-common-den) or "numerator_terms"
                "n_terms": n_terms,
                "n_pieces_searched": sum(1 for p in per_term if p["searched"]),
                "split_capped": capped,
                "raw_split_skipped": raw_skip_reason,   # None | no_raw_input | unparseable | single_term | cap | atomize_error
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
                **guard_extra,   # gbfs_f: guard_winner + guard_candidates; empty for (e)
            },
        )
