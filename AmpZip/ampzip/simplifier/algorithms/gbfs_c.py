"""GBFS version C ("version C") — expand-everything + polynomial-division cancel.

Variant (c) is specified as: fully expand the whole expression **including the denominator**
into one rational function, then at every search step cancel with polynomial division /
factorization. This is an
**experiment / comparison** variant (NOT a performance play): it is the *inverse* of the shipped
keep-factored design.

How it differs from the a/b/d/e family (all of which run on the Rust engine):
  * The engine's `RatFn` keeps the denominator FACTORED — a monomial `den` plus symbolic sum
    factors `den_opaque` kept unexpanded — and cancels only by exact construct-and-divide
    (`cancel_opaque`). It has no denominator-expansion primitive, no polynomial GCD, no
    factorization.
  * (c) instead expands the denominator into ONE polynomial and cancels each step with sympy
    `sp.cancel` (polynomial GCD division) / `sp.factor`. Because the Rust `RatFn` cannot hold an
    expanded-polynomial denominator, (c) reimplements the best-first loop in Python and uses the
    engine ONLY as a per-node move generator + numerator rewriter. Zero Rust changes.

What is held IDENTICAL to a/b/d (so the comparison is controlled): the move set
(`legal_moves` = schouten/momentum/momentum_sq), the anti-cycle discipline, the wall-clock/apps
budget, the mod-p soundness gate, and the display metric
(`_post.display_bracket_count`). The differences are: (1) the state carries a
fully expanded polynomial denominator; (2) each step uses `sp.cancel` for
polynomial GCD cancellation, while the engine cancels monomial factors and
attempts exact division by opaque polynomial factors; (3) the objective counts
the whole expression, including the denominator, so denominator cancellation
can reduce the search objective.

`factor_mode` (default "cancel") switches the per-step cancel between `sp.cancel` (polynomial
division), `sp.factor` (full factorization — slower, can hang on many-generator polys), and
`sp.factor_terms` (common-factor extraction after cancellation). `count_scope`
(default "full") switches the objective between the whole-expression count and
the numerator-only count for comparison with the default search.

Substrate note: this is the Python/sympy prototype (correct, bounded by max_apps/max_seconds; slow
on large 5/6pt inputs). If it proves worthwhile and Python is the bottleneck, the escalation is
`sp.Poly(expr, *gens)` with precomputed generators, then `python-flint` (FLINT `fmpz_mpoly` GCD /
factorization) — NOT a hand-rolled Rust polynomial-GCD engine. See `docs/GBFS_VARIANTS.md`.
"""
from __future__ import annotations

import heapq
import time
from typing import Optional

import sympy as sp

from ..result import AlgorithmResult
from ._post import count_brackets as _count_brackets, maybe_simplify
from .gbfs import GBFS, verify_equal, withheld
from ._post import judge_solved, post_process
from ..limits import DEFAULT_GBFS_C_CANCEL_MAX_TERMS

# `cancel_max_terms` limits the numerator and denominator sizes passed to SymPy.
# In the default mode, each candidate that passes the first size check calls
# `sp.cancel` after the Rust rewrite. Candidates that pass the next size check
# call it again to construct the deduplication key. Polynomial cancellation can
# be expensive, and SIGALRM cannot interrupt it from a web worker thread.
# `max_apps` limits the move count. `max_seconds` is checked between search
# operations and cannot interrupt an ongoing cancellation.


def _strip_opaque(e):
    """opaque_den(S) -> 1/S. Mirrors ampzip/verify/verifier.py:_strip_opaque_den so a sum
    denominator kept symbolic upstream is turned back into a real polynomial denominator that (c)
    then fully expands."""
    return e.replace(
        lambda x: x.is_Function and x.func.__name__ == "opaque_den" and len(x.args) == 1,
        lambda x: 1 / x.args[0],
    )


class GBFSVersionC(GBFS):
    name = "gbfs_c"
    use_den_aware = False          # custom Python loop; no engine value_fn

    def __init__(self, *args, cancel_max_terms: int = DEFAULT_GBFS_C_CANCEL_MAX_TERMS, **kwargs):
        super().__init__(*args, **kwargs)
        # A node or child whose numerator or denominator has more additive terms than this is
        # not fed to sympy's polynomial cancel (uninterruptible) and is skipped; every skip is
        # counted. 0 = no bound. This is (c)'s own knob (`--gbfs-c-cancel-max-terms`).
        self.cancel_max_terms = int(cancel_max_terms)

    def _too_many(self, k: int) -> bool:
        return self.cancel_max_terms > 0 and k > self.cancel_max_terms
    count_scope = "full"           # objective counts num+den (rewards den cancellation); "numerator" avail
    factor_mode = "cancel"         # per-step cancel: "cancel" | "factor" | "factor_terms"

    # --- per-step cancel -------------------------------------------------------------------- #
    def _cancel(self, expr, factor_mode):
        """The polynomial-division / factorization cancel step. Never raises (a sympy failure
        must not sink the search) — falls back to `sp.together`, then the raw expr."""
        try:
            if factor_mode == "factor":
                return sp.factor(expr)
            if factor_mode == "factor_terms":
                return sp.factor_terms(sp.cancel(expr))
            return sp.cancel(expr)          # default: polynomial GCD division
        except Exception:  # noqa: BLE001
            try:
                return sp.together(expr)
            except Exception:  # noqa: BLE001
                return expr

    # --- run -------------------------------------------------------------------------------- #
    def run(
        self,
        expr: sp.Expr,
        n_point: int,
        target: Optional[sp.Expr] = None,
        max_steps: int = 50,                 # ignored (not step-budgeted)
        per_step_timeout: float = 30.0,      # ignored
        sample_timeout: float = 120.0,       # ignored (SFT/CDS rollout budget; gbfs is bounded by max_seconds only)
        capture_derivation: bool = False,    # (c) emits no linear derivation chain
        simplify_mode: str = "none",
        sft_patience: int = 0,               # ignored
        cancel_event=None,                   # threading.Event; stop at next node/move
        **kwargs,                            # absorb other runner kwargs (factor_mode/count_scope override)
    ) -> AlgorithmResult:
        factor_mode = kwargs.get("factor_mode") or self.factor_mode
        count_scope = kwargs.get("count_scope") or self.count_scope
        t0 = time.time()
        start_brackets = _count_brackets(expr)
        # The wall-clock budget is max_seconds alone. sample_timeout is the SFT/CDS rollout
        # budget and is NOT folded in (it used to be, silently capping every gbfs run at
        # the 120 s default). Deployments bound this through
        # the ceiling applied to gbfs_max_seconds (web: WEBSVC_MAX_GBFS_MAX_SECONDS).
        budget_s = self.max_seconds
        eff_max_apps = self.max_apps
        skips = {"node_skips": 0, "child_term_skips": 0, "engine_error_skips": 0,
                 "term_limit_skips": 0, "overflow_skips": 0, "cancel_error_skips": 0}
        reason = "frontier_empty"

        # defaults so the final return is safe even if the search body raises early
        best_full = expr
        best_depth = 0
        apps = expanded = 0
        is_zero = stopped = False
        n_den_terms = 0
        expanded_start_brackets = start_brackets
        initial_too_large = False

        try:
            from ampzip.search.actions import legal_moves
            from ampzip.search.gbfs import _classify_skip, new_expr
            from ampzip.atoms import canonical_order as reorder_expr
            from ampzip.atoms import ab as _ab, sb as _sb
            loc = {"ab": _ab, "sb": _sb, "opaque_den": sp.Function("opaque_den")}

            def _score(num, den):
                """Frontier priority + best-tracking objective: (bracket count, numerator terms).
                count_scope=="full" counts the WHOLE expression (den included) — rewards
                cancelling the denominator; "numerator" isolates the expand+cancel variable."""
                whole = num if count_scope == "numerator" else num / den
                return (_count_brackets(whole), len(sp.Add.make_args(num)))

            def _canon(num, den):
                # Re-expand + canonicalize both sides every step: numerator flat so E.Expr parses
                # all brackets for legal_moves; denominator FULLY EXPANDED (the (c) invariant).
                # reorder_expr is REQUIRED — sympy keeps ab(2,1) distinct from ab(1,2), while the
                # engine emits canonical i<j; without it a numerator ab(1,2) silently fails to
                # cancel a denominator ab(2,1).
                return reorder_expr(sp.expand(num)), reorder_expr(sp.expand(den))

            # --- initial FULL expansion, including the denominator (the novel step) ---
            input_full = sp.sympify(str(expr), locals=loc)
            full0 = sp.together(_strip_opaque(input_full))
            e_num, e_den = _canon(*sp.fraction(full0))            # fully expanded, PRE-cancel
            n_den_terms = len(sp.Add.make_args(e_den))
            expanded_start_brackets = _count_brackets(e_num / e_den)   # report the blow-up

            # Expand everything first (denominator included), then cancel: do the initial
            # polynomial cancel now (bounded by the size gate so a monster expanded form can't
            # hang sympy). This is depth-0 of the search and, for a trivially-redundant input,
            # already yields the answer with no moves.
            if not (self._too_many(len(sp.Add.make_args(e_num))) or
                    self._too_many(len(sp.Add.make_args(e_den)))):
                num0, den0 = _canon(*sp.fraction(sp.together(
                    self._cancel(e_num / e_den, factor_mode))))
            else:
                initial_too_large = True
                num0, den0 = e_num, e_den

            # best seeded from the COMPACT input, so the expansion blow-up never "wins" and the
            # final maybe_simplify only ever runs on a cancelled/compact form (never a monster).
            best_num, best_den = None, None                       # None sentinel = best is `expr`
            in_num, _in_den = sp.fraction(sp.together(input_full))
            best_h = (start_brackets, len(sp.Add.make_args(in_num)))

            if num0 == 0 or (hasattr(num0, "is_zero") and num0.is_zero):
                is_zero = True
            else:
                # the cancelled initial node is itself a candidate best (expand+cancel, zero moves)
                h0 = _score(num0, den0)
                if h0 < best_h:
                    best_h, best_num, best_den, best_depth = h0, num0, den0, 0
                key0 = str(sp.cancel(num0 / den0))
                visited = {key0}
                counter = 0
                frontier = [(h0, counter, key0, num0, den0, 0)]
                counter += 1

                def _stop():
                    return cancel_event is not None and cancel_event.is_set()

                while frontier:
                    if apps >= eff_max_apps:
                        reason = "budget"
                        break
                    if time.time() - t0 > budget_s:
                        reason = "time"
                        break
                    if _stop():
                        stopped = True
                        reason = "cancelled"
                        break
                    _, _, _k, num_i, den_i, depth_i = heapq.heappop(frontier)
                    expanded += 1
                    try:
                        num_handle = new_expr(str(num_i), self.max_terms)
                    except Exception:  # noqa: BLE001
                        skips["engine_error_skips"] += 1
                        continue
                    if self._too_many(num_handle.num_terms()):
                        if depth_i == 0:
                            initial_too_large = True
                        skips["node_skips"] += 1
                        continue                                  # too big to expand from — skip
                    for sq, i, j, ts, ident, params in legal_moves(num_handle, n_point):
                        if apps >= eff_max_apps:
                            reason = "budget"
                            break
                        if time.time() - t0 > budget_s:
                            reason = "time"
                            break
                        if _stop():
                            stopped = True
                            reason = "cancelled"
                            break
                        try:
                            child_h = num_handle.apply(sq, i, j, ident, list(params),
                                                       n_point, term_slot=ts,
                                                       max_terms=self.max_terms)
                        except Exception as exc:  # noqa: BLE001  (degenerate identity / term cap)
                            skips[_classify_skip(exc) + "_skips"] += 1
                            continue
                        apps += 1
                        # pre-cancel size gate (cheap Rust) — do NOT feed a monster to sympy
                        if self._too_many(child_h.num_terms()):
                            skips["child_term_skips"] += 1
                            continue
                        try:
                            child_expr = sp.sympify(child_h.to_str(), locals=loc)
                            # the cancel step: child numerator (carrying any identity-introduced
                            # monomial denominator) over the current expanded denominator
                            child_full = self._cancel(child_expr / den_i, factor_mode)
                            c_num, c_den = sp.fraction(sp.together(child_full))
                            c_num, c_den = _canon(c_num, c_den)
                        except Exception:  # noqa: BLE001
                            skips["cancel_error_skips"] += 1
                            continue
                        # bound the pushed node so future cancels against it stay affordable
                        if (self._too_many(len(sp.Add.make_args(c_num))) or
                                self._too_many(len(sp.Add.make_args(c_den)))):
                            skips["child_term_skips"] += 1
                            continue
                        ck = str(sp.cancel(c_num / c_den))
                        if ck in visited:
                            continue
                        visited.add(ck)
                        h = _score(c_num, c_den)
                        if h < best_h:
                            best_h = h
                            best_num, best_den = c_num, c_den
                            best_depth = depth_i + 1
                            if child_h.is_zero() or c_num == 0:
                                is_zero = True
                        heapq.heappush(frontier, (h, counter, ck, c_num, c_den, depth_i + 1))
                        counter += 1
                    if stopped or reason in ("budget", "time"):
                        break
                else:
                    reason = "frontier_empty"

            if is_zero and best_num is None:
                best_full = sp.Integer(0)
            elif best_num is not None:
                best_full = best_num / best_den
            else:
                best_full = expr                                  # no improvement: return the input
        except Exception as e:  # noqa: BLE001
            return AlgorithmResult(
                algorithm=self.name, final_expr_str=str(expr),
                bracket_count_start=start_brackets, bracket_count_final=start_brackets,
                bracket_count_min=start_brackets, n_steps=0, solved=False, timed_out=False,
                duration_s=time.time() - t0,
                extra={"error": f"{type(e).__name__}: {e}", "stop_reason": "error", **self._params()},
            )

        duration = time.time() - t0
        post = post_process(str(best_full), simplify_mode, timeout_s=self.post_timeout_s)
        if post.bracket_count >= 0:
            final_str, bk_final = post.expr_str, post.bracket_count
        else:
            bk_final, final_str = _count_brackets(best_full), str(best_full)

        # --- soundness gate: the presented answer must be on-shell equal to the input ---
        best_str = str(best_full)
        sound, note = verify_equal(best_str, str(expr), n_point)
        if sound is not True:
            return withheld(self.name, expr, start_brackets, duration,
                            ("soundness check FAILED (gbfs_c output not on-shell equal to input)"
                             if sound is False else f"soundness check could not be completed ({note})")
                            + " — result withheld", best_str, stop_reason=reason, **self._params())
        target_equal, target_counts = None, {}
        if target is None:
            solved = bk_final < start_brackets
        else:
            target_equal, tnote = verify_equal(best_str, str(target), n_point)
            if self.target_criterion == "none":
                solved = target_equal is True
            else:
                by_count, target_counts = judge_solved(best_str, str(target), self.target_criterion,
                                                      max_terms=self.max_terms)
                solved = bool(by_count) and target_equal is True
            if tnote:
                target_counts["target_check_note"] = tnote

        try:
            raw_min = _count_brackets(sp.sympify(str(best_full),
                                                 locals={"ab": _ab, "sb": _sb,
                                                         "opaque_den": sp.Function("opaque_den")}))
        except Exception:  # noqa: BLE001
            raw_min = bk_final

        return AlgorithmResult(
            algorithm=self.name,
            final_expr_str=final_str,
            min_expr_str=final_str,
            bracket_count_start=start_brackets,
            bracket_count_final=bk_final,
            bracket_count_min=bk_final,
            n_steps=best_depth,
            solved=solved,
            timed_out=(reason == "time"),
            duration_s=duration,
            derivation=None,
            extra={
                "strategy": "expand_den_poly_cancel",
                "factor_mode": factor_mode,
                "count_scope": count_scope,
                "apps": apps,
                "expanded": expanded,
                "is_zero": is_zero,
                "stopped": stopped,
                "stop_reason": reason,
                **skips,
                "soundness_verified": True,
                "target_equal": target_equal,
                **target_counts,
                "raw_final_expr": str(best_full),
                "raw_bracket_count_min": raw_min,
                "raw_bracket_count_final": raw_min,
                "simplify_mode": simplify_mode,
                "post_process_timed_out": post.timed_out,
                "post_process_bounded": post.bounded,
                **self._params(),
                "cancel_max_terms": self.cancel_max_terms,
                "start_den_terms": n_den_terms,
                "expanded_start_brackets": expanded_start_brackets,
                "initial_expanded_too_large": initial_too_large,
            },
        )
