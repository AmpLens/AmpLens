"""The gbfs_c variant, which expands everything and cancels by polynomial division.

gbfs_c fully expands the whole expression, denominator included, into one rational function,
then cancels at every search step by polynomial division or factorization. It is a comparison
variant, the inverse of the keep-factored design of the engine.

The engine's `RatFn` keeps the denominator factored, a monomial `den` plus unexpanded sum
factors in `den_opaque`, and cancels only by exact construction and division in
`cancel_opaque`. It has no denominator expansion, no polynomial GCD and no factorization.
gbfs_c instead expands the denominator into one polynomial and cancels each step with
`sp.cancel`, polynomial GCD division, or `sp.factor`. Because `RatFn` cannot hold an
expanded-polynomial denominator, gbfs_c reimplements the best-first loop in Python and uses
the engine as a per-node move generator and numerator rewriter.

The move set of `legal_moves`, the anti-cycle discipline, the time and move budgets, the
finite-field test and the display count of `_post.display_bracket_count` are those of the
other variants, so the comparison is controlled. The differences are that the state carries a
fully expanded polynomial denominator, that each step cancels by polynomial GCD where the
engine cancels monomial factors and attempts exact division by sum factors, and that the
objective counts the whole expression, denominator included, so denominator cancellation
can reduce it.

`factor_mode`, "cancel" by default, switches the per-step cancel between `sp.cancel`,
`sp.factor`, which is slower and can hang on polynomials with many generators, and
`sp.factor_terms`, which extracts common factors after cancellation. `count_scope`, "full" by
default, switches the objective between the whole-expression count and the numerator-only
count for comparison with the default search.

This is the Python and sympy prototype, bounded by max_apps and max_seconds and slow on
large 5 and 6pt inputs. If Python proves to be the bottleneck, the next step is
`sp.Poly(expr, *gens)` with precomputed generators, then FLINT's `fmpz_mpoly` through
python-flint. `docs/GBFS_VARIANTS.md` has the details.
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
    """opaque_den(S) -> 1/S, as in _strip_opaque_den of ampzip/verify/verifier.py, so a sum
    denominator kept symbolic upstream becomes a polynomial denominator that gbfs_c expands."""
    return e.replace(
        lambda x: x.is_Function and x.func.__name__ == "opaque_den" and len(x.args) == 1,
        lambda x: 1 / x.args[0],
    )


class GBFSVersionC(GBFS):
    name = "gbfs_c"
    use_den_aware = False          # a Python loop with no engine value_fn

    def __init__(self, *args, cancel_max_terms: int = DEFAULT_GBFS_C_CANCEL_MAX_TERMS, **kwargs):
        super().__init__(*args, **kwargs)
        # A node or child whose numerator or denominator has more additive terms than this is
        # skipped before sympy's polynomial cancel, which cannot be interrupted, and every skip
        # is counted. 0 means no bound. This is gbfs_c's own parameter, `--gbfs-c-cancel-max-terms`.
        self.cancel_max_terms = int(cancel_max_terms)

    def _too_many(self, k: int) -> bool:
        return self.cancel_max_terms > 0 and k > self.cancel_max_terms
    count_scope = "full"           # the objective counts numerator and denominator; "numerator" counts the numerator only
    factor_mode = "cancel"         # the per-step cancel, "cancel", "factor" or "factor_terms"

    # --- per-step cancel -------------------------------------------------------------------- #
    def _cancel(self, expr, factor_mode):
        """The polynomial-division or factorization cancel step. A sympy failure falls back to
        `sp.together`, then to the raw expr."""
        try:
            if factor_mode == "factor":
                return sp.factor(expr)
            if factor_mode == "factor_terms":
                return sp.factor_terms(sp.cancel(expr))
            return sp.cancel(expr)          # the default, polynomial GCD division
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
        max_steps: int = 50,                 # ignored, since gbfs has no step budget
        per_step_timeout: float = 30.0,      # ignored
        sample_timeout: float = 120.0,       # ignored, the SFT and CDS rollout budget
        capture_derivation: bool = False,    # gbfs_c emits no derivation
        simplify_mode: str = "none",
        sft_patience: int = 0,               # ignored
        cancel_event=None,                   # a threading.Event checked at the next node or move
        **kwargs,                            # other runner kwargs, which may override factor_mode and count_scope
    ) -> AlgorithmResult:
        factor_mode = kwargs.get("factor_mode") or self.factor_mode
        count_scope = kwargs.get("count_scope") or self.count_scope
        t0 = time.time()
        start_brackets = _count_brackets(expr)
        # The wall-clock budget is max_seconds alone. sample_timeout is the SFT/CDS rollout
        # budget. Deployments bound this through
        # the limit on gbfs_max_seconds, WEBSVC_MAX_GBFS_MAX_SECONDS on the web.
        budget_s = self.max_seconds
        eff_max_apps = self.max_apps
        skips = {"node_skips": 0, "child_term_skips": 0, "engine_error_skips": 0,
                 "term_limit_skips": 0, "overflow_skips": 0, "cancel_error_skips": 0}
        reason = "frontier_empty"

        # The defaults for the final return when the search body raises early.
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
                """The frontier priority and best-tracking objective, (bracket count, numerator
                terms). count_scope "full" counts the whole expression, denominator included,
                which rewards cancelling the denominator. "numerator" isolates the effect of
                expanding and cancelling."""
                whole = num if count_scope == "numerator" else num / den
                return (_count_brackets(whole), len(sp.Add.make_args(num)))

            def _canon(num, den):
                # Re-expand and canonicalise both sides every step, the numerator flat so E.Expr
                # parses all brackets for legal_moves and the denominator fully expanded, the
                # gbfs_c invariant. reorder_expr is required because sympy keeps ab(2,1) distinct
                # from ab(1,2) while the engine emits canonical i<j. Without it a numerator ab(1,2)
                # silently fails to cancel a denominator ab(2,1).
                return reorder_expr(sp.expand(num)), reorder_expr(sp.expand(den))

            # --- the initial full expansion, denominator included ---
            input_full = sp.sympify(str(expr), locals=loc)
            full0 = sp.together(_strip_opaque(input_full))
            e_num, e_den = _canon(*sp.fraction(full0))            # fully expanded, before cancel
            n_den_terms = len(sp.Add.make_args(e_den))
            expanded_start_brackets = _count_brackets(e_num / e_den)   # reports the blow-up

            # Expand everything first, denominator included, then cancel. The initial polynomial
            # cancel runs now, bounded by the size cap so an expanded form does not hang sympy.
            # This is depth 0 of the search and, for a trivially redundant input, already
            # yields the answer with no moves.
            if not (self._too_many(len(sp.Add.make_args(e_num))) or
                    self._too_many(len(sp.Add.make_args(e_den)))):
                num0, den0 = _canon(*sp.fraction(sp.together(
                    self._cancel(e_num / e_den, factor_mode))))
            else:
                initial_too_large = True
                num0, den0 = e_num, e_den

            # best is seeded from the compact input, so the expansion does not win and the final
            # maybe_simplify only runs on a cancelled or compact form.
            best_num, best_den = None, None                       # the None sentinel means best is `expr`
            in_num, _in_den = sp.fraction(sp.together(input_full))
            best_h = (start_brackets, len(sp.Add.make_args(in_num)))

            if num0 == 0 or (hasattr(num0, "is_zero") and num0.is_zero):
                is_zero = True
            else:
                # The cancelled initial node is itself a candidate best, reached with zero moves.
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
                        continue                                  # too big to expand from
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
                        except Exception as exc:  # noqa: BLE001  a degenerate identity or the term cap
                            skips[_classify_skip(exc) + "_skips"] += 1
                            continue
                        apps += 1
                        # pre-cancel size gate, cheap in Rust
                        if self._too_many(child_h.num_terms()):
                            skips["child_term_skips"] += 1
                            continue
                        try:
                            child_expr = sp.sympify(child_h.to_str(), locals=loc)
                            # The cancel step, the child numerator, carrying any monomial denominator
                            # the identity introduced, over the current expanded denominator.
                            child_full = self._cancel(child_expr / den_i, factor_mode)
                            c_num, c_den = sp.fraction(sp.together(child_full))
                            c_num, c_den = _canon(c_num, c_den)
                        except Exception:  # noqa: BLE001
                            skips["cancel_error_skips"] += 1
                            continue
                        # Bound the pushed node so future cancels against it stay affordable.
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
                best_full = expr                                  # no improvement, so the input is returned
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

        # --- the presented answer must equal the input on shell ---
        best_str = str(best_full)
        sound, note = verify_equal(best_str, str(expr), n_point)
        if sound is not True:
            return withheld(self.name, expr, start_brackets, duration,
                            ("the finite-field test failed, the gbfs_c output differing from the input on shell"
                             if sound is False else f"the finite-field test could not be completed, {note}")
                            + ", so the result is withheld", best_str, stop_reason=reason, **self._params())
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
