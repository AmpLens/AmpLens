"""gbfs backend — model-free greedy best-first search over the fast Rust engine.

No neural network: best-first over `ampzip_engine` ranked by bracket count, exploring
legal identity moves until the budget is spent, returning the smallest form found. By default
the run is **target-free**. With a target and a target criterion the
search stops at the first node that meets the criterion (the paper's target-stopped runs).

Every move is value-preserving by construction; we additionally **gate** the returned answer
through the exact mod-p verifier (`ampzip.verify.equal`) before presenting it. A result whose
verification fails or cannot be completed is withheld, never shown as unverified.

Every way the search can stop or skip work is reported in `extra`: `stop_reason` (goal /
budget / time / frontier_empty / cancelled / patience), the per-cause skip counters, and the
parameters that bounded the run. `timed_out` is exactly `stop_reason == "time"`.

Imports of the engine are lazy (inside `run`) so the runner still imports even where
`ampzip_engine` (the Rust extension) isn't built — gbfs just errors at call time there.
"""
from __future__ import annotations

import time
from typing import Optional

import sympy as sp

from ..result import AlgorithmResult
from ._post import (count_brackets as _count_brackets, count_brackets_bounded, gate_rng,
                    judge_solved, post_process)
from ..limits import (DEFAULT_GBFS_MAX_APPS, DEFAULT_GBFS_MAX_SECONDS, DEFAULT_GBFS_MAX_TERMS,
                      DEFAULT_GBFS_ORDERING, DEFAULT_GBFS_TARGET_CRITERION, DEFAULT_GBFS_THREADS,
                      DEFAULT_POST_TIMEOUT_S, GBFS_ORDERINGS, GBFS_TARGET_CRITERIA)


def verify_equal(a: str, b: str, n_point: int):
    """Finite-field equality of two expression strings with a reproducible, input-derived
    random stream. Returns `(verdict, note)`: verdict True/False, or None when the check could
    not be completed (note says why). An inconclusive sampling run is retried once with a
    second seed before giving up."""
    from ampzip.verify import equal
    for attempt in ("", ":retry"):
        try:
            return equal(a, b, n_point, rng=gate_rng(a + attempt, b)) is True, None
        except RuntimeError as exc:          # inconclusive: too many singular sample points
            note = f"inconclusive: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - unsupported input / evaluation failure
            return None, f"{type(exc).__name__}: {exc}"
    return None, note


def withheld(name: str, expr, start_brackets: int, duration: float, reason: str, rejected: str,
             **extra) -> AlgorithmResult:
    """The result returned when the answer cannot be presented: the input, unchanged."""
    return AlgorithmResult(
        algorithm=name, final_expr_str=str(expr), min_expr_str=str(expr),
        bracket_count_start=start_brackets, bracket_count_final=start_brackets,
        bracket_count_min=start_brackets, n_steps=0, solved=False, timed_out=False,
        duration_s=duration,
        extra={"error": reason, "rejected_expr": rejected, "soundness_verified": False, **extra},
    )


class GBFS:
    name = "model_free_gbfs"

    # Subclasses override these to swap the simplicity objective. The defaults reproduce the
    # paper's gbfs exactly: factored numerator bracket count, no denominator-aware bias.
    objective_fn = None          # None -> engine's factored-bracket `heuristic`
    use_den_aware = False        # class default for `den_aware` (constructor overrides)

    def __init__(self, max_apps: int = DEFAULT_GBFS_MAX_APPS,
                 max_seconds: float = DEFAULT_GBFS_MAX_SECONDS,
                 max_terms: int = DEFAULT_GBFS_MAX_TERMS,
                 den_aware: Optional[bool] = None,
                 ordering: str = DEFAULT_GBFS_ORDERING,
                 threads: int = DEFAULT_GBFS_THREADS,
                 target_criterion: str = DEFAULT_GBFS_TARGET_CRITERION,
                 audit_steps: bool = False,
                 post_timeout_s: Optional[float] = DEFAULT_POST_TIMEOUT_S,
                 **_ignored):
        self.max_apps = max_apps
        self.max_seconds = max_seconds
        # Per-move numerator term cap (guards a combinatorial substitution blow-up). Passed to
        # every apply() so the search skips only the exploding move and keeps going; each skip
        # is counted (`term_limit_skips`). Inherited by all gbfs variants.
        self.max_terms = max_terms
        self.den_aware = self.use_den_aware if den_aware is None else bool(den_aware)
        if ordering not in GBFS_ORDERINGS:
            raise ValueError(f"gbfs ordering must be one of {GBFS_ORDERINGS}, got {ordering!r}")
        if target_criterion not in GBFS_TARGET_CRITERIA:
            raise ValueError(f"gbfs target criterion must be one of {GBFS_TARGET_CRITERIA}, "
                             f"got {target_criterion!r}")
        if self.den_aware and ordering == "raw":
            raise ValueError("gbfs: den-aware ordering and raw ordering are both frontier "
                             "orderings; choose one")
        self.ordering = ordering
        self.threads = int(threads)
        self.target_criterion = target_criterion
        self.audit_steps = bool(audit_steps)
        self.post_timeout_s = post_timeout_s

    # --- frontier ordering ------------------------------------------------------------- #
    def _value_fn(self):
        """The `value_fn` handed to the engine search: None for the paper's factored ordering,
        the raw numerator count for gbfsraw, or the denominator-aware bias when asked for."""
        if self.ordering == "raw":
            return lambda h: h.bracket_count()
        if self.den_aware:
            from ampzip.search.value_den import make_den_aware_value
            return make_den_aware_value()
        return None

    def _goal_kwargs(self, target) -> dict:
        """Target-stopped search: the raw-count goal(s) for the chosen criterion."""
        if target is None or self.target_criterion == "none":
            return {}
        from ampzip.search.gbfs import new_expr
        t = new_expr(str(target), self.max_terms)
        kw = {"goal_raw_brackets": t.bracket_count()}
        if self.target_criterion == "brackets-and-terms":
            kw["goal_raw_terms"] = t.num_terms()
        return kw

    def _params(self) -> dict:
        return {
            "max_apps": self.max_apps, "max_seconds": self.max_seconds, "max_terms": self.max_terms,
            "ordering": self.ordering, "den_aware": self.den_aware, "threads": self.threads,
            "target_criterion": self.target_criterion, "audit_steps": self.audit_steps,
            "post_timeout_s": self.post_timeout_s,
        }

    def run(
        self,
        expr: sp.Expr,
        n_point: int,
        target: Optional[sp.Expr] = None,
        max_steps: int = 50,                 # ignored (gbfs is not step-budgeted)
        per_step_timeout: float = 30.0,      # ignored
        sample_timeout: float = 120.0,       # ignored (SFT/CDS rollout budget; gbfs is bounded by max_seconds only)
        capture_derivation: bool = False,
        simplify_mode: str = "none",
        sft_patience: int = 0,               # ignored
        cancel_event=None,                   # threading.Event; checked during node expansion
        **kwargs,                            # absorb any other runner kwargs
    ) -> AlgorithmResult:
        start_brackets = _count_brackets(expr)
        t0 = time.time()
        # The wall-clock budget is max_seconds alone. sample_timeout is the SFT/CDS rollout
        # budget and is NOT folded in (it used to be, silently capping every gbfs run at
        # the 120 s default). Deployments bound this through
        # the ceiling applied to gbfs_max_seconds (web: WEBSVC_MAX_GBFS_MAX_SECONDS).
        budget_s = self.max_seconds
        expr_str = str(expr)
        target_str = None if target is None else str(target)

        try:
            from ampzip.search.gbfs import new_expr, simplify as gbfs_simplify
            # Parsed under the same term cap as the search, so no engine-internal default binds
            # below --gbfs-max-terms.
            raw_num_start = new_expr(expr_str, self.max_terms).bracket_count()
            res = gbfs_simplify(expr_str, n_point, max_apps=self.max_apps,
                                goal_brackets=None, max_seconds=budget_s,
                                value_fn=self._value_fn(), objective_fn=self.objective_fn,
                                should_stop=(cancel_event.is_set
                                             if cancel_event is not None else None),
                                max_terms=self.max_terms, threads=self.threads,
                                **self._goal_kwargs(target))
        except Exception as e:  # noqa: BLE001
            return AlgorithmResult(
                algorithm=self.name, final_expr_str=str(expr),
                bracket_count_start=start_brackets, bracket_count_final=start_brackets,
                bracket_count_min=start_brackets, n_steps=0, solved=False, timed_out=False,
                duration_s=time.time() - t0,
                extra={"error": f"{type(e).__name__}: {e}", "stop_reason": "error", **self._params()},
            )

        duration = time.time() - t0
        best_str = res["best_expr"]
        post = post_process(best_str, simplify_mode, timeout_s=self.post_timeout_s)
        if post.bracket_count >= 0:
            final_str, bk_final = post.expr_str, post.bracket_count
        else:
            final_str, bk_final = best_str, res["best_brackets"]

        # --- soundness gate: the presented answer must be on-shell equal to the input ---
        sound, note = verify_equal(best_str, expr_str, n_point)
        if sound is not True:
            return withheld(self.name, expr, start_brackets, duration,
                            ("soundness check FAILED (gbfs output not on-shell equal to input)"
                             if sound is False else f"soundness check could not be completed ({note})")
                            + " — result withheld", best_str,
                            stop_reason=res["stop_reason"], apps=res["apps"], **self._params())

        # --- solved: target-relative when a target is given ---
        target_counts = {}
        target_equal = None
        if target is None:
            solved = bk_final < start_brackets
        else:
            target_equal, tnote = verify_equal(best_str, target_str, n_point)
            if self.target_criterion == "none":
                solved = target_equal is True
            else:
                by_count, target_counts = judge_solved(best_str, target_str, self.target_criterion,
                                                      max_terms=self.max_terms)
                solved = bool(by_count) and target_equal is True
            if tnote:
                target_counts["target_check_note"] = tnote

        # Raw (pre-post-process) bracket count of the search's best node, in the sympy display
        # metric used for the path rows. The front-end matches derivation rows against this to
        # locate the proving prefix, so it must equal the final derivation row's brackets_after.
        raw_min = count_brackets_bounded(best_str, timeout_s=self.post_timeout_s)
        if raw_min < 0:
            raw_min = res["best_brackets"]

        derivation = None
        audit = None
        path_exprs = res.get("path_exprs", [])
        path_moves = res.get("path_moves", [])
        if capture_derivation:
            # Rows in the schema the front-end / SFT backends use (expr_after / brackets_before /
            # brackets_after / identity / bracket / term_slot), zipping path_exprs with path_moves.
            from ampzip.search.actions import move_rule

            def _bc(s):
                return count_brackets_bounded(s, timeout_s=self.post_timeout_s)

            derivation = []
            for i, mv in enumerate(path_moves):
                if i + 1 >= len(path_exprs):
                    break
                sq, li, lj, ts, ident, params = mv
                rule_lhs, rule_rhs = move_rule(sq, li, lj, ident, params, n_point)
                row = {
                    "bracket": f"{'sb' if sq else 'ab'}({li},{lj})",
                    "term_slot": ts,
                    "is_whole_expression": (ts == 0),
                    "identity": ident,
                    "params": list(params),
                    "brackets_before": _bc(path_exprs[i]),
                    "brackets_after": _bc(path_exprs[i + 1]),
                    "expr_after": path_exprs[i + 1],
                    "rule_lhs": rule_lhs,
                }
                if rule_rhs is not None:
                    row["rule_rhs"] = rule_rhs
                derivation.append(row)
        if self.audit_steps:
            # Every step of the derivation through the same finite-field check as the endpoint.
            bad, notes = [], []
            for i in range(len(path_exprs) - 1):
                ok, anote = verify_equal(path_exprs[i], path_exprs[i + 1], n_point)
                if ok is not True:
                    bad.append(i)
                    notes.append(anote or "mismatch")
            audit = {"steps": max(len(path_exprs) - 1, 0), "step_bad": bad, "notes": notes}
            if bad:
                return withheld(self.name, expr, start_brackets, duration,
                                f"derivation audit FAILED at step(s) {bad} — result withheld",
                                best_str, audit=audit, stop_reason=res["stop_reason"],
                                **self._params())

        return AlgorithmResult(
            algorithm=self.name,
            final_expr_str=final_str,
            min_expr_str=final_str,                 # gbfs returns one (the smallest) form
            bracket_count_start=start_brackets,
            bracket_count_final=bk_final,
            bracket_count_min=bk_final,
            n_steps=res["depth"],
            solved=solved,
            timed_out=(res["stop_reason"] == "time"),
            duration_s=duration,
            derivation=derivation,
            extra={
                **self._params(),
                "apps": res["apps"],
                "expanded": res["expanded"],
                "is_zero": res["is_zero"],
                "goal_reached": res["goal_reached"],
                "stop_reason": res["stop_reason"],
                "stopped": res.get("stopped", False),   # user-cancelled mid-search
                "term_limit_skips": res["term_limit_skips"],
                "overflow_skips": res["overflow_skips"],
                "engine_error_skips": res["engine_error_skips"],
                "soundness_verified": True,
                "target_equal": target_equal,
                **target_counts,
                "engine_best_brackets": res["best_brackets"],
                # Engine raw numerator bracket counts for the input and the expression selected
                # by the search.
                "raw_numerator_brackets_start": raw_num_start,
                "raw_numerator_brackets_min": res["best_raw_brackets"],
                "raw_bracket_count_min": raw_min,
                "raw_bracket_count_final": raw_min,
                "raw_final_expr": best_str,
                "simplify_mode": simplify_mode,
                "post_process_timed_out": post.timed_out,
                "post_process_bounded": post.bounded,
                "audit": audit,
            },
        )
