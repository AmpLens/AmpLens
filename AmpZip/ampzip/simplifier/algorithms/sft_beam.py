"""The SFT best-of-N backend.

It runs N rollouts of the SFT-greedy model, one greedy and N-1 at temperature T, and
returns the best result by bracket count. It reuses the backend cache of sft_greedy so the
model is loaded only once.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import sympy as sp

from ..limits import SFT_LIMITS
from ..result import AlgorithmResult
from .sft_greedy import (_get_backend, _default_checkpoint_path, _serializable_trajectory,
                         _sft_bracket_classes, _worker_lock)
from ._post import maybe_simplify


DEFAULT_N_ROLLOUTS = 5
DEFAULT_TEMPERATURE = 1.5


class SFTBeam:
    name = "sft_best_of_n"

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
        n_rollouts: int = DEFAULT_N_ROLLOUTS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_rollouts = n_rollouts
        self.temperature = temperature

    def run(
        self,
        expr: sp.Expr,
        n_point: int,
        target: Optional[sp.Expr] = None,
        max_steps: int = 50,
        per_step_timeout: float = 30.0,
        sample_timeout: float = 120.0,
        capture_derivation: bool = False,
        simplify_mode: str = "none",
        sft_patience: int = 0,
        cancel_event=None,          # a threading.Event checked between and within rollouts
    ) -> AlgorithmResult:
        import torch
        ckpt = self.checkpoint_path or _default_checkpoint_path(n_point)
        backend = _get_backend(n_point, ckpt, self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        L = SFT_LIMITS[n_point]

        eff_target = target if target is not None else sp.Integer(1)

        t0 = time.time()
        candidates = []

        should_stop = cancel_event.is_set if cancel_event is not None else None
        for rollout_idx in range(self.n_rollouts):
            # Rollout 0 always runs, stopping early internally through should_stop, so
            # `candidates` is never empty. After that, no more rollouts are launched once
            # cancelled.
            if rollout_idx > 0 and should_stop is not None and should_stop():
                break
            rollout_temp = 0.0 if rollout_idx == 0 else self.temperature

            # Serialise the shared sympy worker per rollout so other jobs can interleave
            # between rollouts, through _worker_lock of sft_greedy.
            with _worker_lock(n_point):
                r = backend.eval_mod.evaluate_sample(
                    backend.model,
                    initial_expr=expr,
                    target_expr=eff_target,
                    action_space=backend.action_space,
                    n_point=n_point,
                    max_terms=L.max_terms,
                    max_brackets_per_term=L.max_brackets_per_term,
                    max_steps=max_steps,
                    device=backend.device,
                    timeout_sec=per_step_timeout,
                    sample_timeout=sample_timeout,
                    verbose=False,
                    sample_idx=rollout_idx,
                    use_anticycle=True,
                    random_policy=False,
                    proper_stop=False,
                    reject_term_increase=None,
                    temperature=rollout_temp,
                    patience=sft_patience,
                    should_stop=should_stop,
                )
            candidates.append(r)

        duration = time.time() - t0

        # Pick the lexicographically lowest (min_brackets, min_terms, steps).
        best = min(candidates, key=lambda r: (
            r["min_brackets"], r.get("min_terms", float("inf")), r["steps"]
        ))

        bk_start = best["start_brackets"]
        bk_min_raw = best["min_brackets"]
        bk_final_raw = best["final_brackets"]
        raw_final_str = str(best["final_expr"])
        raw_min_str = str(best.get("min_expr", raw_final_str))

        ab_cls, sb_cls = _sft_bracket_classes()
        final_str, bk_final, _ = maybe_simplify(raw_final_str, simplify_mode, ab_cls, sb_cls)
        min_str, bk_min, _ = maybe_simplify(raw_min_str, simplify_mode, ab_cls, sb_cls)
        if bk_final < 0:
            bk_final = bk_final_raw
        if bk_min < 0:
            bk_min = bk_min_raw

        solved = bk_min < bk_start

        derivation = best.get("trajectory") if capture_derivation else None
        if derivation:
            from ampzip.search.actions import annotate_trajectory_rules
            annotate_trajectory_rules(derivation, n_point)

        candidate_summaries = [
            {
                "rollout_idx": i,
                "temperature": 0.0 if i == 0 else self.temperature,
                "min_brackets": c["min_brackets"],
                "final_brackets": c["final_brackets"],
                "steps": c["steps"],
                "final_expr": str(c["final_expr"]),
                "min_expr": str(c.get("min_expr", c["final_expr"])),
            }
            for i, c in enumerate(candidates)
        ]

        return AlgorithmResult(
            algorithm=self.name,
            final_expr_str=final_str,
            min_expr_str=min_str,
            bracket_count_start=bk_start,
            bracket_count_final=bk_final,
            bracket_count_min=bk_min,
            n_steps=best["steps"],
            solved=solved,
            timed_out=(best.get("timeout_count", 0) > 0
                       and best["steps"] < max_steps),
            duration_s=duration,
            derivation=_serializable_trajectory(derivation) if derivation else None,
            extra={
                "stopped": bool(cancel_event is not None and cancel_event.is_set()),
                "n_rollouts": self.n_rollouts,
                "temperature": self.temperature,
                "all_candidates": candidate_summaries,
                "solved_proper": best.get("solved_proper"),
                "timeout_count": best.get("timeout_count", 0),
                "raw_final_expr": raw_final_str,
                "raw_min_expr": raw_min_str,
                "raw_bracket_count_final": bk_final_raw,
                "raw_bracket_count_min": bk_min_raw,
                "simplify_mode": simplify_mode,
                "sft_patience": sft_patience,
                "patience_triggered_per_rollout": [
                    c.get("patience_triggered", False) for c in candidates
                ],
                "min_terms_at_min": best.get("min_terms"),
            },
        )
