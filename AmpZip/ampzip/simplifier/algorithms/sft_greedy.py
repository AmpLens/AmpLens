"""SFT-greedy backend.

It wraps `RL_amplitudes_claude/src/eval_sft_{n}pt.py` with `action_space_{n}pt.py` and
`train_sft_transformer.py` to perform a single greedy rollout on a user-supplied sympy
expression.

The model and the persistent sympy worker are constructed lazily on the first call and
reused across calls within the same process. atexit shuts them down.
"""

from __future__ import annotations
import atexit
import importlib
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

import sympy as sp
import torch

from .. import _paths
from ..limits import SFT_LIMITS
from ..result import AlgorithmResult
from ._post import maybe_simplify

_paths.assert_rl_repo_present()  # the research packages are fetched on the user's machine


def _sft_bracket_classes():
    """The ab/sb classes the SFT rollout uses, the ones the validator uses too."""
    from ..validate import _get_bracket_classes
    return _get_bracket_classes()


# The class-name suffix per n_point in action_space_*pt.py
_ACTION_SPACE_CLASS = {
    4: "TermSpecificActionSpace",
    5: "TermSpecificActionSpace5pt",
    6: "TermSpecificActionSpace6pt",
}

class _LoadedBackend:
    """One per n_point, caching the model, the action space and the eval module."""

    def __init__(self, n_point: int, checkpoint_path: Path, device: str):
        self.n_point = n_point
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Imports for this n_point
        self.eval_mod = importlib.import_module(f"eval_sft_{n_point}pt")
        self.action_space_mod = importlib.import_module(f"action_space_{n_point}pt")
        self.train_mod = importlib.import_module("train_sft_transformer")

        L = SFT_LIMITS[n_point]
        self._limits = L

        action_space_cls = getattr(
            self.action_space_mod, _ACTION_SPACE_CLASS[n_point]
        )
        self.action_space = action_space_cls(
            n_point=n_point,
            max_terms=L.max_terms_action,
            max_brackets=L.max_brackets,
        )

        # The model dimensions, as in eval_sft_4pt.main
        bracket_feature_dim = 2 + 2 * n_point
        term_feature_dim = 2 + L.max_brackets_per_term * bracket_feature_dim + 1
        n_term_slots = 1 + L.max_terms_action

        self.model = self.train_mod.TransformerPolicySFT(
            max_terms=L.max_terms,
            term_feature_dim=term_feature_dim,
            n_actions=self.action_space.n_actions,
            max_brackets=L.max_brackets,
            actions_per_bracket=n_term_slots * self.action_space.identities_per_bracket,
            embed_dim=64,        # the SFT default
            num_heads=4,
            num_layers=3,
            ff_dim=128,
            dropout=0.1,
            features_dim=128,
            global_feature_dim=2,
        ).to(device)

        # The published checkpoints hold only a tensor state_dict and ints.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Spawn the persistent sympy worker
        self.eval_mod.init_worker(n_point)
        atexit.register(self._shutdown)

    def _shutdown(self):
        try:
            if hasattr(self.eval_mod, "shutdown_worker"):
                self.eval_mod.shutdown_worker()
        except Exception:
            pass


# The process-wide cache, one backend per (n_point, checkpoint_str)
_BACKENDS: Dict[str, _LoadedBackend] = {}


def _get_backend(n_point: int, checkpoint_path: Path, device: str) -> _LoadedBackend:
    key = f"{n_point}:{checkpoint_path}"
    if key not in _BACKENDS:
        _BACKENDS[key] = _LoadedBackend(n_point, checkpoint_path, device)
    return _BACKENDS[key]


# One lock per n-point. The persistent sympy worker in eval_sft_{n}pt is a module global
# shared by every job of that n-point, greedy and beam alike, and its
# `apply_action_with_timeout` assumes a single caller when draining the response queue and
# restarting the worker after a timeout. Holding this lock for an entire `evaluate_sample`
# rollout serialises worker access across jobs. A new request waits until a cancelled rollout
# releases the lock. The lock is keyed by `n_point`, matching the worker's scope, and is
# shared by checkpoints with the same `n_point`.
_WORKER_LOCKS: Dict[int, threading.Lock] = {}
_WORKER_LOCKS_GUARD = threading.Lock()


def _worker_lock(n_point: int) -> threading.Lock:
    with _WORKER_LOCKS_GUARD:
        lk = _WORKER_LOCKS.get(n_point)
        if lk is None:
            lk = threading.Lock()
            _WORKER_LOCKS[n_point] = lk
        return lk


class SFTGreedy:
    name = "sft_greedy"

    def __init__(self, checkpoint_path: Optional[Path] = None, device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path  # None means the default for the n_point
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
        cancel_event=None,          # a threading.Event checked at the next step boundary
    ) -> AlgorithmResult:
        ckpt = self.checkpoint_path or _default_checkpoint_path(n_point)
        backend = _get_backend(n_point, ckpt, self.device)
        L = SFT_LIMITS[n_point]

        # evaluate_sample stops early when `count_brackets(current) <= count_brackets(target)`.
        # Without a target, the 0-bracket sentinel sp.Integer(1) makes the rollout run to
        # max_steps or to an anti-cycle dead end. `min_brackets` still tracks the best state
        # along the way.
        eff_target = target if target is not None else sp.Integer(1)

        t0 = time.time()
        # Serialise the shared sympy worker for the whole rollout through _worker_lock.
        with _worker_lock(n_point):
            rollout = backend.eval_mod.evaluate_sample(
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
                sample_idx=0,
                use_anticycle=True,
                random_policy=False,
                proper_stop=False,
                reject_term_increase=None,
                temperature=0.0,           # greedy
                patience=sft_patience,
                should_stop=(cancel_event.is_set if cancel_event is not None else None),
            )
        duration = time.time() - t0

        bk_start = rollout["start_brackets"]
        bk_min_raw = rollout["min_brackets"]
        bk_final_raw = rollout["final_brackets"]
        raw_final_str = str(rollout["final_expr"])
        raw_min_str = str(rollout.get("min_expr", raw_final_str))

        ab_cls, sb_cls = _sft_bracket_classes()
        final_str, bk_final, _ = maybe_simplify(raw_final_str, simplify_mode, ab_cls, sb_cls)
        min_str, bk_min, _ = maybe_simplify(raw_min_str, simplify_mode, ab_cls, sb_cls)
        # If simplification failed with -1, the rollout's count is used.
        if bk_final < 0:
            bk_final = bk_final_raw
        if bk_min < 0:
            bk_min = bk_min_raw

        solved = bk_min < bk_start  # strict progress at the minimum

        derivation = rollout.get("trajectory") if capture_derivation else None
        if derivation:
            from ampzip.search.actions import annotate_trajectory_rules
            annotate_trajectory_rules(derivation, n_point)

        return AlgorithmResult(
            algorithm=self.name,
            final_expr_str=final_str,
            min_expr_str=min_str,
            bracket_count_start=bk_start,
            bracket_count_final=bk_final,
            bracket_count_min=bk_min,
            n_steps=rollout["steps"],
            solved=solved,
            timed_out=(rollout.get("timeout_count", 0) > 0
                       and rollout["steps"] < max_steps),
            duration_s=duration,
            derivation=_serializable_trajectory(derivation) if derivation else None,
            extra={
                "stopped": bool(cancel_event is not None and cancel_event.is_set()),
                "solved_proper": rollout.get("solved_proper"),
                "solved_source_rel": rollout.get("solved_source_rel"),
                "retry_count": rollout.get("retry_count", 0),
                "timeout_count": rollout.get("timeout_count", 0),
                "start_terms": rollout.get("start_terms"),
                "final_terms": rollout.get("final_terms"),
                "raw_final_expr": raw_final_str,
                "raw_min_expr": raw_min_str,
                "raw_bracket_count_final": bk_final_raw,
                "raw_bracket_count_min": bk_min_raw,
                "simplify_mode": simplify_mode,
                "sft_patience": sft_patience,
                "patience_triggered": rollout.get("patience_triggered", False),
                "min_terms_at_min": rollout.get("min_terms"),
            },
        )


def _default_checkpoint_path(n_point: int) -> Path:
    return _paths.RL_REPO / "models" / f"sft_{n_point}pt_500k" / "best_model.pt"


def _serializable_trajectory(traj) -> Optional[list]:
    if traj is None:
        return None
    out = []
    for step in traj:
        rec: Dict[str, Any] = {}
        for k, v in step.items() if isinstance(step, dict) else []:
            if isinstance(v, (int, float, str, bool, type(None))):
                rec[k] = v
            else:
                rec[k] = str(v)
        out.append(rec)
    return out
