"""Algorithm protocol — every backend implements this."""

from __future__ import annotations
from typing import Protocol, Optional, runtime_checkable

import sympy as sp

from ..result import AlgorithmResult


@runtime_checkable
class Algorithm(Protocol):
    name: str

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
        cancel_event=None,   # threading.Event for cooperative cancel (gbfs + SFT)
    ) -> AlgorithmResult: ...
