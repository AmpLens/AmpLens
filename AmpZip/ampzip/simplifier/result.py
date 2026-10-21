"""Result dataclasses. Serializable to JSON for log records."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ValidationIssue:
    tag: str            # e.g. "LIMIT_ATOM_UNKNOWN"
    message: str
    detail: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    ok: bool
    inferred_n_point: Optional[int]
    canonical_expr_str: Optional[str]   # parsed expression in SymPy notation
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class AlgorithmResult:
    algorithm: str               # e.g. "sft_greedy"
    final_expr_str: str          # final expression reported by the backend
    bracket_count_start: int
    bracket_count_final: int
    bracket_count_min: int
    n_steps: int
    solved: bool                 # vs supplied target if any
    timed_out: bool
    duration_s: float
    min_expr_str: str = ""       # best expression selected by the backend
    derivation: Optional[List[Dict[str, Any]]] = None   # per-step actions, optional
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CLIInvocation:
    """Full record of one CLI call. JSONL-serializable."""
    timestamp_utc: str
    invocation_id: str
    cwd: str
    cli_args: Dict[str, Any]
    raw_input: str
    validation: ValidationResult
    results: List[AlgorithmResult] = field(default_factory=list)
    error: Optional[str] = None
    # Wall-clock breakdown (seconds). Compare against per-algorithm
    # AlgorithmResult.duration_s to see import/load overhead.
    validation_duration_s: float = 0.0
    total_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "invocation_id": self.invocation_id,
            "cwd": self.cwd,
            "cli_args": self.cli_args,
            "raw_input": self.raw_input,
            "validation": self.validation.to_dict() if self.validation else None,
            "validation_duration_s": self.validation_duration_s,
            "results": [r.to_dict() for r in self.results],
            "total_duration_s": self.total_duration_s,
            "error": self.error,
        }
