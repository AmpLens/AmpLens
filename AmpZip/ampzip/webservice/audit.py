"""Append-only audit log for the exposed service.

Each lifecycle event is recorded as one JSONL entry. A submission records the
submitter, canonical expression and request parameters after clamping. Completion
records a summary of the outcome.
Mirrors the shape of `simplifier/log.py` but lives in a separate file,
`webservice.jsonl` in the same log directory (`$WEBSVC_LOG_DIR`, else `$AMPZIP_LOG_DIR`,
else the per-user log dir), and is keyed by the verified email.

Events:
  - "submit"   : request accepted + enqueued (after auth/safety/validate/clamp)
  - "finish"   : job reached a terminal state (done | error | timeout | cancelled)
  - "reject"   : request refused at one of the stages listed in `audit_reject`

Logging never raises into the request path — failures here are swallowed.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..simplifier.log import default_log_dir

_LOG_NAME = "webservice.jsonl"
_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _log_path() -> Path:
    env = os.environ.get("WEBSVC_LOG_DIR")
    base = Path(env).expanduser() if env else default_log_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / _LOG_NAME


def _write(event: Dict[str, Any]) -> None:
    try:
        line = json.dumps(event, default=str, ensure_ascii=False)
        with _lock, open(_log_path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # never break the request path on a logging failure
        pass


def audit_submit(*, email: str, job_id: str, n_point: Optional[int],
                 canonical_expr: Optional[str], request: Dict[str, Any]) -> None:
    _write({
        "ts": _utc_now_iso(),
        "event": "submit",
        "email": email,
        "job_id": job_id,
        "n_point": n_point,
        "canonical_expr": canonical_expr,
        "request": request,
    })


def audit_reject(*, email: Optional[str], stage: str, detail: Any,
                 raw_input: Optional[str] = None) -> None:
    """Record a rejection at auth, ratelimit, safety, atomize, validate, algorithms
    or queue."""
    _write({
        "ts": _utc_now_iso(),
        "event": "reject",
        "email": email,
        "stage": stage,
        "detail": detail,
        # Keep a truncated copy of what was attempted (useful for abuse triage).
        "raw_input": (raw_input[:500] if raw_input else None),
    })


def audit_finish(*, email: str, job_id: str, status: str,
                 duration_s: float, results: Any = None,
                 error: Optional[str] = None) -> None:
    # Store a compact per-algorithm summary, not the full result blobs.
    summary = None
    if isinstance(results, list):
        summary = [
            {
                "algorithm": r.get("algorithm"),
                "solved": r.get("solved"),
                "bracket_count_start": r.get("bracket_count_start"),
                "bracket_count_min": r.get("bracket_count_min"),
                "n_steps": r.get("n_steps"),
                "duration_s": r.get("duration_s"),
            }
            for r in results if isinstance(r, dict)
        ]
    _write({
        "ts": _utc_now_iso(),
        "event": "finish",
        "email": email,
        "job_id": job_id,
        "status": status,
        "duration_s": duration_s,
        "summary": summary,
        "error": error,
    })
