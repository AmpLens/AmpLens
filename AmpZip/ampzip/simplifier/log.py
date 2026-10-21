"""Timestamped JSONL logger for CLI invocations.

One CLI call → one log line, schema = CLIInvocation.to_dict(). Append-only.
Default location: `$AMPZIP_LOG_DIR/cli.jsonl`, else `<user log dir>/ampzip/cli.jsonl`
(`platformdirs.user_log_dir`; `~/.local/state/ampzip/log` without platformdirs).
"""

from __future__ import annotations
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .result import CLIInvocation



def default_log_dir() -> Path:
    """`$AMPZIP_LOG_DIR` when set, else the per-user log directory for ampzip."""
    env = os.environ.get("AMPZIP_LOG_DIR")
    if env:
        return Path(env).expanduser()
    try:
        from platformdirs import user_log_dir
        return Path(user_log_dir("ampzip", appauthor=False))
    except ImportError:
        state = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
        return Path(state) / "ampzip" / "log"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_invocation_id() -> str:
    return uuid.uuid4().hex[:12]


def resolve_log_path(log_dir: Optional[str] = None) -> Path:
    base = Path(log_dir) if log_dir else default_log_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / "cli.jsonl"


def append_invocation(record: CLIInvocation, log_dir: Optional[str] = None) -> Path:
    path = resolve_log_path(log_dir)
    line = json.dumps(record.to_dict(), default=str, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
