"""Locate the optional learned backends (the SFT and CDS packages) and make them importable.

The training-free search needs nothing from here. Only the `sft_*` and `cds_*` backends import
the two upstream packages (the rollout code and the spinor-helicity environment), which are not
part of this distribution: they are fetched onto the user's machine by `ampzip fetch-backends`
and found again through this module. Resolution order:

1. ``$AMPZIP_BACKENDS`` — the directory that holds the fetched ``RL_amplitudes_claude`` tree
   (the tree itself, i.e. the directory with ``src/`` and ``spinorhelicity/`` inside).
2. ``$RL_AMPLITUDES_REPO`` — the older name of the same setting, kept so existing
   environments keep working.
3. ``external/RL_amplitudes_claude`` next to the ``ampzip`` package — a source checkout that
   keeps its own copy.
4. ``<user data dir>/ampzip/backends/RL_amplitudes_claude`` — where ``fetch-backends`` puts it
   (``platformdirs.user_data_dir``; ``~/.local/share/ampzip`` without platformdirs).

Importing this module has no effect when none of these exists; `backends_available()` then
reports False and the learned backends register themselves as unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def _user_data_dir() -> Path:
    try:
        from platformdirs import user_data_dir
        return Path(user_data_dir("ampzip", appauthor=False))
    except ImportError:  # platformdirs is a declared dependency; this is only a fallback
        return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "ampzip"


_SOURCE_CHECKOUT_COPY = Path(__file__).resolve().parents[2] / "external" / "RL_amplitudes_claude"
FETCH_DIR = _user_data_dir() / "backends"
_FETCHED_COPY = FETCH_DIR / "RL_amplitudes_claude"


def _is_repo(p: Path) -> bool:
    return (p / "src").is_dir() and (p / "spinorhelicity").is_dir()


def candidates() -> List[Path]:
    """Every location consulted, in order."""
    out: List[Path] = []
    for key in ("AMPZIP_BACKENDS", "RL_AMPLITUDES_REPO"):
        v = os.environ.get(key)
        if v:
            out.append(Path(v).expanduser())
    out.append(_SOURCE_CHECKOUT_COPY)
    out.append(_FETCHED_COPY)
    return out


def _resolve() -> Optional[Path]:
    for c in candidates():
        if _is_repo(c):
            return c
    return None


_FOUND = _resolve()
# The path reported in messages and used for checkpoints: the found tree, or else the place
# `fetch-backends` would create.
RL_REPO: Path = _FOUND if _FOUND is not None else _FETCHED_COPY


def backends_available() -> bool:
    return _FOUND is not None


def ensure_importable() -> Path:
    """Put the backend tree's ``src/`` and ``spinorhelicity/`` on ``sys.path``; raise if absent."""
    assert_rl_repo_present()
    for sub in ("src", "spinorhelicity"):
        p = str(RL_REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return RL_REPO


def assert_rl_repo_present() -> None:
    if _FOUND is None:
        looked = "\n".join(f"  - {c}" for c in candidates())
        raise FileNotFoundError(
            "The learned backends (sft_*, cds_*) need the upstream packages, which are not part "
            "of ampzip. Run `ampzip fetch-backends` (add --cds for the CDS weights), or point "
            f"$AMPZIP_BACKENDS at an existing RL_amplitudes_claude tree. Looked in:\n{looked}"
        )


if _FOUND is not None:
    ensure_importable()
