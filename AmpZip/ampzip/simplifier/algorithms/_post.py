"""Shared post-processing helpers used by all algorithm backends.

It covers bracket counting that works with either sympy `ab`/`sb` class, the SFT-side
`spin_helicity_env.ab` or the CDS-side `bracket_env.ab`, which register under the same name,
and the output post-processing controlled by `--simplify-mode`, none by default. For
example, expand rewrites
   `(2*ab(1,3)*ab(2,4) - ab(1,4)*ab(2,3))/ab(1,4)` as
   `2*ab(1,3)*ab(2,4)/ab(1,4) - ab(2,3)`.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
from typing import NamedTuple, Optional, Tuple

import sympy as sp

from ..limits import DEFAULT_POST_TIMEOUT_S


def _count_brackets_raw(expr) -> int:
    """Count ab/sb occurrences in the expression tree. Positive integer powers
    multiply the count of the base by the exponent. Other powers contribute the
    count of the base once.

    Bracket functions are identified by class name to support both SFT and CDS
    atoms. A factor present in several additive terms is counted separately in
    each term. Counts reported to the user are computed by
    `display_bracket_count`."""
    if expr is None:
        return 0
    if expr.is_Atom:
        return 0
    name = type(expr).__name__
    if name in ("ab", "sb"):
        return 1
    if expr.is_Pow:
        base, exp = expr.args
        sub = _count_brackets_raw(base)
        if exp.is_Integer and int(exp) > 0:
            return int(exp) * sub
        return sub
    return sum(_count_brackets_raw(a) for a in expr.args)


def count_brackets(expr) -> int:
    """Internal bracket count after extracting common factors.

    Applies `sp.factor_terms` throughout the expression, without polynomial
    expansion, then counts ab/sb occurrences. Within each sum, a factor common
    to all terms is counted once:
        ab(1,2)*ab(2,3)*ab(3,4) + ab(1,2)*ab(2,3)*ab(4,5)  ->  4

    If `factor_terms` fails, the count is computed from the original expression.

    This count includes denominator brackets because it traverses the whole
    expression tree. The engine's `factored_bracket_count` counts only the
    numerator. This function is used internally for gbfs_c's ordering and for
    gbfs_f's and CDS's candidate selection. The runner recomputes reported counts
    with `display_bracket_count`, which counts the numerator only. `_count_brackets_raw`
    keeps the plain occurrence count for callers that need it.
    """
    if expr is None:
        return 0
    try:
        expr = sp.factor_terms(expr)
    except Exception:
        pass
    return _count_brackets_raw(expr)


def _display_count_sympy(expr_str: str) -> int:
    """`display_bracket_count` by sympy, for expressions the engine refuses. The numerator of the
    combined fraction is expanded as the engine expands it, `(⟨12⟩+⟨34⟩)**2` being three terms
    and six brackets, the monomial factor common to all terms is taken out, and every ab/sb
    power is counted with multiplicity. Raises on anything it cannot parse."""
    from ampzip.atoms import ab, sb
    e = sp.sympify(str(expr_str), locals={"ab": ab, "sb": sb, "opaque_den": sp.Function("opaque_den")})
    num, _den = sp.fraction(sp.together(e))
    return _count_brackets_raw(sp.factor_terms(sp.expand(num)))


def display_bracket_count(expr_str: str, max_terms: int = 0) -> int:
    """Report-side bracket count: the factored numerator bracket count, the default search's
    objective. Numerator factors are counted with multiplicity after the monomial factor common
    to every term is taken out, and denominator brackets are not counted. The README's
    `ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))` counts 3, `ab(1,2)/ab(3,4)**5` counts 1,
    `ab(1,2)*ab(2,3)*ab(3,4) + ab(1,2)*ab(2,3)*ab(4,5)` counts 4.

    The engine computes it as `Expr.factored_bracket_count`. An expression the engine refuses,
    one with a non-integer rational coefficient or `I`, is counted by sympy under the same
    rule. Returns -1 when neither can count it, which callers treat as unknown.
    It differs from `count_brackets`, the backends' internal whole-tree count."""
    s = str(expr_str or "").strip()
    if not s:
        return -1
    try:
        from ampzip.search.gbfs import new_expr
        return int(new_expr(s, max_terms).factored_bracket_count())
    except Exception:  # noqa: BLE001  an engine limit, so the sympy rule applies
        pass
    try:
        return int(_display_count_sympy(s))
    except Exception:  # noqa: BLE001
        return -1


SIMPLIFY_MODES = ("none", "cancel", "expand", "simplify")


def _apply_mode(expr, mode: str):
    """The post-processing step itself, in process. A sympy failure returns the input
    unchanged."""
    if mode == "cancel":
        try:
            expr = sp.cancel(expr)
        except Exception:
            pass
    elif mode == "expand":
        try:
            expr = sp.cancel(expr)
            expr = sp.expand(expr)
        except Exception:
            pass
    elif mode == "simplify":
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
    # mode "none" does nothing
    return expr


class PostResult(NamedTuple):
    expr_str: str
    bracket_count: int        # the factored count of count_brackets, -1 when the step failed
    # Time limit expired or helper exited without a complete reply.
    # In either case, expr_str contains the original input.
    timed_out: bool
    bounded: bool             # the step ran in the killable helper process
    error: Optional[str]


def _pump_lines(stream, replies: "queue.Queue[Optional[bytes]]") -> None:
    """Forward the helper's reply lines to `replies`, then None at EOF. It runs in a daemon
    thread per helper process, since `select()` cannot wait on a pipe on Windows."""
    try:
        for line in iter(stream.readline, b""):
            replies.put(line.rstrip(b"\r\n"))
    except (OSError, ValueError):
        pass
    replies.put(None)


class _PostWorker:
    """One persistent helper process (`_post_worker`) per parent process, lock-serialised.

    It is started lazily and restarted after a kill. A per-call child would cost a sympy
    import of about 1.5 s per expression, which the per-term drivers call up to 64 times per
    run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._replies = None

    def _spawn(self):
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "ampzip.simplifier.algorithms._post_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=dict(os.environ, PYTHONWARNINGS="ignore"))
        # A fresh queue per process, so a late reply from a killed helper is never read as the
        # answer to the next request.
        self._replies = queue.Queue()
        threading.Thread(target=_pump_lines, args=(self._proc.stdout, self._replies),
                         daemon=True).start()

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None

    def _readline(self, timeout_s: float) -> Optional[bytes]:
        """Read one reply line from the child, or return None on timeout or EOF."""
        if timeout_s <= 0:
            return None
        try:
            return self._replies.get(timeout=timeout_s)   # None means the child died
        except queue.Empty:
            return None

    def request(self, payload: dict, timeout_s: float) -> dict:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._kill()
                self._spawn()
            try:
                self._proc.stdin.write((json.dumps(payload) + "\n").encode())
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._kill()
                return {"error": f"post-process worker unavailable: {exc}"}
            line = self._readline(timeout_s)
            if line is None:
                self._kill()
                return {"timed_out": True}
            try:
                return json.loads(line)
            except ValueError as exc:
                self._kill()
                return {"error": f"post-process worker reply unreadable: {exc}"}


_WORKER = _PostWorker()


def post_process(expr_in, mode: str = "none", *, timeout_s: Optional[float] = DEFAULT_POST_TIMEOUT_S,
                 ab_cls=None, sb_cls=None) -> PostResult:
    """Post-process a backend output and compute its internal bracket count. Calls
    with a time limit use a helper process because SymPy work cannot be
    interrupted safely within a thread. The parent waits at most `timeout_s` for
    the helper's reply. On timeout, the original string is returned with
    `timed_out=True` and `bracket_count=-1`. A helper that exits without a complete
    reply is also reported as timed out.

    With `timeout_s=None` or 0, the work runs in the current process without a
    time limit (`bounded=False`). A pipe write failure or an invalid JSON response
    also causes execution in the current process with `bounded=False`. Errors while
    starting the worker process are raised to the caller."""
    if expr_in is None or expr_in == "":
        return PostResult("", 0, False, False, None)
    if mode not in SIMPLIFY_MODES:
        raise ValueError(f"unknown simplify mode: {mode!r}; choose from {SIMPLIFY_MODES}")
    raw = expr_in if isinstance(expr_in, str) else str(expr_in)
    if timeout_s:
        reply = _WORKER.request({"op": "simplify", "expr": raw, "mode": mode}, float(timeout_s))
        if reply.get("timed_out"):
            return PostResult(raw, -1, True, True, "post-processing exceeded the time bound")
        if "error" not in reply:
            return PostResult(reply["expr"], int(reply["count"]), False, True, None)
        if not str(reply["error"]).startswith("post-process worker"):
            return PostResult(raw, -1, False, True, reply["error"])     # a sympy failure
        # A pipe-write error or invalid JSON reply causes the work to run
        # in the current process.
    s_out, bk, _ = _maybe_simplify_inprocess(raw, mode, ab_cls, sb_cls)
    return PostResult(s_out, bk, False, False, None if bk >= 0 else "post-processing failed")


def count_brackets_bounded(expr_str: str, *, timeout_s: Optional[float] = DEFAULT_POST_TIMEOUT_S) -> int:
    """`count_brackets` of a sympy string in the helper process, -1 on timeout or failure."""
    if not timeout_s:
        try:
            from ampzip.atoms import ab, sb
            return count_brackets(sp.sympify(expr_str, locals={"ab": ab, "sb": sb,
                                                              "opaque_den": sp.Function("opaque_den")}))
        except Exception:  # noqa: BLE001
            return -1
    reply = _WORKER.request({"op": "count", "expr": expr_str}, float(timeout_s))
    return int(reply["count"]) if "count" in reply else -1


def gate_rng(a: str, b: str) -> random.Random:
    """Seeded stream for the finite-field gate, derived from the two expressions being
    compared, so a verdict can be reproduced exactly without any parameter."""
    h = hashlib.blake2b((a + "\0" + b).encode(), digest_size=16).digest()
    return random.Random(int.from_bytes(h, "big"))


def judge_solved(final_str: str, target_str: str, criterion: str,
                 max_terms: int = 0) -> Tuple[Optional[bool], dict]:
    """Target-relative success on the engine's raw numerator counts.

    `brackets` asks for a raw numerator bracket count at most the target's, and
    `brackets-and-terms` for the additive term count too. Returns `(verdict, counts)`, with
    verdict None when either side cannot be parsed by the engine, in which case counts carries
    the error. Both sides are parsed under `max_terms`, the search's own cap, so no
    engine-internal default binds below it."""
    try:
        from ampzip.search.gbfs import new_expr
        f, t = new_expr(final_str, max_terms), new_expr(target_str, max_terms)
        counts = {"final_raw_brackets": f.bracket_count(), "final_terms": f.num_terms(),
                  "target_raw_brackets": t.bracket_count(), "target_terms": t.num_terms()}
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"}
    ok = counts["final_raw_brackets"] <= counts["target_raw_brackets"]
    if criterion == "brackets-and-terms":
        ok = ok and counts["final_terms"] <= counts["target_terms"]
    return bool(ok), counts


def maybe_simplify(
    expr_in,
    mode: str = "none",
    ab_cls=None,
    sb_cls=None,
    simplify_timeout_s: Optional[float] = DEFAULT_POST_TIMEOUT_S,
) -> Tuple[str, int, object]:
    """Post-process a backend's output, return (str, bracket_count, sympy_expr).

    It runs through `post_process`, bounded in the helper process, when `simplify_timeout_s`
    is set, and the sympy expression is then None. With `simplify_timeout_s` None or 0 the
    work runs in process without a bound and the sympy expression is returned too.
    """
    if simplify_timeout_s:
        r = post_process(expr_in, mode, timeout_s=simplify_timeout_s, ab_cls=ab_cls, sb_cls=sb_cls)
        return r.expr_str, r.bracket_count, None
    return _maybe_simplify_inprocess(expr_in, mode, ab_cls, sb_cls)


def _maybe_simplify_inprocess(expr_in, mode, ab_cls=None, sb_cls=None) -> Tuple[str, int, object]:
    """Re-parse if needed, apply `mode`-specific post-processing, return (str, bracket_count, sympy_expr).

    `mode` none applies no post-processing. cancel runs `sp.cancel`, polynomial gcd
    cancellation, which never blows up the term count. expand runs `sp.cancel` then
    `sp.expand`, which distributes denominators into numerators and catches partial-fraction
    splits like `(num + A*d)/d → num/d + A`. simplify runs `sp.simplify`, which can find
    much shorter forms but may hang on big expressions.

    `expr_in` may be a `sp.Expr` or a `str`. On any failure the raw string is returned with
    bracket_count -1.
    """
    if expr_in is None or expr_in == "":
        return "", 0, None
    if mode not in SIMPLIFY_MODES:
        raise ValueError(f"unknown simplify mode: {mode!r}; choose from {SIMPLIFY_MODES}")

    try:
        if isinstance(expr_in, str):
            local = {}
            if ab_cls is not None:
                local["ab"] = ab_cls
            if sb_cls is not None:
                local["sb"] = sb_cls
            expr = sp.sympify(expr_in, locals=local) if local else sp.sympify(expr_in)
        else:
            expr = expr_in

        expr = _apply_mode(expr, mode)
        return str(expr), count_brackets(expr), expr
    except Exception:
        s = expr_in if isinstance(expr_in, str) else str(expr_in)
        return s, -1, None
