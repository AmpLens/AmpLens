"""Algorithm orchestrator. Runs N backends on one input and collects results."""

from __future__ import annotations
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Callable, List, Optional

import sympy as sp

import importlib

from .algorithms.base import Algorithm
from .result import AlgorithmResult
from .limits import (
    DEFAULT_SFT_BEAM_SIZE,
    DEFAULT_SFT_BEAM_TEMPERATURE,
    DEFAULT_CDS_BEAM_SIZE,
    DEFAULT_CDS_TEMPERATURE,
    DEFAULT_CDS_NUCLEUS_P,
    DEFAULT_CDS_LENGTH_PENALTY,
    DEFAULT_SIMPLIFY_MODE,
    DEFAULT_SFT_PATIENCE,
    DEFAULT_GBFS_MAX_APPS,
    DEFAULT_GBFS_MAX_SECONDS,
    DEFAULT_GBFS_MAX_TERMS,
    DEFAULT_GBFS_ORDERING,
    DEFAULT_GBFS_TARGET_CRITERION,
    DEFAULT_GBFS_THREADS,
    DEFAULT_GBFS_SPLIT_MAX_TERMS,
    DEFAULT_GBFS_C_CANCEL_MAX_TERMS,
    DEFAULT_POST_TIMEOUT_S,
    DEFAULT_ALGO_GRACE_S,
    DEFAULT_MAX_PARALLEL_ALGOS,
)


# Registry of backends: name -> (module, class). Backends are imported on first use, so the
# training-free gbfs family runs on an install without torch or the SFT/CDS packages; the two
# learned backends are optional extras and report themselves unavailable instead of breaking
# the import of this module.
_REGISTRY = {
    "sft_greedy": (".algorithms.sft_greedy", "SFTGreedy"),
    "sft_best_of_n": (".algorithms.sft_beam", "SFTBeam"),
    "cds_oneshot": (".algorithms.cds", "CDS"),
    "model_free_gbfs": (".algorithms.gbfs", "GBFS"),
    "gbfs_c": (".algorithms.gbfs_c", "GBFSVersionC"),
    "gbfs_d": (".algorithms.gbfs_d", "GBFSVersionD"),
    "gbfs_e": (".algorithms.gbfs_e", "GBFSVersionE"),
    "gbfs_f": (".algorithms.gbfs_f", "GBFSVersionF"),
}

# Backends that need the optional extras (torch, and the SFT/CDS packages fetched separately).
OPTIONAL_BACKENDS = ("sft_greedy", "sft_best_of_n", "cds_oneshot")

def list_algorithms() -> List[str]:
    return list(_REGISTRY.keys())


def is_known(name: str) -> bool:
    return name in _REGISTRY


def backend_class(name: str):
    """Import and return the backend class; ImportError if its extras are missing."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown algorithm '{name}'; available: {list_algorithms()}")
    module, cls = _REGISTRY[name]
    return getattr(importlib.import_module(module, __package__), cls)


def available_algorithms() -> List[str]:
    """The registered backends whose imports succeed on this install."""
    out = []
    for name in _REGISTRY:
        try:
            backend_class(name)
            out.append(name)
        except Exception:  # noqa: BLE001  (missing torch / upstream packages)
            continue
    return out


def _construct(name: str, **algo_kwargs) -> Algorithm:
    return backend_class(name)(**algo_kwargs)


def get_algorithm(name: str, **kwargs) -> Algorithm:
    return _construct(name, **kwargs)


def _algo_kwargs_for(
    name: str,
    sft_beam_size: int,
    sft_beam_temperature: float,
    cds_beam_size: int,
    cds_temperature: float,
    cds_nucleus_p: float,
    cds_length_penalty: float,
    gbfs_max_apps: int,
    gbfs_max_seconds: float,
    gbfs_max_terms: int,
    gbfs_den_aware: bool = False,
    gbfs_ordering: str = DEFAULT_GBFS_ORDERING,
    gbfs_threads: int = DEFAULT_GBFS_THREADS,
    gbfs_target_criterion: str = DEFAULT_GBFS_TARGET_CRITERION,
    gbfs_audit_steps: bool = False,
    gbfs_split_max_terms: int = DEFAULT_GBFS_SPLIT_MAX_TERMS,
    gbfs_c_cancel_max_terms: int = DEFAULT_GBFS_C_CANCEL_MAX_TERMS,
    post_timeout_s: Optional[float] = DEFAULT_POST_TIMEOUT_S,
) -> dict:
    """Pick the construction-time kwargs each backend cares about."""
    if name == "sft_greedy":
        return {}
    if name == "sft_best_of_n":
        return {
            "n_rollouts": sft_beam_size,
            "temperature": sft_beam_temperature,
        }
    if name == "cds_oneshot":
        return {
            "beam_size": cds_beam_size,
            "temperature": cds_temperature,
            "nucleus_p": cds_nucleus_p,
            "length_penalty": cds_length_penalty,
        }
    if name in GBFS_FAMILY:
        kw = {
            "max_apps": gbfs_max_apps,
            "max_seconds": gbfs_max_seconds,
            "max_terms": gbfs_max_terms,
            "den_aware": gbfs_den_aware,
            "ordering": gbfs_ordering,
            "threads": gbfs_threads,
            "target_criterion": gbfs_target_criterion,
            "audit_steps": gbfs_audit_steps,
            "post_timeout_s": post_timeout_s,
        }
        if name in ("gbfs_e", "gbfs_f"):
            kw["split_max_terms"] = gbfs_split_max_terms
        if name == "gbfs_c":
            kw["cancel_max_terms"] = gbfs_c_cancel_max_terms
        return kw
    return {}


# The gbfs family — all share the fast Rust engine and can consume opaque_den. Public (imported
# by cli.py and webservice/app.py for the opaque-denominator router, so there is ONE source of
# truth for "which algorithms are gbfs"); `_GBFS_FAMILY` is kept as a backward-compatible alias.
GBFS_FAMILY = ("model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f")
_GBFS_FAMILY = GBFS_FAMILY


def is_gbfs(name: str) -> bool:
    """Membership test for the search family."""
    return name in GBFS_FAMILY

# Backends that check `cancel_event.is_set()` for cooperative cancellation. CDS
# accepts the argument through `**kwargs` but does not use it.
_CANCELLABLE = GBFS_FAMILY + ("sft_greedy", "sft_best_of_n")

# The grace added on top of each algorithm's OWN wall-clock budget before the parallel
# collector stops waiting for it is a parameter of run_all (`algo_grace_s`; CLI
# `--algo-grace-seconds`, web `WEBSVC_ALGO_GRACE_S`). Each backend already self-bounds, so the
# collector deadline is a safety net; a backend that overruns it still has its result
# delivered (marked `overran`) if it finishes before run_all returns.

# How long the parallel collector waits per slice while an algorithm is still QUEUED in
# the pool (more algorithms than workers). Only a liveness poll — the queue wait is never
# charged against the algorithm's own budget — so it just needs to be short enough that a
# start is noticed promptly and long enough not to spin.
_START_POLL_S = 0.25


class _CombinedStop:
    """Stop signal handed to one algorithm as its `cancel_event`. `is_set()` is True when
    EITHER the shared job-level cancel (the web 'Stop' button) OR this algorithm's own
    per-run event (set on its individual wall-clock timeout) is set — so job-cancel
    broadcasts to every algorithm while a single-algorithm timeout stops only that one.
    Backends only ever call `.is_set()` on their cancel_event (see gbfs.py, sft_*.py)."""
    __slots__ = ("_job", "_own")

    def __init__(self, job_cancel, own_event: threading.Event):
        self._job = job_cancel
        self._own = own_event

    def is_set(self) -> bool:
        return (self._job is not None and self._job.is_set()) or self._own.is_set()

    def set(self) -> None:
        self._own.set()

    def reason(self) -> Optional[str]:
        if self._job is not None and self._job.is_set():
            return "job_cancel"
        if self._own.is_set():
            return "collector_timeout"
        return None


# Backends whose `solved` without a target is exactly "fewer brackets than the input"; the
# report-side recount re-derives it for them. The others carry a further condition (CDS's
# numerical match, the per-term drivers' no-progress guard), so the recount can only withdraw.
_COUNT_ONLY_SOLVED = frozenset({"model_free_gbfs", "gbfs_c", "gbfs_d", "sft_greedy", "sft_best_of_n"})


def _report_counts(r: AlgorithmResult, start_str: str, has_target: bool, max_terms: int) -> None:
    """Replace the reported bracket counts by the display count (`display_bracket_count`: the
    factored numerator count) for the input, the final and the minimal form and every
    derivation row. The backends' own counts (whole-tree, denominator included) move to
    `extra["backend_bracket_count_*"]`; their search and candidate selection are untouched.
    Without a target, `solved` follows the new counts (see `_COUNT_ONLY_SOLVED`). An
    uncountable form gives -1 and is never compared."""
    from .algorithms._post import display_bracket_count

    if not r.final_expr_str:
        return                                   # an error placeholder: nothing to count
    r.extra["backend_bracket_count_start"] = r.bracket_count_start
    r.extra["backend_bracket_count_final"] = r.bracket_count_final
    r.extra["backend_bracket_count_min"] = r.bracket_count_min
    start = display_bracket_count(start_str, max_terms)
    final = display_bracket_count(r.final_expr_str, max_terms)
    mn = display_bracket_count(r.min_expr_str, max_terms) if r.min_expr_str else final
    r.bracket_count_start, r.bracket_count_final, r.bracket_count_min = start, final, mn

    rows = r.derivation or []
    if rows and all(isinstance(row, dict) and row.get("expr_after") for row in rows):
        prev = start
        for row in rows:
            after = display_bracket_count(row["expr_after"], max_terms)
            row["brackets_before"], row["brackets_after"] = prev, after
            if after >= 0:
                prev = after
    # The backend's own final and minimal forms before post-processing, in the same unit:
    # `raw_final_expr` / `raw_min_expr` when the backend saved them (SFT keeps both; the
    # search family and CDS save one form, which is both), else the last derivation row;
    # a key that cannot be restated is dropped so no stale whole-tree number remains.
    last_row = rows[-1]["brackets_after"] if rows and rows[-1].get("brackets_after", -1) >= 0 else None
    raw_final = raw_min = None
    if r.extra.get("raw_final_expr"):
        raw_final = display_bracket_count(r.extra["raw_final_expr"], max_terms)
    if r.extra.get("raw_min_expr"):
        raw_min = display_bracket_count(r.extra["raw_min_expr"], max_terms)
    if raw_final is None:
        raw_final = last_row
    if raw_min is None:
        raw_min = raw_final
    for key, val in (("raw_bracket_count_final", raw_final), ("raw_bracket_count_min", raw_min)):
        if key in r.extra:
            if val is not None and val >= 0:
                r.extra[key] = val
            else:
                del r.extra[key]

    if not has_target and start >= 0 and mn >= 0:
        improved = mn < start
        if r.algorithm in _COUNT_ONLY_SOLVED:
            r.solved = improved
        else:
            r.solved = bool(r.solved) and improved


def _judge_target(r: AlgorithmResult, target_str: str, n_point: int, criterion: str,
                  max_terms: int) -> None:
    """One target-relative `solved` for every backend, the rule the search family applies
    inside its own run: the simplest form reported (`min_expr`, else `final_expr`) must equal
    the target over finite fields and meet `criterion` on the engine's raw numerator counts
    (`brackets`: bracket count at or below the target's; `brackets-and-terms`: term count too;
    `none`: equality alone). The search family is left as judged by its run (same functions,
    same fields); the learned backends get the same verdict here.
    `extra` carries `target_equal`, the counts and any note."""
    if is_gbfs(r.algorithm) or not r.final_expr_str:
        return
    from .algorithms._post import judge_solved
    from .algorithms.gbfs import verify_equal
    form = r.min_expr_str or r.final_expr_str
    equal, note = verify_equal(form, target_str, n_point)
    counts: dict = {}
    by_count = True
    if criterion != "none":
        by_count, counts = judge_solved(form, target_str, criterion, max_terms=max_terms)
    r.solved = (equal is True) and (by_count is True)
    r.extra["target_equal"] = equal
    r.extra["target_criterion"] = criterion
    r.extra.update(counts)
    if note:
        r.extra["target_check_note"] = note
    if by_count is None:
        r.extra["target_check_note"] = (r.extra.get("target_check_note") or
                                        f"target count check unavailable: {counts.get('error', '?')}")


def run_all(
    algorithm_names: List[str],
    expr: sp.Expr,
    n_point: int,
    target: Optional[sp.Expr] = None,
    max_steps: int = 50,
    per_step_timeout: float = 30.0,
    sample_timeout: float = 120.0,
    capture_derivation: bool = False,
    simplify_mode: str = DEFAULT_SIMPLIFY_MODE,
    sft_patience: int = DEFAULT_SFT_PATIENCE,
    # SFT-beam
    sft_beam_size: int = DEFAULT_SFT_BEAM_SIZE,
    sft_beam_temperature: float = DEFAULT_SFT_BEAM_TEMPERATURE,
    # CDS
    cds_beam_size: int = DEFAULT_CDS_BEAM_SIZE,
    cds_temperature: float = DEFAULT_CDS_TEMPERATURE,
    cds_nucleus_p: float = DEFAULT_CDS_NUCLEUS_P,
    cds_length_penalty: float = DEFAULT_CDS_LENGTH_PENALTY,
    # gbfs
    gbfs_max_apps: int = DEFAULT_GBFS_MAX_APPS,
    gbfs_max_seconds: float = DEFAULT_GBFS_MAX_SECONDS,
    gbfs_max_terms: int = DEFAULT_GBFS_MAX_TERMS,
    gbfs_den_aware: bool = False,
    gbfs_ordering: str = DEFAULT_GBFS_ORDERING,
    gbfs_threads: int = DEFAULT_GBFS_THREADS,
    gbfs_target_criterion: str = DEFAULT_GBFS_TARGET_CRITERION,
    gbfs_audit_steps: bool = False,
    gbfs_split_max_terms: int = DEFAULT_GBFS_SPLIT_MAX_TERMS,
    gbfs_c_cancel_max_terms: int = DEFAULT_GBFS_C_CANCEL_MAX_TERMS,
    # bound on the sympy post-processing / recounts of every backend's output
    post_timeout_s: Optional[float] = DEFAULT_POST_TIMEOUT_S,
    # grace on top of a backend's own budget before the parallel collector stops waiting
    algo_grace_s: float = DEFAULT_ALGO_GRACE_S,
    # Cooperative cancellation via threading.Event. GBFS checks it between search
    # operations. SFT checks it between rollout steps and, for best-of-N, between
    # rollouts. CDS decoding does not use this event.
    cancel_event=None,
    # ORIGINAL pre-atomize input string. Used by the per-term drivers (gbfs_e / gbfs_f) to split
    # by the original additive terms BEFORE the atomizer combines them over a common denominator;
    # every other backend ignores it. None -> the driver falls back to an expr-level split.
    raw_input: Optional[str] = None,
    # At most `max_parallel` backends run concurrently. Additional backends wait for
    # a worker, and this queueing time does not count toward their individual time
    # budgets. When all fit in the pool and finish normally, elapsed time is
    # approximately the longest backend runtime, plus dispatch and collection
    # overhead. The default, 1, runs them sequentially in input order. The web
    # service passes `CONFIG.max_parallel_algos`.
    max_parallel: int = DEFAULT_MAX_PARALLEL_ALGOS,
    # Optional progressive sink: on_result(index, AlgorithmResult) is invoked exactly once
    # per algorithm the instant it finishes (natural completion or per-algo timeout), so a
    # caller can surface first-done results before the whole batch returns. `index` is the
    # algorithm's position in `algorithm_names`. Exceptions in the sink are swallowed.
    on_result: Optional[Callable[[int, AlgorithmResult], None]] = None,
) -> List[AlgorithmResult]:
    """Run every requested algorithm and return their results IN INPUT ORDER."""

    def _run_one(name: str, algo_cancel) -> AlgorithmResult:
        """Construct + run a single backend and fold in the display reconstruction.
        Identical to the old per-name loop body; `algo_cancel` is the cancel_event this
        backend sees (the shared job event in sequential mode, a per-algo _CombinedStop
        in parallel mode)."""
        kwargs = _algo_kwargs_for(
            name,
            sft_beam_size=sft_beam_size,
            sft_beam_temperature=sft_beam_temperature,
            cds_beam_size=cds_beam_size,
            cds_temperature=cds_temperature,
            cds_nucleus_p=cds_nucleus_p,
            cds_length_penalty=cds_length_penalty,
            gbfs_max_apps=gbfs_max_apps,
            gbfs_max_seconds=gbfs_max_seconds,
            gbfs_max_terms=gbfs_max_terms,
            gbfs_den_aware=gbfs_den_aware,
            gbfs_ordering=gbfs_ordering,
            gbfs_threads=gbfs_threads,
            gbfs_target_criterion=gbfs_target_criterion,
            gbfs_audit_steps=gbfs_audit_steps,
            gbfs_split_max_terms=gbfs_split_max_terms,
            gbfs_c_cancel_max_terms=gbfs_c_cancel_max_terms,
            post_timeout_s=post_timeout_s,
        )
        try:
            algo = get_algorithm(name, **kwargs)
        except Exception as e:  # noqa: BLE001 — an optional backend whose extras are not installed
            return AlgorithmResult(
                algorithm=name,
                final_expr_str="",
                bracket_count_start=-1,
                bracket_count_final=-1,
                bracket_count_min=-1,
                n_steps=0,
                solved=False,
                timed_out=False,
                duration_s=0.0,
                extra={"error": f"backend unavailable: {type(e).__name__}: {e}"},
            )
        run_kwargs = dict(
            expr=expr,
            n_point=n_point,
            target=target,
            max_steps=max_steps,
            per_step_timeout=per_step_timeout,
            sample_timeout=sample_timeout,
            capture_derivation=capture_derivation,
            simplify_mode=simplify_mode,
            sft_patience=sft_patience,
        )
        # The GBFS and SFT backends use `cancel_event` for cooperative cancellation.
        # CDS accepts it through `**kwargs` but ignores it and is excluded here.
        if name in _CANCELLABLE:
            run_kwargs["cancel_event"] = algo_cancel
        # The per-term drivers (gbfs_e / gbfs_f) split the ORIGINAL additive terms before the
        # common denominator, so they need the raw pre-atomize string. Gate on the class flag,
        # not the name, so any future driver opts in by setting wants_raw_input=True (other
        # backends have no use for it). Missing this on gbfs_f silently degrades it to a whole-
        # expression search on opaque input (raw_split_skipped=="no_raw_input" is the tell).
        if getattr(algo, "wants_raw_input", False):
            run_kwargs["raw_input"] = raw_input
        try:
            r = algo.run(**run_kwargs)
        except Exception as e:
            r = AlgorithmResult(
                algorithm=name,
                final_expr_str="",
                bracket_count_start=-1,
                bracket_count_final=-1,
                bracket_count_min=-1,
                n_steps=0,
                solved=False,
                timed_out=False,
                duration_s=0.0,
                extra={"error": f"{type(e).__name__}: {e}"},
            )
        # Reconstructor (display-only): fold atomic ab/sb output back into s()/chain()
        # notation. Never re-runs a backend; failures are non-fatal (keep atomic form).
        try:
            from .reconstruct import reconstruct
            # gbfs_f asks for the leading -1/A*(B/C + D/E) prefactor spelling; every other backend
            # keeps the default (shape is NEVER sniffed — see reconstruct()'s docstring).
            _pf = (name == "gbfs_f")
            if r.final_expr_str:
                r.extra["reconstructed_final"] = reconstruct(r.final_expr_str, prefactor=_pf)
            if r.min_expr_str:
                r.extra["reconstructed_min"] = reconstruct(r.min_expr_str, prefactor=_pf)
        except Exception:  # noqa: BLE001  (display sugar must never break a result)
            pass
        _report_counts(r, str(expr), has_target=(target is not None), max_terms=gbfs_max_terms)
        if target is not None:
            if not is_gbfs(name):
                r.extra["backend_solved"] = r.solved
            _judge_target(r, str(target), n_point, gbfs_target_criterion, gbfs_max_terms)
        if isinstance(algo_cancel, _CombinedStop) and algo_cancel.reason():
            r.extra.setdefault("cancel_source", algo_cancel.reason())
        return r

    def _publish(i: int, res: AlgorithmResult) -> None:
        if on_result is not None:
            try:
                on_result(i, res)
            except Exception:  # noqa: BLE001 — a sink error must never break the run
                pass

    # --- sequential path (default; CLI) — behavior identical to the old loop ----------
    if max_parallel <= 1 or len(algorithm_names) <= 1:
        out = []
        for i, name in enumerate(algorithm_names):
            r = _run_one(name, cancel_event)
            _publish(i, r)
            out.append(r)
        return out

    # --- parallel path -----------------------------------------------------------------
    def _budget(name: str) -> float:
        """Per-algorithm wall-clock budget derived from the front-end params (the same
        bound each backend enforces internally): gbfs family = gbfs_max_seconds;
        everything else (sft/cds) = sample_timeout.

        gbfs is deliberately NOT also capped by sample_timeout. sample_timeout is the
        SFT/CDS per-sample rollout budget; folding it in silently overrode the user's
        explicit gbfs_max_seconds whenever it was the smaller of the two (the web UI
        exposes gbfs_max_seconds but not sample_timeout, so a raised gbfs budget had no
        effect past sample_timeout's default). The overall job stays bounded by the
        caller's own ceiling (the web service's job_timeout_s)."""
        if is_gbfs(name):
            return gbfs_max_seconds
        return sample_timeout

    n = len(algorithm_names)
    stops = [_CombinedStop(cancel_event, threading.Event()) for _ in range(n)]
    results: List[Optional[AlgorithmResult]] = [None] * n
    workers = min(max_parallel, n)

    # Publish each index exactly once — whichever fires first wins (natural completion via
    # the done-callback, out of order, OR a timeout/error placeholder from the collector).
    _published = [False] * n
    _pub_lock = threading.Lock()

    def _publish_once(i: int, res: AlgorithmResult) -> None:
        with _pub_lock:
            if _published[i]:
                return
            _published[i] = True
        _publish(i, res)

    def _make_done_cb(i: int):
        # Fires the instant future i completes (in the pool thread), so a fast backend's
        # result is surfaced immediately even while the in-order collector still waits on
        # an earlier, slower one. Cancelled/errored futures are left to the collector.
        def _cb(fut) -> None:
            if fut.cancelled():
                return
            try:
                r = fut.result()
            except Exception:  # noqa: BLE001 — collector emits the error placeholder
                return
            _publish_once(i, r)
        return _cb

    # Use shutdown(wait=False) because leaving an executor context would wait for
    # every running backend. This allows `run_all` to return after it stops waiting
    # for results. A timed-out backend may continue running, and its result is
    # discarded if it arrives after return.
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="algo")
    try:
        t_submit = time.monotonic()
        # Wall-clock instant each algorithm actually BEGAN running, filled in by the pool
        # thread. The pool only runs `workers` at a time, so with more algorithms than
        # workers the tail ones sit in the queue first; timing them from t_submit charged
        # them for that wait and could expire their deadline before they ever started
        # (6 gbfs variants at max_parallel=4 => the last 2 timed out on arrival). Each
        # deadline is therefore anchored to its own start instant.
        starts: List[Optional[float]] = [None] * n

        def _timed(idx: int, name: str):
            def _call():
                starts[idx] = time.monotonic()
                return _run_one(name, stops[idx])
            return _call

        expired_idx: List[int] = []
        futs = [ex.submit(_timed(i, name)) for i, name in enumerate(algorithm_names)]
        for i, f in enumerate(futs):
            f.add_done_callback(_make_done_cb(i))
        for i, name in enumerate(algorithm_names):
            budget = _budget(name) + algo_grace_s
            expired = False
            try:
                while True:
                    st = starts[i]
                    # Queued (no worker free yet): poll in short slices so the wait is
                    # never charged against this algorithm's own budget. Running: wait
                    # out the remainder of the budget measured from its start instant.
                    wait = (_START_POLL_S if st is None
                            else max(0.0, st + budget - time.monotonic()))
                    try:
                        results[i] = futs[i].result(timeout=wait)
                        break            # completed; already surfaced by the done-callback
                    except FutureTimeout:
                        st = starts[i]
                        if st is not None and time.monotonic() >= st + budget:
                            expired = True
                            break
                        # Still queued, or it started while we were waiting — go around
                        # and re-derive the deadline from the real start instant.
            except Exception as e:  # noqa: BLE001 — a backend crash must not sink the job
                results[i] = AlgorithmResult(
                    algorithm=name,
                    final_expr_str="",
                    bracket_count_start=-1,
                    bracket_count_final=-1,
                    bracket_count_min=-1,
                    n_steps=0,
                    solved=False,
                    timed_out=False,
                    duration_s=time.monotonic() - (starts[i] or t_submit),
                    extra={"error": f"{type(e).__name__}: {e}"},
                )
                _publish_once(i, results[i])
                continue
            if not expired:
                continue
            stops[i].set()   # nudge the still-running backend to stop (best-so-far)
            results[i] = AlgorithmResult(
                algorithm=name,
                final_expr_str="",
                bracket_count_start=-1,
                bracket_count_final=-1,
                bracket_count_min=-1,
                n_steps=0,
                solved=False,
                timed_out=True,
                duration_s=time.monotonic() - (starts[i] or t_submit),
                extra={"error": f"algorithm exceeded per-algorithm wall-clock budget "
                                f"(~{_budget(name):.0f}s + {algo_grace_s:.0f}s grace)",
                       "stop_reason": "collector_timeout",
                       "collector_budget_s": _budget(name) + algo_grace_s},
            )
            _publish_once(i, results[i])
            expired_idx.append(i)
        # A backend that overran its collector deadline but finished before this call returns
        # still has a real answer: deliver it (marked) instead of the empty placeholder.
        for i in expired_idx:
            if futs[i].done() and not futs[i].cancelled():
                try:
                    late = futs[i].result()
                except Exception:  # noqa: BLE001
                    continue
                late.extra.update(overran=True, collector_budget_s=_budget(algorithm_names[i]) + algo_grace_s,
                                  supersedes_timeout_placeholder=True)
                results[i] = late
                _publish(i, late)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return results
