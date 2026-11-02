"""Job queue and in-memory results for one web-service process.

Jobs run in a thread pool. The max_concurrent setting limits running jobs, and max_queue limits the combined number of queued and running jobs.

A supervisor waits job_timeout_s seconds, including queue time, before marking a job as timed out. This does not forcibly stop backend computation. See docs/SECURITY.md#resource-limits.

Job records are stored in memory and are lost when the process exits.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import CONFIG
from . import audit
from ..simplifier.limits import DEFAULT_GBFS_MAX_TERMS


@dataclass
class JobRecord:
    job_id: str
    email: str                       # the verified owner
    status: str = "queued"           # queued, running, done, error, timeout or cancelled
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    request: Dict[str, Any] = field(default_factory=dict)   # the clamped params
    results: Optional[List[Dict[str, Any]]] = None          # AlgorithmResult dicts
    error: Optional[str] = None
    # Progressive results, one slot per requested algorithm in request order, filled in as
    # each backend finishes so the page can show the first result immediately and a skeleton
    # with a live timer for the still-running ones. Each slot is
    # {"algorithm": name, "status": "running" or "done", "finished_at": ts or None, "result": dict or None}.
    partial: Optional[List[Dict[str, Any]]] = None
    _partial_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Cooperative cancellation event set by JobManager.cancel(). GBFS and SFT check
    # it while running. It is not included in the public job record.
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def init_partial(self, algorithm_names: List[str]) -> None:
        with self._partial_lock:
            self.partial = [{"algorithm": name, "status": "running",
                             "finished_at": None, "result": None}
                            for name in algorithm_names]

    def set_partial(self, index: int, result_dict: Dict[str, Any], finished_at: float) -> None:
        """Called from a worker thread as backend `index` completes."""
        with self._partial_lock:
            if self.partial is not None and 0 <= index < len(self.partial):
                self.partial[index] = {
                    "algorithm": result_dict.get("algorithm", self.partial[index]["algorithm"]),
                    "status": "done", "finished_at": finished_at, "result": result_dict,
                }

    def public_dict(self) -> Dict[str, Any]:
        with self._partial_lock:
            partial_snapshot = list(self.partial) if self.partial is not None else None
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            # The server clock at response time. The client anchors the running-skeleton timer
            # to server_now - started_at so the displayed elapsed is the time since the job
            # started, the same across page reloads and reconnects and independent of clock skew
            # between client and server, since the browser's own clock is not the start
            # reference. It is job-level and applies to every algorithm's skeleton.
            "server_now": time.time(),
            "finished_at": self.finished_at,
            "request": self.request,
            "results": self.results,
            "partial": partial_snapshot,
            "error": self.error,
        }


class QueueFull(Exception):
    pass


# The statuses at which a job is finished and its record may be evicted from the store.
# Queued and running jobs are live and are never evicted.
_TERMINAL_STATUSES = frozenset({"done", "error", "timeout", "cancelled"})


def _execute(req: Dict[str, Any], cancel_event=None, on_result=None) -> List[Dict[str, Any]]:
    """Build sympy exprs from the checked and validated request and run the requested
    backends. Imports are local so torch and sympy load lazily.

    The optional `on_result(index, AlgorithmResult)` is invoked as each backend finishes so
    the caller can publish progressive results before the whole job completes."""
    # A test seam that exercises the queue, timeout and cancel without importing torch. app.py
    # does not set it and it is no part of the clamped request, so HTTP cannot reach it.
    if "_test_sleep" in req:
        dur = float(req["_test_sleep"])
        waited = 0.0
        while waited < dur:                     # polled so a cancel is observable in tests
            if cancel_event is not None and cancel_event.is_set():
                break
            time.sleep(min(0.05, dur - waited))
            waited += 0.05
        return [{"algorithm": "_test", "solved": True, "bracket_count_start": 0,
                 "bracket_count_min": 0, "n_steps": 0, "duration_s": waited}]

    import sympy as sp

    from ampzip.simplifier.validate import _get_bracket_classes
    from ampzip.simplifier.runner import run_all
    from ampzip.simplifier.parse import parse_expr

    ab_cls, sb_cls = _get_bracket_classes()
    local_dict = {"ab": ab_cls, "sb": sb_cls, "I": sp.I, "opaque_den": sp.Function("opaque_den")}

    expr = parse_expr(req["expr"], local_dict, lenient=True)
    target = (parse_expr(req["target"], local_dict, lenient=True)
              if req.get("target") else None)

    results = run_all(
        algorithm_names=req["algorithms"],
        expr=expr,
        n_point=req["n_point"],
        target=target,
        max_steps=req["max_steps"],
        per_step_timeout=req["per_step_timeout"],
        sample_timeout=req["sample_timeout"],
        capture_derivation=req.get("capture_derivation", False),
        cancel_event=cancel_event,
        simplify_mode=req["simplify_mode"],
        sft_beam_size=req["sft_beam_size"],
        sft_beam_temperature=req["sft_beam_temperature"],
        sft_patience=req["sft_patience"],
        cds_beam_size=req["cds_beam_size"],
        cds_temperature=req["cds_temperature"],
        cds_nucleus_p=req["cds_nucleus_p"],
        cds_length_penalty=req["cds_length_penalty"],
        gbfs_max_apps=req["gbfs_max_apps"],
        gbfs_max_seconds=req["gbfs_max_seconds"],
        # An older saved request may carry 0 for the engine default, the same value made explicit.
        gbfs_max_terms=req.get("gbfs_max_terms") or DEFAULT_GBFS_MAX_TERMS,
        gbfs_den_aware=req.get("gbfs_den_aware", False),
        gbfs_ordering=req.get("gbfs_ordering", "factored"),
        gbfs_threads=req.get("gbfs_threads", 1),
        gbfs_target_criterion=req.get("gbfs_target_criterion", "brackets"),
        gbfs_audit_steps=req.get("gbfs_audit_steps", False),
        gbfs_split_max_terms=req.get("gbfs_split_max_terms", 64),
        gbfs_c_cancel_max_terms=req.get("gbfs_c_cancel_max_terms", 1200),
        post_timeout_s=CONFIG.post_timeout_s,
        algo_grace_s=CONFIG.algo_grace_s,
        raw_input=req.get("raw_input"),   # for gbfs_e's split by original additive terms
        on_result=on_result,              # progressive publishing as each backend finishes
        # Run the requested algorithms concurrently up to the cap, so the job wall clock is
        # about the largest per-algorithm budget. Each algorithm still honours its own timeout
        # through run_all's per-algorithm deadline.
        max_parallel=CONFIG.max_parallel_algos,
    )
    return [r.to_dict() for r in results]


class JobManager:
    """Owns the worker pool and the job store. One instance per process."""

    def __init__(self) -> None:
        # max_workers is the concurrency cap. Excess submissions queue here.
        self._pool = ThreadPoolExecutor(max_workers=CONFIG.max_concurrent,
                                        thread_name_prefix="simp-job")
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._inflight = 0  # queued plus running

    # --- admission control ---------------------------------------------------
    def submit(self, email: str, clamped: Dict[str, Any]) -> JobRecord:
        with self._lock:
            if self._inflight >= CONFIG.max_queue:
                raise QueueFull(f"queue full ({self._inflight}/{CONFIG.max_queue})")
            self._inflight += 1
            # Guarantee a unique id. A 64-bit truncation could, astronomically rarely, collide
            # with a live job, and `self._jobs[id] = job` would then overwrite it and hand the
            # old job's poller the new job's results. Regenerating on collision rules that out.
            job_id = uuid.uuid4().hex[:16]
            while job_id in self._jobs:
                job_id = uuid.uuid4().hex[:16]
            job = JobRecord(
                job_id=job_id,
                email=email,
                created_at=time.time(),
                request=clamped,
            )
            self._jobs[job.job_id] = job
            self._evict_terminal_over_cap_locked()

        # Pre-seed one progressive slot per requested algorithm in request order, so the
        # client can render skeletons and live timers before any backend finishes.
        job.init_partial(list(clamped.get("algorithms", [])))

        # The sink publishes each backend's result into the job the instant it finishes.
        def _on_result(index: int, algo_result) -> None:
            try:
                job.set_partial(index, algo_result.to_dict(), time.time())
            except Exception:  # noqa: BLE001  a publishing error is ignored
                pass

        fut = self._pool.submit(_execute, job.request, job.cancel_event, _on_result)
        # One lightweight supervisor thread per job, mostly blocked on the result.
        t = threading.Thread(target=self._supervise, args=(job, fut),
                             name=f"supervise-{job.job_id}", daemon=True)
        t.start()
        return job

    def _evict_terminal_over_cap_locked(self) -> None:
        """Bound the in-memory job store. The caller holds self._lock. If the store exceeds
        CONFIG.max_jobs, the oldest finished jobs are dropped first, and queued or running
        jobs, still live and polled, stay. A client polling an evicted job gets a 404 and
        re-runs. Unfinished jobs are separately bounded by max_queue, at most 32, so enough
        finished jobs exist to get back under the cap. Evicting the oldest finished first
        keeps a just-completed result."""
        overflow = len(self._jobs) - CONFIG.max_jobs
        if overflow <= 0:
            return
        terminal = [j for j in self._jobs.values() if j.status in _TERMINAL_STATUSES]
        # The oldest first, by finish time and then by creation time.
        terminal.sort(key=lambda j: (j.finished_at or j.created_at or 0.0))
        for j in terminal[:overflow]:
            self._jobs.pop(j.job_id, None)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str, email: str) -> Optional[JobRecord]:
        """Return the job if `email` owns it, otherwise return None for a 404 response.

        Set its cancellation event only while it is queued or running. GBFS and
        SFT check the event and return their best form so far. CDS decoding
        continues. On a normal return, `_supervise` reports cancelled if the event
        is set. Timeout and error paths retain their respective statuses.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.email != email:
                return None
            if job.status in ("queued", "running"):
                job.cancel_event.set()
            return job

    # --- per-job supervisor --------------------------------------------------
    def _supervise(self, job: JobRecord, fut) -> None:
        job.status = "running"   # approximate, since it may still be queued in the pool
        job.started_at = time.time()
        try:
            results = fut.result(timeout=CONFIG.job_timeout_s)
            job.results = results
            # A stopped search may still return results. Preserve the cancellation
            # status on this normal return.
            job.status = "cancelled" if job.cancel_event.is_set() else "done"
        except FutureTimeout:
            # The job has exceeded its waiting time limit. Its computation may
            # continue in the background, and any later result is discarded.
            job.status = "timeout"
            job.error = (f"job exceeded WEBSVC_JOB_TIMEOUT_S ({CONFIG.job_timeout_s:.0f} s), "
                         "and background computation may continue")
        except Exception as e:  # noqa: BLE001  surface any backend failure
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = time.time()
            with self._lock:
                self._inflight -= 1
            audit.audit_finish(
                email=job.email, job_id=job.job_id, status=job.status,
                duration_s=(job.finished_at - (job.started_at or job.finished_at)),
                results=job.results, error=job.error,
            )


# Process-wide singleton.
MANAGER = JobManager()
