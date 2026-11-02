"""FastAPI app — the public entry point.

Each simplification job passes through these stages in the following order:

    authentication (configured method)
      → rate limit (per email)
      → safety.prescreen (char firewall, RCE guard)
      → atomize input notation
      → simplifier.validate (atom whitelist + size limits)
      → clamp params to the server limits
      → select backends compatible with the input
      → enqueue job (bounded queue)

These stages precede backend execution. For deployment behind cloudflared, bind
to `127.0.0.1` so the tunnel connects locally. The supplied Docker image binds to
`0.0.0.0` inside the container, while Compose publishes the service on the
host's loopback address. Never expose the service port publicly when
authentication is disabled.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import audit
from .. import __version__
from .config import CONFIG
from .auth import AuthError, verify_request_email
from .ratelimit import LIMITER
from .safety import UnsafeInputError, prescreen
from .jobs import MANAGER, QueueFull

import threading
from dataclasses import asdict
from contextlib import contextmanager

import sympy as sp

# Bounds concurrent in-thread atomize()+validate() (CPU-bound sympy/AST that cannot be
# hard-interrupted in a thread). Without this, a burst of large inputs could occupy every
# Starlette worker thread and freeze the whole service — /healthz and job polling included.
# Shared by /api/jobs and /api/numerical_check (both parse untrusted input). See config.py.
_ATOMIZE_SEM = threading.Semaphore(CONFIG.atomize_concurrency)


@contextmanager
def _atomize_slot():
    """Hold one atomize/validate permit, or 503 if the pool stays saturated past the
    acquire timeout. Releases on any exit (including the HTTPExceptions raised inside)."""
    if not _ATOMIZE_SEM.acquire(timeout=CONFIG.atomize_acquire_timeout_s):
        raise HTTPException(
            status_code=503,
            detail="server busy (too many expressions parsing concurrently); retry shortly",
        )
    try:
        yield
    finally:
        _ATOMIZE_SEM.release()


app = FastAPI(
    title="AmpZip",
    version=__version__,
    docs_url=None,        # disable interactive docs on a public endpoint
    redoc_url=None,
    openapi_url=None,
)


# --- request / response models ----------------------------------------------
class SubmitRequest(BaseModel):
    expr: str = Field(..., description="expression in ab/sb, s() or chain() notation, e.g. ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))")
    algorithms: List[str] = Field(default_factory=lambda: ["sft_greedy"])
    n_point: Optional[Union[int, str]] = None   # 4/5/6, or "auto"/None = infer from the leg labels
    target: Optional[str] = None
    simplify_mode: str = "none"
    capture_derivation: bool = False
    # tunables (clamped server-side)
    max_steps: int = 50
    per_step_timeout: float = 30.0
    sample_timeout: float = 120.0
    sft_beam_size: int = 5
    sft_beam_temperature: float = 1.5
    sft_patience: int = 0
    cds_beam_size: int = 5
    cds_temperature: float = 1.5
    cds_nucleus_p: float = 0.0
    cds_length_penalty: float = 1.0
    gbfs_max_apps: int = 60000        # paper headline budget (limits.DEFAULT_GBFS_MAX_APPS)
    gbfs_max_seconds: float = 300.0   # paper headline cap; clamped to the deployment ceiling
    gbfs_max_terms: int = 50000   # per-move numerator term cap (limits.DEFAULT_GBFS_MAX_TERMS)
    gbfs_den_aware: bool = False
    gbfs_ordering: str = "factored"           # factored (paper gbfs) | raw (paper gbfsraw)
    gbfs_target_criterion: str = "brackets"   # none | brackets | brackets-and-terms
    gbfs_threads: int = 1
    gbfs_audit_steps: bool = False
    gbfs_split_max_terms: int = 64
    gbfs_c_cancel_max_terms: int = 1200

    @field_validator("n_point", mode="before")
    @classmethod
    def _n_point_auto(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto")):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError("n_point must be 4, 5, 6 or 'auto'")


class NumCheckRequest(BaseModel):
    """Synchronous numerical-equivalence check (no model, no queue).

    Two expressions are compared at the CDS pre-computed momentum points
    (MOMENTA_DICT1 + MOMENTA_DICT2 in spinorhelicity). Sub-second per call.
    Supports n_point in {4, 5, 6} only.
    """
    A: str = Field(..., description="left expression in ab/sb, s() or chain() notation")
    B: str = Field(..., description="right expression in ab/sb, s() or chain() notation")
    n_point: Optional[int] = None


# --- auth dependency ---------------------------------------------------------
def require_email(request: Request) -> str:
    client_ip = request.client.host if request.client else "?"
    try:
        return verify_request_email(request.headers, client_ip=client_ip)
    except AuthError as e:
        audit.audit_reject(email=None, stage="auth", detail=e.detail)
        # Propagate WWW-Authenticate (Basic mode) so browsers show a login dialog.
        raise HTTPException(status_code=e.status, detail=e.detail,
                            headers=e.headers or None)


# --- helpers -----------------------------------------------------------------
def _clamp(value, lo, hi):
    return max(lo, min(value, hi))


def _build_clamped_request(body: SubmitRequest, n_point: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply all server ceilings. Returns `(request, clamped)` where `clamped` lists every
    field the server changed as `{field: [requested, applied]}` (ceilings, fallbacks,
    dropped algorithms) so nothing is adjusted silently."""
    from ampzip.simplifier.limits import DEFAULT_GBFS_MAX_TERMS
    clamps: Dict[str, Any] = {}

    def _c(field, value, lo, hi, cast):
        applied = cast(_clamp(value, lo, hi))
        if applied != value:
            clamps[field] = [value, applied]
        return applied

    algos = [a for a in body.algorithms if a in CONFIG.allowed_algorithms]
    unknown = [a for a in body.algorithms if a not in CONFIG.allowed_algorithms]
    if unknown:
        clamps["algorithms_dropped_unknown"] = unknown
    if not algos:
        raise HTTPException(
            status_code=400,
            detail=f"no valid algorithms; allowed: {list(CONFIG.allowed_algorithms)}",
        )
    if body.simplify_mode not in ("none", "cancel", "expand", "simplify"):
        clamps["simplify_mode"] = [body.simplify_mode, "none"]
    for field, allowed, default in (("gbfs_ordering", ("factored", "raw"), "factored"),
                                    ("gbfs_target_criterion", ("none", "brackets", "brackets-and-terms"), "brackets")):
        if getattr(body, field) not in allowed:
            clamps[field] = [getattr(body, field), default]
    den_aware = bool(body.gbfs_den_aware)
    ordering = body.gbfs_ordering if body.gbfs_ordering in ("factored", "raw") else "factored"
    if den_aware and ordering == "raw":
        clamps["gbfs_den_aware"] = [True, False]      # the two orderings are exclusive; raw wins
        den_aware = False
    gbfs_max_terms = body.gbfs_max_terms
    if gbfs_max_terms == 0:
        gbfs_max_terms = DEFAULT_GBFS_MAX_TERMS
        clamps["gbfs_max_terms"] = [0, gbfs_max_terms]
    req = {
        "expr": body.expr,
        "target": body.target,
        "algorithms": algos,
        "n_point": n_point,
        "simplify_mode": (body.simplify_mode
                          if body.simplify_mode in ("none", "cancel", "expand", "simplify")
                          else "none"),
        # Always capture the per-step derivation — the UI renders it as a panel,
        # and the cost (storing step expression strings) is negligible.
        "capture_derivation": True,
        "max_steps": _c("max_steps", body.max_steps, 1, CONFIG.max_max_steps, int),
        "per_step_timeout": _c("per_step_timeout", body.per_step_timeout, 1.0, CONFIG.max_per_step_timeout, float),
        "sample_timeout": _c("sample_timeout", body.sample_timeout, 1.0, CONFIG.max_sample_timeout, float),
        "sft_beam_size": _c("sft_beam_size", body.sft_beam_size, 1, CONFIG.max_sft_beam_size, int),
        "sft_beam_temperature": _c("sft_beam_temperature", body.sft_beam_temperature, 0.1, 5.0, float),
        "sft_patience": _c("sft_patience", body.sft_patience, 0, CONFIG.max_sft_patience, int),
        "cds_beam_size": _c("cds_beam_size", body.cds_beam_size, 1, CONFIG.max_sft_beam_size, int),
        "cds_temperature": _c("cds_temperature", body.cds_temperature, 0.1, 5.0, float),
        "cds_nucleus_p": _c("cds_nucleus_p", body.cds_nucleus_p, 0.0, 1.0, float),
        "cds_length_penalty": _c("cds_length_penalty", body.cds_length_penalty, 0.1, 3.0, float),
        "gbfs_max_apps": _c("gbfs_max_apps", body.gbfs_max_apps, 100, CONFIG.max_gbfs_max_apps, int),
        # GBFS has separate time and move budgets. `sample_timeout` applies to SFT/CDS.
        "gbfs_max_seconds": _c("gbfs_max_seconds", body.gbfs_max_seconds, 1.0, CONFIG.max_gbfs_max_seconds, float),
        # 0 was the old spelling of "engine default"; it becomes the explicit default (reported
        # below), so no caller can end up with a one-term cap. Positive values clamp to the ceiling.
        "gbfs_max_terms": _c("gbfs_max_terms", gbfs_max_terms, 1, CONFIG.max_gbfs_max_terms, int),
        "gbfs_den_aware": den_aware,
        "gbfs_ordering": ordering,
        "gbfs_target_criterion": (body.gbfs_target_criterion if body.gbfs_target_criterion in ("none", "brackets", "brackets-and-terms") else "brackets"),
        "gbfs_threads": _c("gbfs_threads", body.gbfs_threads, 1, CONFIG.max_gbfs_threads, int),
        "gbfs_audit_steps": bool(body.gbfs_audit_steps),
        "gbfs_split_max_terms": _c("gbfs_split_max_terms", body.gbfs_split_max_terms, 0, CONFIG.max_gbfs_split_max_terms, int),
        "gbfs_c_cancel_max_terms": _c("gbfs_c_cancel_max_terms", body.gbfs_c_cancel_max_terms, 0, CONFIG.max_gbfs_c_cancel_max_terms, int),
    }
    return req, clamps


# --- routes ------------------------------------------------------------------
_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Unauthenticated liveness probe (cloudflared / monitoring)."""
    return {"ok": True, "version": app.version}


@app.get("/api/limits")
def limits(email: str = Depends(require_email)) -> Dict[str, Any]:
    """Deployment limits used by the page for input bounds and tooltips."""
    return {
        "gbfs_max_apps": CONFIG.max_gbfs_max_apps,
        "gbfs_max_seconds": CONFIG.max_gbfs_max_seconds,
        "gbfs_max_terms": CONFIG.max_gbfs_max_terms,
        "sft_patience": CONFIG.max_sft_patience,
        "gbfs_threads": CONFIG.max_gbfs_threads,
        "gbfs_split_max_terms": CONFIG.max_gbfs_split_max_terms,
        "gbfs_c_cancel_max_terms": CONFIG.max_gbfs_c_cancel_max_terms,
        "algo_grace_s": CONFIG.algo_grace_s,
        # Input limits shared with the CLI, checked before parsing.
        "input_limits": asdict(CONFIG.input_limits),
        "post_timeout_s": CONFIG.post_timeout_s,
        "allowed_algorithms": list(CONFIG.allowed_algorithms),
    }


_VENDORED = {"mathjax/tex-svg.js": "application/javascript"}


@app.get("/static/{name:path}")
def static_asset(name: str) -> FileResponse:
    """The vendored front-end assets, by explicit allowlist (never the page itself, which is
    served by `/` behind auth). Unauthenticated: a script file carries nothing to protect and
    the browser requests it before any credential exchange."""
    if name not in _VENDORED:
        raise HTTPException(status_code=404)
    return FileResponse(_STATIC / name, media_type=_VENDORED[name])


@app.get("/")
def index(email: str = Depends(require_email)) -> FileResponse:
    """Serve the single-page UI — gated by auth so the browser prompts on page
    load (Basic mode) and the cached credentials ride along on the /api/* fetches.
    In cloudflare mode the Access JWT is present on this request too; in dev mode
    auth is bypassed."""
    return FileResponse(_STATIC / "index.html")


@app.post("/api/jobs", status_code=202)
def submit_job(body: SubmitRequest, email: str = Depends(require_email)) -> Dict[str, Any]:
    # 1) rate limit
    if not LIMITER.allow(email):
        audit.audit_reject(email=email, stage="ratelimit",
                           detail=f"{CONFIG.rate_per_min}/min")
        raise HTTPException(status_code=429,
                            detail=f"rate limit exceeded ({CONFIG.rate_per_min}/min)")

    # 2) Input rules and limits shared with the CLI, from ampzip.simplifier.input_limits.
    try:
        prescreen(body.expr, CONFIG.input_limits)
        if body.target:
            prescreen(body.target, CONFIG.input_limits)
    except UnsafeInputError as e:
        audit.audit_reject(email=email, stage="safety", detail=e.reasons,
                           raw_input=body.expr)
        raise HTTPException(status_code=400, detail={"unsafe_input": e.reasons})

    # 2b) Convert s()/chain() to ab/sb, retaining supported denominator sums as
    #     opaque_den markers. Keep the original string too: gbfs_e and gbfs_f split
    #     its additive terms before atomization combines them over a common denominator.
    raw_original = body.expr
    from ampzip.simplifier.atomize import atomize, AtomizeError, has_opaque_den
    from ampzip.simplifier.validate import validate, SIZE_LIMIT_TAGS, ENGINE_LIMIT_TAGS
    # atomize + validate are the CPU-bound sympy stage; bound their concurrency so a burst
    # of large inputs cannot occupy every request thread (503 on sustained saturation).
    with _atomize_slot():
        # Atomize both the input and the optional target before validation.
        try:
            atomic_expr = atomize(body.expr, n_hint=body.n_point)
            atomic_target = atomize(body.target, n_hint=body.n_point) if body.target else None
        except AtomizeError as e:
            audit.audit_reject(email=email, stage="atomize", detail=str(e),
                               raw_input=body.expr)
            raise HTTPException(status_code=422, detail={"atomize_failed": str(e)})
        body = body.model_copy(update={"expr": atomic_expr, "target": atomic_target})

        # 3) semantic validation (now safe to parse). enforce_size_limits=False: the SFT/CDS
        #    size budget must not reject the whole request, because a too-big expression is
        #    still fair game for the model-free gbfs family. Size issues are reported in
        #    v.issues (SIZE_LIMIT_TAGS) and handled by the drop filter below; only UNIVERSAL
        #    failures (atoms/symbols/coeffs/leg labels/n_point) 422 here.
        v = validate(body.expr, override_n_point=body.n_point, enforce_size_limits=False)
        if not v.ok:
            issues = [{"tag": i.tag, "message": i.message} for i in v.issues]
            audit.audit_reject(email=email, stage="validate", detail=issues,
                               raw_input=body.expr)
            raise HTTPException(status_code=422, detail={"validation_failed": issues})

    # 4) clamp request parameters
    try:
        clamped, clamps = _build_clamped_request(body, n_point=v.inferred_n_point)
    except HTTPException as e:
        audit.audit_reject(email=email, stage="algorithms", detail=e.detail,
                           raw_input=body.expr)
        raise
    clamped["raw_input"] = raw_original   # for gbfs_e/gbfs_f term splitting

    # 5) Route by denominator support, checkpoint size limits and engine restrictions.
    #    Each filter updates the current algorithm list. Use the shared GBFS_FAMILY
    #    registry so the filters include every registered search variant.
    from ampzip.simplifier.runner import GBFS_FAMILY
    notes: List[str] = []
    for field, change in clamps.items():
        if isinstance(change, list) and len(change) == 2:
            notes.append(f"{field}: {change[0]} -> {change[1]} (server limit)")
        else:
            notes.append(f"{field}: {change}")

    # 5a) Opaque polynomial denominator — a gbfs-FAMILY capability (SFT/CDS would mangle the
    #     opaque_den(S) marker). Keep ALL selected gbfs variants, drop only the non-gbfs ones;
    #     if none was selected, fall back to model_free_gbfs.
    if has_opaque_den(atomic_expr):
        allowed_gbfs = [a for a in GBFS_FAMILY if a in CONFIG.allowed_algorithms]
        if not allowed_gbfs:
            raise HTTPException(status_code=422, detail={
                "opaque_unsupported": "polynomial/spurious-pole denominator needs a gbfs "
                                      "backend, none of which is enabled on this server"})
        kept = [a for a in clamped["algorithms"] if a in allowed_gbfs]
        if not kept:
            kept = (["model_free_gbfs"] if "model_free_gbfs" in allowed_gbfs
                    else [allowed_gbfs[0]])
        dropped = [a for a in clamped["algorithms"] if a not in kept]
        clamped["algorithms"] = kept
        if dropped:
            clamps["algorithms_dropped_opaque"] = dropped
        notes.append("opaque (polynomial) denominator detected — running gbfs only"
                     + (f"; dropped {dropped}" if dropped else ""))

    # 5b) Expression exceeds the SFT/CDS size budget. Those checkpoints have fixed
    #     observation widths, so they cannot run — but the model-free gbfs family has no such
    #     limit. Drop SFT/CDS and keep gbfs; if nothing gbfs was selected, there is nothing
    #     left to run, so 422 with the size issues plus a pointer to gbfs.
    size_issues = [i for i in v.issues if i.tag in SIZE_LIMIT_TAGS]
    if size_issues:
        gbfs_kept = [a for a in clamped["algorithms"] if a in GBFS_FAMILY]
        dropped = [a for a in clamped["algorithms"] if a not in GBFS_FAMILY]
        if not gbfs_kept:
            detail = {"validation_failed": [{"tag": i.tag, "message": i.message}
                                            for i in size_issues],
                      "hint": "expression exceeds the SFT/CDS checkpoint size limits. "
                              "Select a GBFS backend (e.g. model_free_gbfs)."}
            audit.audit_reject(email=email, stage="validate", detail=detail["validation_failed"],
                               raw_input=body.expr)
            raise HTTPException(status_code=422, detail=detail)
        clamped["algorithms"] = gbfs_kept
        if dropped:
            clamps["algorithms_dropped_size"] = dropped
            notes.append("expression exceeds SFT/CDS size limits — running gbfs only; "
                         f"dropped {dropped}")

    # Engine route: a rational coefficient or the imaginary unit is accepted by the learned
    # backends but not representable by the search engine (LIMIT_ENGINE_COEFF). Drop the
    # search family with a note; 422 only when nothing is left to run.
    engine_issues = [i for i in v.issues if i.tag in ENGINE_LIMIT_TAGS]
    if engine_issues:
        kept = [a for a in clamped["algorithms"] if a not in GBFS_FAMILY]
        dropped = [a for a in clamped["algorithms"] if a in GBFS_FAMILY]
        if not kept:
            detail = {"validation_failed": [{"tag": i.tag, "message": i.message} for i in engine_issues],
                      "hint": "the search engine cannot represent this input; select a learned "
                              "backend (sft_greedy, sft_best_of_n, cds_oneshot) or rewrite it"}
            audit.audit_reject(email=email, stage="validate", detail=detail["validation_failed"],
                               raw_input=body.expr)
            raise HTTPException(status_code=422, detail=detail)
        clamped["algorithms"] = kept
        clamps["algorithms_dropped_engine"] = dropped
        notes.append("search engine cannot represent the input (" +
                     "; ".join(i.message for i in engine_issues) + f") — dropped {dropped}")

    clamped["clamped"] = clamps
    try:
        job = MANAGER.submit(email, clamped)
    except QueueFull as e:
        audit.audit_reject(email=email, stage="queue", detail=str(e))
        raise HTTPException(status_code=429, detail=str(e))

    audit.audit_submit(email=email, job_id=job.job_id,
                       n_point=v.inferred_n_point,
                       canonical_expr=v.canonical_expr_str, request=clamped)
    resp = {"job_id": job.job_id, "status": job.status,
            "canonical_expr": v.canonical_expr_str, "n_point": v.inferred_n_point,
            "clamped": clamps}
    if notes:
        resp["note"] = "; ".join(notes)
    return resp


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, email: str = Depends(require_email)) -> Dict[str, Any]:
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.email != email:
        # Don't leak existence to non-owners.
        raise HTTPException(status_code=404, detail="job not found")
    return job.public_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, email: str = Depends(require_email)) -> Dict[str, Any]:
    # Request cooperative cancellation for GBFS and SFT. They stop at their next
    # cancellation check and return the best form found so far. Unknown jobs and
    # jobs owned by another user both return 404, as in get_job.
    job = MANAGER.cancel(job_id, email)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.public_dict()


# --- numerical equivalence check --------------------------------------------
def _flat_subs(n_point: int, ab_cls, sb_cls) -> Dict[Any, Any]:
    """{ab(i,j): ±Symbol('abij'), sb(i,j): ±Symbol('sbij')} for 1<=i,j<=n, i!=j.

    spinorhelicity's check_numerical_equiv_local uses flat-symbol names
    (ab12, sb34, ...) keyed only by the i<j canonical ordering. Spinor
    brackets are antisymmetric: <ji> = -<ij>, so non-canonical inputs
    like ab(4,1) get mapped to -Symbol('ab14') here.
    """
    from ampzip.verify import DEFAULT_K, DEFAULT_PRIMES, compare, equal

    try:
        verdict, info = compare(A_text, B_text, n_point, rng=rng)
    except ValueError:
        # An older engine cannot parse I. Evaluate the sympy expressions instead.
        return _sympy_compare(A_text, B_text, n_point, rng)
    method = (f"exact finite-field comparison at {info['k']} random on-shell points modulo "
              f"each of the primes {', '.join(str(p) for p in info['primes'])}")
    if info["has_I"]:
        method += "; I is mapped to a square root of -1 mod p"
    return verdict, method


_GAUSSIAN_PRIMES = (2_147_483_629, 2_147_483_549)      # p = 1 (mod 4)


def _sympy_compare(A_text: str, B_text: str, n_point: int, rng=None) -> Tuple[Optional[bool], str]:
    from sympy.ntheory import sqrt_mod
    from ampzip.simplifier.parse import parse_expr
    from ampzip.simplifier.validate import _get_bracket_classes
    from ampzip.verify import DEFAULT_K, DEFAULT_PRIMES, equal
    from ampzip.verify.verifier import _strip_opaque_den

    ab_cls, sb_cls = _get_bracket_classes()
    local = {"ab": ab_cls, "sb": sb_cls, "I": sp.I, "opaque_den": sp.Function("opaque_den")}
    A = _strip_opaque_den(parse_expr(A_text, local, lenient=True))
    B = _strip_opaque_den(parse_expr(B_text, local, lenient=True))
    method = f"exact finite-field comparison at {DEFAULT_K} random on-shell points modulo each of 2 primes"
    try:
        if not (A.has(sp.I) or B.has(sp.I)):
            return equal(A, B, n_point, primes=DEFAULT_PRIMES, rng=rng, engine=False), method
        method += "; I is mapped to a square root of -1 mod p"
        for p in _GAUSSIAN_PRIMES:
            r = sp.Integer(sqrt_mod(-1, p))
            if not equal(A.subs(sp.I, r), B.subs(sp.I, r), n_point, primes=(p,), rng=rng, engine=False):
                return False, method
        return True, method
    except RuntimeError:
        return None, method


@app.post("/api/numerical_check")
def numerical_check(body: NumCheckRequest,
                    email: str = Depends(require_email)) -> Dict[str, Any]:
    """Compare A and B exactly mod p at random on-shell points.

    check_status is matched, mismatch or inconclusive, and equiv True, False or None. A
    mismatch is certain; a match is probabilistic evidence. Inconclusive means too many
    sampled points were singular. Input and program errors remain HTTP errors.
    """
    # 1) rate limit (cheap, shared with /api/jobs)
    if not LIMITER.allow(email):
        audit.audit_reject(email=email, stage="ratelimit",
                           detail=f"{CONFIG.rate_per_min}/min")
        raise HTTPException(status_code=429,
                            detail=f"rate limit exceeded ({CONFIG.rate_per_min}/min)")

    # 2) Input rules and limits, as in /api/jobs.
    try:
        prescreen(body.A, CONFIG.input_limits)
        prescreen(body.B, CONFIG.input_limits)
    except UnsafeInputError as e:
        audit.audit_reject(email=email, stage="safety", detail=e.reasons,
                           raw_input=f"A={body.A!r} B={body.B!r}"[:200])
        raise HTTPException(status_code=400, detail={"unsafe_input": e.reasons})

    # The whole atomize -> validate -> parse -> numeric-eval body is CPU-bound sympy on
    # untrusted input; bound its concurrency alongside /api/jobs (503 on saturation).
    from ampzip.simplifier.atomize import atomize, AtomizeError
    from ampzip.simplifier.validate import validate, _get_bracket_classes
    with _atomize_slot():
        # 2b) Expand s()/chain() to ab/sb, retaining denominator sums as opaque_den(S).
        #     Each marker represents 1/S with S an ab/sb polynomial. Step 4 converts
        #     the markers to reciprocals for numerical evaluation.
        try:
            A_atomic = atomize(body.A, n_hint=body.n_point, allow_opaque_den=True)
            B_atomic = atomize(body.B, n_hint=body.n_point, allow_opaque_den=True)
        except AtomizeError as e:
            audit.audit_reject(email=email, stage="atomize", detail=str(e),
                               raw_input=f"A={body.A!r} B={body.B!r}"[:200])
            raise HTTPException(status_code=422, detail={"atomize_failed": str(e)})

        # 3) validate both (on the atomized ab/sb form) — must agree on n_point.
        #    enforce_size_limits=False: the numerical check runs no model, so the SFT/CDS size
        #    budget is irrelevant — a big expression is still perfectly evaluable.
        vA = validate(A_atomic, override_n_point=body.n_point, enforce_size_limits=False)
        if not vA.ok:
            issues = [{"tag": i.tag, "message": i.message} for i in vA.issues]
            raise HTTPException(status_code=422,
                                detail={"validation_failed_A": issues})
        vB = validate(B_atomic, override_n_point=body.n_point, enforce_size_limits=False)
        if not vB.ok:
            issues = [{"tag": i.tag, "message": i.message} for i in vB.issues]
            raise HTTPException(status_code=422,
                                detail={"validation_failed_B": issues})
        if vA.inferred_n_point != vB.inferred_n_point:
            raise HTTPException(
                status_code=400,
                detail=f"A is {vA.inferred_n_point}-pt but B is {vB.inferred_n_point}-pt; "
                       f"can't compare different n_point",
            )
        n_point = vA.inferred_n_point
        if n_point not in (4, 5, 6):
            raise HTTPException(status_code=400,
                detail=f"numerical check supports only 4/5/6-pt (got n_point={n_point})")

        # 4) parse both (atomized ab/sb form), convert to flat-symbol form
        ab_cls, sb_cls = _get_bracket_classes()
        local_dict = {"ab": ab_cls, "sb": sb_cls, "I": sp.I,
                      "opaque_den": sp.Function("opaque_den")}
        try:
            from ampzip.simplifier.parse import parse_expr
            A_expr = parse_expr(A_atomic, local_dict, lenient=True)
            B_expr = parse_expr(B_atomic, local_dict, lenient=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"parse error: {e}")
        # opaque_den(S) -> 1/S so the sum lands in a real denominator before evaluation.
        from ampzip.verify.verifier import _strip_opaque_den
        A_expr = _strip_opaque_den(A_expr)
        B_expr = _strip_opaque_den(B_expr)
        flat = _flat_subs(n_point, ab_cls, sb_cls)
        A_flat = A_expr.subs(flat)
        B_flat = B_expr.subs(flat)

        # 5) numerical equivalence check (pre-computed momenta; sub-second)
        from add_ons.numerical_evaluations import check_numerical_equiv_local
        tokens = _all_bracket_names(n_point)
        try:
            valid, rel_diff = check_numerical_equiv_local(tokens, A_flat, B_flat, npt=n_point)
            rel_diff = float(rel_diff)
            if math.isfinite(rel_diff):
                equiv = bool(valid)
                check_status = "matched" if equiv else "mismatch"
            else:
                equiv, rel_diff, check_status = None, None, "inconclusive"
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            # A singular point or non-finite evaluation gives no comparison.
            # Type/parse/import errors and unexpected failures are still errors.
            equiv, rel_diff, check_status = None, None, "inconclusive"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"numerical evaluation failed: {e}")

    return {
        "check_status": check_status,
        "equiv": equiv,
        "rel_diff": rel_diff,
        "n_point": n_point,
        "canonical_A": vA.canonical_expr_str,
        "canonical_B": vB.canonical_expr_str,
        "method": "check_numerical_equiv_local (2 precomputed on-shell configurations, tolerance 2e-9)",
    }
