"""Runtime configuration. All knobs are env-driven so nothing sensitive
(email whitelist, Access AUD) is hard-coded into the image.

Read once at import via `CONFIG = load_config()`. The server caps here are the
*hard ceilings* — user-supplied params (beam size, max_steps, timeouts) are
clamped to these in app.py before any job is enqueued.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_set(key: str) -> FrozenSet[str]:
    raw = os.environ.get(key, "")
    return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())


def _env_basic_creds(key: str) -> Dict[str, str]:
    """Parse WEBSVC_BASIC_CREDS = 'email:pbkdf2hash,email2:pbkdf2hash2'.
    Splits each entry on the FIRST ':' (emails have no ':'; the hash has no ',')."""
    out: Dict[str, str] = {}
    raw = os.environ.get(key, "")
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        email, stored = entry.split(":", 1)
        out[email.strip().lower()] = stored.strip()
    return out


@dataclass(frozen=True)
class Config:
    # --- bind ---
    # Bind to `127.0.0.1` on the host so cloudflared connects locally.
    # The supplied Docker image uses `0.0.0.0` inside the container, while Compose
    # publishes the port on the host's loopback address. Never expose an
    # unauthenticated service port to the public network.
    host: str = "127.0.0.1"
    port: int = 8000

    # --- auth backend selection ---
    #   "cloudflare" : verify Cloudflare Access signed JWT (production target)
    #   "basic"      : INTERIM HTTP Basic auth (while Access is unavailable)
    #   "dev"        : no auth (localhost only); also enabled by dev_allow_no_auth
    auth_mode: str = "cloudflare"

    # --- Cloudflare Access (auth_mode="cloudflare") ---
    # e.g. "myteam.cloudflareaccess.com" — the JWKS issuer.
    cf_team_domain: str = ""
    # The Access *application* AUD tag (from the CF dashboard). JWT audience must match.
    cf_access_aud: str = ""

    # --- Interim HTTP Basic (auth_mode="basic"): email -> pbkdf2 hash ---
    basic_creds: Dict[str, str] = field(default_factory=dict)
    # Lock out an IP after this many failed Basic logins within the window.
    basic_max_fails: int = 8
    basic_fail_window_s: float = 300.0

    # --- email whitelist (defense-in-depth; CF Access is the real gate) ---
    # Even with CF in front, re-check the verified email here in case the
    # tunnel is ever bypassed or misconfigured.
    email_whitelist: FrozenSet[str] = field(default_factory=frozenset)

    # --- DEV ONLY: skip JWT verification (localhost testing). NEVER true in prod. ---
    dev_allow_no_auth: bool = False

    # --- queue / concurrency / DoS caps ---
    max_concurrent: int = 4        # simultaneous running jobs (CPU/sympy bound)
    max_queue: int = 32            # queued+running before 429
    # Job waiting time limit, including time in the job queue. Input atomization and
    # validation finish before submission. On timeout, the job is marked as timed
    # out, but its computation may continue. For five GBFS backends and four backend
    # workers, two batches give 2 * (120 + 30) = 300 s at the default limits, or
    # 1260 s at a 600 s GBFS limit. Queueing and work outside the search loop
    # require additional time.
    job_timeout_s: float = 1400.0
    # Maximum concurrent backends within one job. Additional backends wait in that
    # job's executor queue. Keep this limit modest because the backends perform
    # CPU-intensive SymPy and search work. A job timeout removes the job from
    # admission counting, but its job-pool worker remains occupied until `run_all`
    # returns. Timed-out backend threads may continue after that return, so these
    # settings do not cap the total thread count.
    max_parallel_algos: int = 4
    # Upper bound on the in-memory job store. Jobs live forever otherwise, so a long-running
    # server would leak memory as finished jobs pile up. Past this, the OLDEST TERMINAL
    # (done/error/timeout/cancelled) jobs are evicted first — never queued/running ones. A
    # client polling an evicted job gets 404 and re-runs. Generous so only long-stale results
    # are dropped; non-terminal jobs are separately capped by max_queue.
    max_jobs: int = 500

    # --- input caps (cheap pre-checks before validate()) ---
    max_input_chars: int = 8000
    max_digit_run: int = 3         # reject integer literals longer than this

    # --- atomize/validate concurrency guard ---
    # atomize()+validate() run synchronously in the request thread and are CPU-bound
    # (sympy/AST). They are NOT hard-killable in-thread, so a burst of large inputs could
    # otherwise occupy every Starlette worker thread and make the WHOLE service (incl.
    # /healthz and job polling) unresponsive. This semaphore bounds how many run at once;
    # excess submits wait up to acquire_timeout then get a clean 503 instead of hanging the
    # pool. Keep it well under Starlette's default 40-thread pool so other endpoints always
    # have threads.
    atomize_concurrency: int = 6
    atomize_acquire_timeout_s: float = 20.0

    # --- per-email rate limit ---
    rate_per_min: int = 20

    # --- algorithm param ceilings (user requests clamped to these) ---
    allowed_algorithms: Tuple[str, ...] = ("sft_greedy", "sft_best_of_n", "cds_oneshot", "model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f")
    max_sft_beam_size: int = 10
    max_sft_patience: int = 50      # 0 = off; ceiling for user-supplied --sft-patience
    max_max_steps: int = 80
    max_sample_timeout: float = 180.0     # SFT rollout / CDS decoding wait
    max_per_step_timeout: float = 60.0
    max_gbfs_max_apps: int = 200000   # ceiling for user-supplied gbfs apply budget
    # Ceiling for `gbfs_max_seconds`. The search also has a move budget,
    # `gbfs_max_apps`. `sample_timeout` applies to SFT and CDS. Set this ceiling
    # with `WEBSVC_MAX_GBFS_MAX_SECONDS`. When raising it, also check
    # `job_timeout_s` against ceil(n_algos / max_parallel_algos) *
    # (max_gbfs_max_seconds + algo_grace_s), allowing additional time for queueing
    # and work outside the search loop.
    max_gbfs_max_seconds: float = 120.0
    # Ceiling for `gbfs_max_terms`, the numerator term limit during input expansion
    # and rewriting. The web service replaces a request value of 0 with
    # `DEFAULT_GBFS_MAX_TERMS` (50_000). The cap limits expression growth and may
    # also exclude intermediates needed for a simplification.
    max_gbfs_max_terms: int = 2000000
    # Ceiling for user-supplied gbfs_threads (worker threads per search inside one job).
    # 1 keeps a multi-user deployment fair; raise with WEBSVC_MAX_GBFS_THREADS on a dedicated box.
    max_gbfs_threads: int = 1
    # Ceilings for the per-variant knobs of gbfs_e/gbfs_f (split cap) and gbfs_c (cancel cap).
    max_gbfs_split_max_terms: int = 1000
    max_gbfs_c_cancel_max_terms: int = 5000
    # Grace on top of a backend's own budget before the parallel collector stops waiting.
    algo_grace_s: float = 30.0
    # Wall-clock bound on the sympy post-processing / recounts of a backend's output.
    post_timeout_s: float = 30.0


def load_config() -> Config:
    return Config(
        host=_env_str("WEBSVC_HOST", "127.0.0.1"),
        port=_env_int("WEBSVC_PORT", 8000),
        auth_mode=_env_str("WEBSVC_AUTH_MODE", "cloudflare").strip().lower(),
        cf_team_domain=_env_str("CF_ACCESS_TEAM_DOMAIN", ""),
        cf_access_aud=_env_str("CF_ACCESS_AUD", ""),
        basic_creds=_env_basic_creds("WEBSVC_BASIC_CREDS"),
        basic_max_fails=_env_int("WEBSVC_BASIC_MAX_FAILS", 8),
        basic_fail_window_s=_env_float("WEBSVC_BASIC_FAIL_WINDOW_S", 300.0),
        email_whitelist=_env_set("WEBSVC_EMAIL_WHITELIST"),
        dev_allow_no_auth=_env_bool("WEBSVC_DEV_ALLOW_NO_AUTH", False),
        max_concurrent=_env_int("WEBSVC_MAX_CONCURRENT", 4),
        max_queue=_env_int("WEBSVC_MAX_QUEUE", 32),
        job_timeout_s=_env_float("WEBSVC_JOB_TIMEOUT_S", 1400.0),
        max_parallel_algos=_env_int("WEBSVC_MAX_PARALLEL_ALGOS", 4),
        max_jobs=_env_int("WEBSVC_MAX_JOBS", 500),
        max_input_chars=_env_int("WEBSVC_MAX_INPUT_CHARS", 8000),
        max_digit_run=_env_int("WEBSVC_MAX_DIGIT_RUN", 3),
        atomize_concurrency=_env_int("WEBSVC_ATOMIZE_CONCURRENCY", 6),
        atomize_acquire_timeout_s=_env_float("WEBSVC_ATOMIZE_ACQUIRE_TIMEOUT_S", 20.0),
        rate_per_min=_env_int("WEBSVC_RATE_PER_MIN", 20),
        max_sft_beam_size=_env_int("WEBSVC_MAX_SFT_BEAM_SIZE", 10),
        max_sft_patience=_env_int("WEBSVC_MAX_SFT_PATIENCE", 50),
        max_max_steps=_env_int("WEBSVC_MAX_MAX_STEPS", 80),
        max_sample_timeout=_env_float("WEBSVC_MAX_SAMPLE_TIMEOUT", 180.0),
        max_per_step_timeout=_env_float("WEBSVC_MAX_PER_STEP_TIMEOUT", 60.0),
        max_gbfs_max_apps=_env_int("WEBSVC_MAX_GBFS_MAX_APPS", 200000),
        max_gbfs_max_seconds=_env_float("WEBSVC_MAX_GBFS_MAX_SECONDS", 120.0),
        max_gbfs_max_terms=_env_int("WEBSVC_MAX_GBFS_MAX_TERMS", 2000000),
        max_gbfs_threads=_env_int("WEBSVC_MAX_GBFS_THREADS", 1),
        max_gbfs_split_max_terms=_env_int("WEBSVC_MAX_GBFS_SPLIT_MAX_TERMS", 1000),
        max_gbfs_c_cancel_max_terms=_env_int("WEBSVC_MAX_GBFS_C_CANCEL_MAX_TERMS", 5000),
        algo_grace_s=_env_float("WEBSVC_ALGO_GRACE_S", 30.0),
        post_timeout_s=_env_float("WEBSVC_POST_TIMEOUT_S", 30.0),
    )


CONFIG = load_config()
