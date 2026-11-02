"""Runtime configuration. Every setting comes from the environment, so nothing sensitive
such as the email whitelist or the Access AUD is hard-coded into the image.

It is read once at import through `CONFIG = load_config()`. The limits here are the hard
limits. User-supplied params such as beam size, max_steps and timeouts are clamped to them
in app.py before any job is enqueued.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

from ..simplifier.input_limits import DEFAULT_INPUT_LIMITS, LIMIT_CONTROLS, InputLimits


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
    """Parse WEBSVC_BASIC_CREDS = 'email:pbkdf2hash,email2:pbkdf2hash2'. Each entry is split
    on its first ':', since an email has no ':' and the hash has no ','."""
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
    # publishes the port on the host's loopback address.
    host: str = "127.0.0.1"
    port: int = 8000

    # --- auth backend selection ---
    # "cloudflare" verifies the Cloudflare Access signed JWT, the production target. "basic" is
    # HTTP Basic auth for when Access is unavailable. "dev" is no auth, for localhost only, and
    # dev_allow_no_auth also enables it.
    auth_mode: str = "cloudflare"

    # --- Cloudflare Access, auth_mode="cloudflare" ---
    # The JWKS issuer, such as "myteam.cloudflareaccess.com".
    cf_team_domain: str = ""
    # The Access application AUD tag from the Cloudflare dashboard. The JWT audience must match.
    cf_access_aud: str = ""

    # --- HTTP Basic, auth_mode="basic", email -> pbkdf2 hash ---
    basic_creds: Dict[str, str] = field(default_factory=dict)
    # Lock out an IP after this many failed Basic logins within the window.
    basic_max_fails: int = 8
    basic_fail_window_s: float = 300.0

    # --- email whitelist ---
    # Cloudflare Access checks the email first. It is re-checked here in case the tunnel is
    # bypassed or misconfigured.
    email_whitelist: FrozenSet[str] = field(default_factory=frozenset)

    # --- development only, skipping JWT verification for localhost testing ---
    dev_allow_no_auth: bool = False

    # --- queue and concurrency limits ---
    max_concurrent: int = 4        # simultaneous running jobs, which are CPU-bound in sympy
    max_queue: int = 32            # queued plus running jobs before a 429
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
    # The bound on the in-memory job store. Jobs would otherwise live forever and a long-running
    # server would leak memory as finished jobs pile up. Past this, the oldest finished jobs,
    # done, error, timeout or cancelled, are evicted first, and queued or running ones never. A
    # client polling an evicted job gets 404 and re-runs. The bound is generous so only
    # long-stale results are dropped. max_queue separately caps the unfinished jobs.
    max_jobs: int = 500

    # --- input limits, shared with the CLI through ampzip.simplifier.input_limits ---
    # Checked on the text before parsing. The CLI sets the same bounds with --max-* flags.
    max_input_chars: int = DEFAULT_INPUT_LIMITS.max_input_chars
    max_int_digits: int = DEFAULT_INPUT_LIMITS.max_int_digits
    max_exponent: int = DEFAULT_INPUT_LIMITS.max_exponent
    max_power_nesting: int = DEFAULT_INPUT_LIMITS.max_power_nesting
    max_nesting_depth: int = DEFAULT_INPUT_LIMITS.max_nesting_depth
    max_number_digits: int = DEFAULT_INPUT_LIMITS.max_number_digits

    # --- atomize and validate concurrency bound ---
    # atomize() and validate() run synchronously in the request thread and are CPU-bound in
    # sympy and the AST walk. They cannot be killed in-thread, so a burst of large inputs could
    # otherwise occupy every Starlette worker thread and make the whole service unresponsive,
    # /healthz and job polling included. This semaphore bounds how many run at once. Excess
    # submits wait up to acquire_timeout and then get a 503. It stays well under Starlette's
    # default pool of 40 threads so other endpoints always have threads.
    atomize_concurrency: int = 6
    atomize_acquire_timeout_s: float = 20.0

    # --- per-email rate limit ---
    rate_per_min: int = 20

    # --- algorithm param limits, which user requests are clamped to ---
    allowed_algorithms: Tuple[str, ...] = ("sft_greedy", "sft_best_of_n", "cds_oneshot", "model_free_gbfs", "gbfs_c", "gbfs_d", "gbfs_e", "gbfs_f")
    max_sft_beam_size: int = 10
    max_sft_patience: int = 50      # limit for a user-supplied --sft-patience, 0 being off
    max_max_steps: int = 80
    max_sample_timeout: float = 180.0     # the SFT rollout and CDS decoding wait
    max_per_step_timeout: float = 60.0
    max_gbfs_max_apps: int = 200000   # limit for a user-supplied gbfs move budget
    # Limit for `gbfs_max_seconds`. The search also has a move budget,
    # `gbfs_max_apps`. `sample_timeout` applies to SFT and CDS. Set this limit
    # with `WEBSVC_MAX_GBFS_MAX_SECONDS`. When raising it, also check
    # `job_timeout_s` against ceil(n_algos / max_parallel_algos) *
    # (max_gbfs_max_seconds + algo_grace_s), allowing additional time for queueing
    # and work outside the search loop.
    max_gbfs_max_seconds: float = 120.0
    # Limit on `gbfs_max_terms`, the numerator term limit during input expansion and
    # rewriting. The web service replaces a request value of 0 with `DEFAULT_GBFS_MAX_TERMS`,
    # 50_000. The cap limits expression growth and may also exclude intermediates needed for
    # a simplification.
    max_gbfs_max_terms: int = 2000000
    # Limit on a user-supplied gbfs_threads, the worker threads per search inside one job.
    # 1 keeps a multi-user deployment fair. A dedicated machine raises it with
    # WEBSVC_MAX_GBFS_THREADS.
    max_gbfs_threads: int = 1
    # Limits on the split cap of gbfs_e and gbfs_f and the cancel cap of gbfs_c.
    max_gbfs_split_max_terms: int = 1000
    max_gbfs_c_cancel_max_terms: int = 5000
    # Grace on top of a backend's own budget before the parallel collector stops waiting.
    algo_grace_s: float = 30.0
    # Wall-clock bound on the sympy post-processing and recounts of a backend's output.
    post_timeout_s: float = 30.0

    @property
    def input_limits(self) -> InputLimits:
        """The input limits as `check_input` takes them."""
        return InputLimits(**{name: getattr(self, name) for name in LIMIT_CONTROLS})


def load_config() -> Config:
    cfg = Config(
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
        **{name: _env_int(env, getattr(DEFAULT_INPUT_LIMITS, name))
           for name, (_flag, env) in LIMIT_CONTROLS.items()},
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
    cfg.input_limits  # noqa: B018  (a bad WEBSVC_MAX_* value fails at start-up)
    return cfg


CONFIG = load_config()
