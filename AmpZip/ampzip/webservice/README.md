# AmpZip web service

The web service exposes the AmpZip simplifier through an HTTP API.

```text
                                   User
                                     │
                                     ▼
┌────────────────────────── ampzip.webservice ───────────────────────────┐
│ auth.py            Verify identity (Cloudflare Access or HTTP Basic)   │
│ ratelimit.py       Limit submissions and numerical checks per user     │
│ safety.py          Check input length, characters and numeric literals │
│ ampzip.simplifier  Convert notation and validate input                 │
│ app.py             Cap parameters and select compatible backends       │
│ jobs.py            Manage the queue, concurrency, results and timeouts │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
                         Selected AmpZip backends
```

Authentication can be disabled explicitly for local development, as shown below.

## Run

```bash
WEBSVC_AUTH_MODE=dev \
WEBSVC_DEV_ALLOW_NO_AUTH=1 \
python -m ampzip.webservice  # http://127.0.0.1:8000

curl -s localhost:8000/healthz

curl -s -X POST localhost:8000/api/jobs \
  -H 'content-type: application/json' \
  -d '{
    "expr": "ab(1,2)**3/(ab(2,3)*ab(3,4)*ab(4,1))",
    "algorithms": ["model_free_gbfs"]
  }'

curl -s "localhost:8000/api/jobs/<job_id>"
```

Replace `<job_id>` with the ID returned by the submission request.

| Endpoint | Description |
|---|---|
| `GET /` | Open the web interface. |
| `GET /healthz` | Check service health. |
| `GET /api/limits` | View server limits. |
| `POST /api/jobs` | Submit a simplification job. |
| `GET /api/jobs/{job_id}` | Get job status and results. |
| `POST /api/jobs/{job_id}/cancel` | Request job cancellation. |
| `POST /api/numerical_check` | Compare two expressions numerically. |

`POST /api/numerical_check` evaluates two expressions at two precomputed on-shell configurations using floating-point arithmetic. It returns `check_status` as `matched`, `mismatch`, or `inconclusive`. Singular or non-finite evaluations give `inconclusive`.

The `equiv` field is `true` for `matched` and `false` for `mismatch`. For `inconclusive`, both `equiv` and `rel_diff` are `null`.

A match provides numerical evidence of equality.

## Configuration

Set the following environment variables before starting the service.

| Variable | Description |
|---|---|
| `WEBSVC_HOST`, `WEBSVC_PORT` | Listening address and port. Keep `127.0.0.1` unless access is restricted to a tunnel or proxy. |
| `WEBSVC_AUTH_MODE` | Authentication method: `cloudflare`, `basic`, or `dev`. |
| `CF_ACCESS_TEAM_DOMAIN`, `CF_ACCESS_AUD`, `WEBSVC_EMAIL_WHITELIST` | Cloudflare Access settings and allowed email addresses. |
| `WEBSVC_BASIC_CREDS`, `WEBSVC_BASIC_MAX_FAILS`, `WEBSVC_BASIC_FAIL_WINDOW_S` | HTTP Basic credentials, failed login limit and time window. |
| `WEBSVC_MAX_CONCURRENT` | Maximum number of jobs running concurrently. |
| `WEBSVC_MAX_QUEUE` | Maximum combined number of queued and running jobs. |
| `WEBSVC_JOB_TIMEOUT_S` | Time in seconds before a job is marked `timeout`. |
| `WEBSVC_RATE_PER_MIN` | Shared limit on job submissions and numerical checks per user per minute. |
| `WEBSVC_MAX_GBFS_MAX_SECONDS`, `WEBSVC_MAX_GBFS_MAX_APPS`, `WEBSVC_MAX_GBFS_MAX_TERMS`, `WEBSVC_MAX_GBFS_THREADS`, `WEBSVC_MAX_GBFS_SPLIT_MAX_TERMS`, `WEBSVC_MAX_GBFS_C_CANCEL_MAX_TERMS` | Upper limits on GBFS request parameters. Current limits are available from `GET /api/limits`, and parameter adjustments are reported in the submission response. |
| `WEBSVC_MAX_INPUT_CHARS` | Maximum input length in characters. Longer inputs are rejected. |
| `WEBSVC_ALGO_GRACE_S` | Additional time in seconds to wait for a backend after its time budget expires. |
| `WEBSVC_POST_TIMEOUT_S` | Time limit in seconds for SymPy post-processing of each backend's output. |
| `WEBSVC_MAX_PARALLEL_ALGOS` | Maximum number of backends running concurrently within each job. |
| `AMPZIP_LOG_DIR`, `WEBSVC_LOG_DIR` | Directory for `webservice.jsonl`. `WEBSVC_LOG_DIR` overrides `AMPZIP_LOG_DIR`. |
| `AMPZIP_BACKENDS` | Location of the backend code installed by `ampzip fetch-backends`. |

See [webservice.env.example](deploy/webservice.env.example) for a deployment example and [config.py](config.py) for server settings and defaults.

## Saved inputs and results

The web interface saves the main form's inputs and results from up to 20 recent jobs in your browser. Reloading the page restores the latest saved result. If a job is still running, reloading the same tab resumes its status updates.

Use “saved in this browser” to view earlier results and “forget all” to clear the saved history.

## Service limitations

- Run the service with one worker process. Job records are stored in memory and are lost when the service restarts.
- When `WEBSVC_JOB_TIMEOUT_S` is reached, the job is marked `timeout`, but backend computation may continue. See [SECURITY.md](../../docs/SECURITY.md) for details.
- For search memory usage and limitations, see [Known issues](../../docs/KNOWN_ISSUES.md).
