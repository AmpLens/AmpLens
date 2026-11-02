"""The AmpZip web service, the one module that faces the public internet.

It is a thin wrapper around the `simplifier` package, which handles only input the service
has checked.

    public user
       │  Google email login
    [Cloudflare Access]   email allowlist, DDoS and WAF protection, hidden origin IP
       │  signed JWT, via the cloudflared tunnel to 127.0.0.1
    ┌───────────────── webservice, untrusted input ──────────────────┐
    │  auth.py     verifies the Access JWT signature and the allowlist │
    │  safety.py   applies the input limits before parsing             │
    │  jobs.py     bounds the queue, concurrency and timeouts          │
    └───────────────────────────┬───────────────────────────────────┘
                                │ checked input only
                    [ simplifier package ]   the simplification itself

Every request passes authentication, the input limits, validation and parameter clamping
before any backend runs. `simplifier` never imports anything from `webservice`.

Run `python -m ampzip.webservice`. It binds 127.0.0.1, with cloudflared in front.
"""

from .._version import __version__  # noqa: F401
