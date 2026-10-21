"""AmpZip — public-facing web service (the *exposed* module).

This is the ONLY module that touches the public internet and untrusted input.
It is a thin, hardened wrapper around the trusted `simplifier` package:

    public user
       │  (Google email login)
    [Cloudflare Access]   ── email-whitelist auth + DDoS + WAF + hides origin IP
       │  signed JWT, via cloudflared tunnel → 127.0.0.1
    ┌───────────────── webservice (UNTRUSTED zone) ─────────────────┐
    │  auth.py     — verify Access JWT signature + email allowlist   │
    │  safety.py   — pre-sympify char whitelist (RCE guard)          │
    │  jobs.py     — bounded queue: concurrency / timeout / caps     │
    └───────────────────────────┬───────────────────────────────────┘
                                │ only sanitized input crosses
                    [ simplifier package (TRUSTED) ]  — pure simplification

Trust-boundary rule: every request must pass auth → safety → validate →
param-clamp BEFORE any backend runs. `simplifier` never imports anything from
`webservice`; the dependency edge is strictly webservice → simplifier.

Run:  python -m ampzip.webservice           (binds 127.0.0.1; put cloudflared in front)
"""

from .._version import __version__  # noqa: F401
