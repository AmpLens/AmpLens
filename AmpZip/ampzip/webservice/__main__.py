"""Run the web service at WEBSVC_HOST and WEBSVC_PORT.

For public deployment, see docs/DEPLOY.md and docs/SECURITY.md.
"""

from __future__ import annotations

import uvicorn

from .config import CONFIG


def main() -> None:
    if CONFIG.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[webservice] WARNING: Listening on {CONFIG.host}. For public deployments, "
              "restrict access to the configured reverse proxy or tunnel.")
    if CONFIG.dev_allow_no_auth:
        print("[webservice] WARNING: Authentication is disabled by WEBSVC_DEV_ALLOW_NO_AUTH=1. "
              "Use this setting only for local access.")
    uvicorn.run(
        "ampzip.webservice.app:app",
        host=CONFIG.host,
        port=CONFIG.port,
        workers=1,           # a single process, since the in-memory job store is not shared
        log_level="info",
    )


if __name__ == "__main__":
    main()
