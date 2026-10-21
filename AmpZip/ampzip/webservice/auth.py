"""Authentication for the web service.

  cloudflare : verify signed Cloudflare Access tokens.
  basic      : verify HTTP Basic credentials against hashes in WEBSVC_BASIC_CREDS.
  dev        : skip authentication for local use.

Returns the authenticated email for request handling. See docs/SECURITY.md#authentication for deployment settings.

The Cloudflare Access mode requires PyJWT[crypto].
"""

from __future__ import annotations

import base64
import binascii
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from .config import CONFIG
from .pwhash import verify_password

try:
    import jwt
    from jwt import PyJWKClient
    _JWT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JWT_AVAILABLE = False


_ACCESS_HEADER = "Cf-Access-Jwt-Assertion"
_jwks_client: Optional["PyJWKClient"] = None


class AuthError(Exception):
    """Auth failure carrying an HTTP status and optional response headers
    (e.g. WWW-Authenticate to trigger the browser Basic-auth prompt)."""

    def __init__(self, status: int, detail: str, headers: Optional[Dict[str, str]] = None):
        self.status = status
        self.detail = detail
        self.headers = headers or {}
        super().__init__(detail)


# --- entry point -------------------------------------------------------------
def verify_request_email(headers, client_ip: str = "?") -> str:
    """Return the authenticated, authorized email or raise AuthError."""
    mode = "dev" if CONFIG.dev_allow_no_auth else CONFIG.auth_mode

    if mode == "dev":
        return "dev@localhost"
    if mode == "basic":
        return _verify_basic(headers, client_ip)
    if mode == "cloudflare":
        return _verify_cloudflare(headers)
    raise AuthError(500, f"unknown WEBSVC_AUTH_MODE '{CONFIG.auth_mode}'")


# --- interim HTTP Basic ------------------------------------------------------
_fails: Dict[str, Deque[float]] = defaultdict(deque)
_fails_lock = threading.Lock()
_BASIC_CHALLENGE = {"WWW-Authenticate": 'Basic realm="simplifier", charset="UTF-8"'}


def _throttled(ip: str) -> bool:
    now = time.time()
    start = now - CONFIG.basic_fail_window_s
    with _fails_lock:
        dq = _fails[ip]
        while dq and dq[0] < start:
            dq.popleft()
        return len(dq) >= CONFIG.basic_max_fails


def _record_fail(ip: str) -> None:
    with _fails_lock:
        _fails[ip].append(time.time())


def _verify_basic(headers, client_ip: str) -> str:
    if not CONFIG.basic_creds:
        raise AuthError(500, "WEBSVC_AUTH_MODE=basic but WEBSVC_BASIC_CREDS is empty")

    if _throttled(client_ip):
        raise AuthError(429, "too many failed logins; try again later")

    raw = headers.get("Authorization") or headers.get("authorization")
    if not raw or not raw.lower().startswith("basic "):
        # No credentials yet → challenge the browser to show its login dialog.
        raise AuthError(401, "authentication required", headers=_BASIC_CHALLENGE)

    try:
        decoded = base64.b64decode(raw.split(" ", 1)[1].strip()).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (ValueError, binascii.Error, UnicodeDecodeError):
        _record_fail(client_ip)
        raise AuthError(401, "malformed Basic credentials", headers=_BASIC_CHALLENGE)

    email = username.strip().lower()
    stored = CONFIG.basic_creds.get(email)
    # Always run verify (even when the email is unknown, against a dummy) to keep
    # timing uniform and avoid leaking which emails exist.
    ok = verify_password(stored, password) if stored else verify_password(
        "pbkdf2_sha256$200000$00$00", password)
    if not stored or not ok:
        _record_fail(client_ip)
        raise AuthError(401, "invalid email or secret", headers=_BASIC_CHALLENGE)

    return email


# --- Cloudflare Access (unchanged) -------------------------------------------
def _certs_url() -> str:
    return f"https://{CONFIG.cf_team_domain}/cdn-cgi/access/certs"


def _issuer() -> str:
    return f"https://{CONFIG.cf_team_domain}"


def _get_jwks_client() -> "PyJWKClient":
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(_certs_url())
    return _jwks_client


def _verify_cloudflare(headers) -> str:
    if not _JWT_AVAILABLE:
        raise AuthError(500, "PyJWT not installed; run `pip install 'pyjwt[crypto]'`")
    if not CONFIG.cf_team_domain or not CONFIG.cf_access_aud:
        raise AuthError(500, "Cloudflare Access not configured "
                             "(set CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD)")

    token = headers.get(_ACCESS_HEADER)
    if not token:
        raise AuthError(401, f"missing {_ACCESS_HEADER} header "
                             f"(request did not come through Cloudflare Access)")

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, signing_key.key, algorithms=["RS256"],
            audience=CONFIG.cf_access_aud, issuer=_issuer(),
        )
    except Exception as e:
        raise AuthError(401, f"invalid Access token: {type(e).__name__}: {e}")

    email = (claims.get("email") or "").strip().lower()
    if not email:
        raise AuthError(403, "token carries no email claim")
    if CONFIG.email_whitelist and email not in CONFIG.email_whitelist:
        raise AuthError(403, f"email '{email}' is not on the service whitelist")
    return email
