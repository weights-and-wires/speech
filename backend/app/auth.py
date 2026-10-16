"""Username and password login, backed by a signed HTTP-only session cookie.

Two people share one password. The username is not a secret and carries no
authority of its own: it scopes progress and saved drafts, so that two readers
working through the course keep separate state.

A browser that is signed in to the single sign-on portal at auth.ifkash.dev
skips the form: Caddy vouches for the user with the Remote-User header, and
`read_session` accepts it ahead of the cookie. See `sso_user`.
"""

import hmac
import time

from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import Settings, get_settings

COOKIE_NAME = "speech_session"


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.session_secret, salt="weights-and-wires-speech")


def verify_credentials(username: str, password: str, settings: Settings) -> bool:
    """Check a login attempt in constant time with respect to the password."""
    known = username.strip().lower() in {u.lower() for u in settings.users}
    correct = hmac.compare_digest(password, settings.app_password)
    # Both comparisons always run, so a wrong username and a wrong password take
    # the same time to reject.
    return known and correct


def issue_session(response: Response, username: str, settings: Settings) -> None:
    token = _serializer(settings).dumps({"name": username, "at": time.time()})
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def clear_session(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        COOKIE_NAME,
        path="/",
        samesite="lax",
        secure=settings.cookie_secure,
        httponly=True,
    )


def sso_user(request: Request, settings: Settings) -> str | None:
    """Return the single sign-on user that Caddy vouches for, or None.

    Caddy drops any Remote-User and X-SSO-Proxy headers the client sends, and
    sets both only when Authelia accepts the browser's session. The secret
    rejects a request that reaches the app without going through Caddy.
    """
    secret = settings.sso_proxy_secret
    if not secret:
        return None
    sent = request.headers.get("x-sso-proxy", "")
    if not hmac.compare_digest(sent.encode(), secret.encode()):
        return None
    name = request.headers.get("remote-user", "").strip()
    if not name:
        return None
    return settings.sso_names.get(name.lower(), name)


def read_session(request: Request, settings: Settings) -> str | None:
    name = sso_user(request, settings)
    if name:
        return name
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        data = _serializer(settings).loads(
            token, max_age=settings.session_days * 86400
        )
    except (BadSignature, SignatureExpired):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None
    # A username removed from the allowlist loses its session at the next request.
    if name.lower() not in {u.lower() for u in settings.users}:
        return None
    return name


def current_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """FastAPI dependency returning the signed-in username, or raising 401."""
    name = read_session(request, settings)
    if not name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in"
        )
    return name
