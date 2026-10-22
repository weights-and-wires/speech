"""Authentication via identity headers set by the trusted Caddy proxy."""

import hmac

from fastapi import Depends, HTTPException, Request, status

from .config import Settings, get_settings


def sso_user(request: Request, settings: Settings) -> str | None:
    """Return the identity vouched for by Caddy, or None.

    Caddy must run Authelia forward_auth before proxying and overwrite these
    headers. The shared secret makes a direct request to the app insufficient.
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


def current_user(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """FastAPI dependency returning the SSO username, or raising 401."""
    name = sso_user(request, settings)
    if not name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in"
        )
    return name

