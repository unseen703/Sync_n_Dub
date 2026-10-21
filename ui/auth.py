"""Bind guard + HTTP basic auth for the operator UI.

Ported from `frontend/app.py::_launch`, which was the only auth thinking in
either frontend before this. Its reasoning is worth restating, because it is
the reason this module exists rather than being an afterthought:

    this UI runs yt-dlp against arbitrary user-supplied URLs and starts GPU
    jobs, so an unauthenticated LAN-exposed instance is an SSRF / GPU-abuse
    vector.

Gradio took `auth=(user, pass)` and did the enforcing itself. FastAPI does
not, so porting only the refuse-to-bind half would leave a guard that refuses
to start and then protects nothing once started. Both halves live here:

  `require_auth_for_nonloopback(host)`  — the refusal, called before uvicorn
                                          binds anything.
  `basic_auth`                          — the enforcement, a dependency that
                                          every mutating route carries.

The supported exposure model is still **loopback + an SSH tunnel**. HTTP
basic over plain HTTP on a LAN is weak; the host allowlist in the job-creation
endpoint and this guard are defence in depth, not a substitute for it.
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

log = logging.getLogger("ui.auth")

USER_ENV = "DUBBING_UI_USER"
PASS_ENV = "DUBBING_UI_PASS"

#: Hosts that mean "this machine only". Anything else is a public bind as far
#: as the guard is concerned — including a specific LAN IP, which is the case
#: someone reaches for when they want a teammate to look at a render.
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}

_security = HTTPBasic(auto_error=False)


def credentials() -> Optional[Tuple[str, str]]:
    """The configured user/password, or None when auth is not set up."""
    user = os.environ.get(USER_ENV) or ""
    pw = os.environ.get(PASS_ENV) or ""
    return (user, pw) if user and pw else None


def is_loopback(host: str) -> bool:
    return (host or "").strip() in _LOOPBACK


def require_auth_for_nonloopback(host: str) -> None:
    """Refuse to bind a public address without credentials.

    Raises SystemExit — this runs before uvicorn starts, and a process that
    exits is the correct outcome. Verbatim in spirit from `frontend/app.py`.
    """
    if is_loopback(host):
        return
    if credentials() is None:
        raise SystemExit(
            f"Refusing to bind to a non-loopback address ({host}) without auth.\n"
            f"  This UI runs yt-dlp against arbitrary user-supplied URLs and\n"
            f"  starts GPU jobs. Set {USER_ENV} and {PASS_ENV} in the\n"
            f"  environment, or drop --host to bind loopback only.\n"
            f"  Preferred: keep it on loopback and reach it over an SSH tunnel:\n"
            f"    ssh -L 7861:localhost:7861 <this-host>"
        )
    log.warning(
        "UI bound to %s with HTTP basic auth over plain HTTP. This UI runs "
        "yt-dlp against arbitrary URLs — do not expose it to untrusted "
        "networks.", host,
    )


def _check(creds: Optional[HTTPBasicCredentials]) -> None:
    configured = credentials()
    if configured is None:
        # No credentials configured at all. On loopback that is the documented
        # default; the bind guard is what stops this state from being reachable
        # from the network.
        return
    if creds is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(creds.username, configured[0])
    pass_ok = secrets.compare_digest(creds.password, configured[1])
    # Both compared before branching, so the reply time does not say which
    # half was wrong.
    if not (user_ok and pass_ok):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def basic_auth(creds: Optional[HTTPBasicCredentials] = Depends(_security)) -> None:
    """Dependency for routes that must not be reachable anonymously.

    A no-op when no credentials are configured (the loopback default), so
    local development is unchanged.
    """
    _check(creds)
