"""Shared constructor-time setup for the Bodhan adapters.

Env resolution, the authenticated session, the pacing floor and the
`Retry-After` aware backoff. The retry LOOPS stay in each adapter, where their
different 4xx diagnoses are readable. Nothing here makes a request.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

#: Every Bodhan service lives behind this host unless told otherwise.
DEFAULT_URL = "https://api.bodhan.ai/v1"


class BodhanClientMixin:
    """Constructor-time setup, pacing and backoff for a Bodhan adapter.

    Mixed in beside the stage's own `Base*` class. It deliberately defines no
    `__init__`: `models._registry`'s kwarg filter reads
    `inspect.signature(cls.__init__)`, so a `**kwargs` passthrough here would
    make every adapter claim to accept everything.
    """

    # Set by `_init_bodhan`; declared for readers.
    base_url: str
    api_key: Optional[str]
    model: str
    timeout_s: float
    max_retries: int
    min_interval_s: float
    max_backoff_s: float

    def _init_bodhan(
        self,
        *,
        env_prefix: str,
        default_model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 10,
        min_interval_s: float = 1.2,
        max_backoff_s: float = 60.0,
        url_fallback_env: Optional[str] = None,
        key_fallback_env: Optional[str] = None,
    ) -> None:
        """Resolve the endpoint from argument, then env, then default.

        `url_fallback_env` / `key_fallback_env` exist for the transliteration
        adapter, which accepts the MT credentials when it has none of its own.
        The retry/pacing knobs are read UNPREFIXED: they describe how hard to
        lean on the host, not on one service.
        """
        url_env = [f"BODHAN_{env_prefix}_URL"]
        key_env = [f"BODHAN_{env_prefix}_KEY"]
        if url_fallback_env:
            url_env.append(url_fallback_env)
        if key_fallback_env:
            key_env.append(key_fallback_env)

        self.base_url = (
            base_url or _first_env(url_env) or DEFAULT_URL
        ).rstrip("/")
        self.api_key = api_key or _first_env(key_env)
        self.model = (model or os.getenv(f"BODHAN_{env_prefix}_MODEL")
                      or default_model)
        self.timeout_s = timeout_s
        self.max_retries = int(os.getenv("BODHAN_MAX_RETRIES", max_retries))
        self.min_interval_s = float(
            os.getenv("BODHAN_MIN_INTERVAL_S", min_interval_s))
        self.max_backoff_s = float(
            os.getenv("BODHAN_MAX_BACKOFF_S", max_backoff_s))
        self._session = None
        self._last_post_ts = 0.0

    def _ensure_bodhan_session(
        self,
        *,
        missing_key_message: str,
        accept: str = "application/json",
        content_type: Optional[str] = "application/json",
    ) -> bool:
        """Build the session once. Returns False if it was already built.

        `content_type=None` is for the multipart ASR adapter, where `requests`
        must set the boundary itself. A missing key raises here rather than at
        the first request, so the failure names the variable instead of
        arriving as a 401 per segment.
        """
        if self._loaded:
            return False
        if not self.api_key:
            raise RuntimeError(missing_key_message)
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "`pip install requests` for the Bodhan adapter.") from e
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": accept,
        }
        if content_type:
            headers["Content-Type"] = content_type
        self._session = requests.Session()
        self._session.headers.update(headers)
        self._loaded = True
        return True

    def _pace(self) -> None:
        """Hold a floor between consecutive requests from this instance, or
        serial per-turn calls burst past the server's per-second budget and
        spend the run in 429 backoff."""
        gap = time.monotonic() - self._last_post_ts
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        """Seconds to wait before attempt `attempt + 1`.

        `Retry-After` wins when the server sends a parseable one, floored at
        1 s; otherwise 2**attempt, capped at `max_backoff_s`.
        """
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
        return min(2 ** attempt, self.max_backoff_s)

    def _log_ratelimit_headers(self, resp, *, stage: str,
                               level: int = logging.INFO) -> None:
        """Surface `x-ratelimit-*` whenever the server sends them.

        Bodhan only ships them on MT today; logging on sight means a later
        ASR/TTS rollout needs no adapter change.
        """
        pairs = [(k, v) for k, v in resp.headers.items()
                 if k.lower().startswith("x-ratelimit")]
        if pairs:
            log.log(level, "bodhan %s rate-limit headers: %s", stage,
                    ", ".join(f"{k}={v}" for k, v in pairs))


def _first_env(names) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None
