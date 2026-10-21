"""Bodhan Indic-Transliterate adapter.

Same chat-shaped endpoint as MT, with `model=indic-transliterate`. Input and
output are the same language — only `target_script` changes (plan §5 /
reference.py). Falls back to the MT endpoint when a dedicated translit
endpoint is not configured.

Env (loaded from `.env`):
  BODHAN_TRANSLIT_URL   (default: BODHAN_MT_URL or https://api.bodhan.ai/v1)
  BODHAN_TRANSLIT_KEY   (default: BODHAN_MT_KEY)
  BODHAN_TRANSLIT_MODEL (default: indic-transliterate)
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from .base import BaseTransliterator, TranslitRequest, TranslitResult, TranslitSegment

log = logging.getLogger(__name__)

DEFAULT_URL = "https://api.bodhan.ai/v1"
DEFAULT_MODEL = "indic-transliterate"


class BodhanTransliterate(BaseTransliterator):
    name = "bodhan"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 10,
        min_interval_s: float = 1.2,
        max_backoff_s: float = 60.0,
    ):
        super().__init__(device=device)
        self.base_url = (
            base_url
            or os.getenv("BODHAN_TRANSLIT_URL")
            or os.getenv("BODHAN_MT_URL")
            or DEFAULT_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("BODHAN_TRANSLIT_KEY")
            or os.getenv("BODHAN_MT_KEY")
        )
        self.model = model or os.getenv("BODHAN_TRANSLIT_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_retries = int(os.getenv("BODHAN_MAX_RETRIES", max_retries))
        self.min_interval_s = float(os.getenv("BODHAN_MIN_INTERVAL_S", min_interval_s))
        self.max_backoff_s = float(os.getenv("BODHAN_MAX_BACKOFF_S", max_backoff_s))
        self._session = None
        self._last_post_ts = 0.0

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.api_key:
            raise RuntimeError(
                "BODHAN_TRANSLIT_KEY (or BODHAN_MT_KEY) is not set. "
                "Add it to .env or pass api_key=…"
            )
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("`pip install requests` for the Bodhan adapter.") from e
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self._loaded = True
        log.info("bodhan translit: base_url=%s model=%s", self.base_url, self.model)

    def transliterate(
        self,
        requests: List[TranslitRequest],
        *,
        target_language: str,
        target_script: str = "native",
    ) -> TranslitResult:
        self._ensure_loaded()
        out: List[TranslitSegment] = []
        for req in requests:
            text = (req.text or "").strip()
            if not text:
                out.append(TranslitSegment(
                    id=req.id, source_text=text, text="",
                    source_language=req.source_language,
                    source_script=req.source_script,
                    target_script=req.target_script or target_script,
                ))
                continue
            scripted = self._post_one(
                text,
                target_language=target_language,
                target_script=req.target_script or target_script,
            )
            out.append(TranslitSegment(
                id=req.id, source_text=text, text=scripted,
                source_language=req.source_language,
                source_script=req.source_script,
                target_script=req.target_script or target_script,
            ))
        return TranslitResult(
            model_name=self.model,
            target_script=target_script,
            segments=out,
            extra={"endpoint": self.base_url},
        )

    # ------------------------------------------------------------------

    def _post_one(self, text: str, *, target_language: str, target_script: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "target_language_code": target_language,
            "target_script": target_script,
        }
        for attempt in range(self.max_retries):
            self._pace()
            resp = self._session.post(url, json=payload, timeout=self.timeout_s)
            self._last_post_ts = time.monotonic()

            if resp.status_code == 429:
                wait = self._backoff(attempt, resp.headers.get("retry-after"))
                log.warning("bodhan translit: 429 on attempt %d, sleeping %.1fs",
                            attempt + 1, wait)
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                wait = self._backoff(attempt, None)
                log.warning("bodhan translit: HTTP %d on attempt %d, backoff %.1fs",
                            resp.status_code, attempt + 1, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Bodhan translit failed: HTTP {resp.status_code} — "
                    f"{resp.text[:400]}"
                )
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(
                    f"Bodhan translit response malformed: {data}"
                ) from e
        raise RuntimeError(
            f"Bodhan translit failed after {self.max_retries} retries at {url}"
        )

    def _pace(self) -> None:
        gap = time.monotonic() - self._last_post_ts
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
        return min(2 ** attempt, self.max_backoff_s)
