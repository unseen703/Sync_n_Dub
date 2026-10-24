"""Bodhan Indic-Transliterate adapter.

Same chat-shaped endpoint as MT, with `model=indic-transliterate`. Input and
output are the same language — only `target_script` changes (plan §5; the
`reference.py` snippet this used to cite was deleted on 2026-10-23, the live
shape is in `_post_one` below). Falls back to the MT endpoint when a
dedicated translit endpoint is not configured.

Env (loaded from `.env`):
  BODHAN_TRANSLIT_URL   (default: BODHAN_MT_URL or https://api.bodhan.ai/v1)
  BODHAN_TRANSLIT_KEY   (default: BODHAN_MT_KEY)
  BODHAN_TRANSLIT_MODEL (default: indic-transliterate)
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from .._bodhan import BodhanClientMixin
from .base import BaseTransliterator, TranslitRequest, TranslitResult, TranslitSegment

log = logging.getLogger(__name__)

DEFAULT_MODEL = "indic-transliterate"


class BodhanTransliterate(BodhanClientMixin, BaseTransliterator):
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
        # Transliteration shares MT's host, and most installs only ever set
        # the MT pair — so `BODHAN_MT_URL` / `BODHAN_MT_KEY` are accepted as
        # the fallback for this service's own variables.
        self._init_bodhan(
            env_prefix="TRANSLIT", default_model=DEFAULT_MODEL,
            base_url=base_url, api_key=api_key, model=model,
            timeout_s=timeout_s, max_retries=max_retries,
            min_interval_s=min_interval_s, max_backoff_s=max_backoff_s,
            url_fallback_env="BODHAN_MT_URL",
            key_fallback_env="BODHAN_MT_KEY",
        )

    def _ensure_loaded(self) -> None:
        if not self._ensure_bodhan_session(
            missing_key_message=(
                "BODHAN_TRANSLIT_KEY (or BODHAN_MT_KEY) is not set. "
                "Add it to .env or pass api_key=…"
            ),
        ):
            return
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

