"""Bodhan Indic-Translate adapter.

OpenAI-chat-shaped endpoint — POST /v1/chat/completions with
`model=indic-translate`, `target_language_code`, `target_script`. One call
per segment: the API is single-utterance, and segment-level translation is
what the manifest contract wants (plan §3).

Env (loaded from `.env`):
  BODHAN_MT_URL    (default: https://api.bodhan.ai/v1)
  BODHAN_MT_KEY    (required)
  BODHAN_MT_MODEL  (default: indic-translate)

Target script is a knob the duration-adjustment and code-mix stages
(§6, §7) read: "native" (Deva/Taml/…), "roman" (Latin), "codemix" (English
technical terms kept in Latin inside the target-script sentence), or server
default when `None`.

`codemix` is the DEFAULT here. Two reasons, both measured on this corpus:

  * `native` re-scripts inconsistently. On one 30 s Hindi clip of the NAG
    lecture, "WT" came back as `डब्ल्यूटी` in one cue, `WT` in another and
    `W` in a third -- the same term, three renderings, inside half a minute.
    Downstream TTS does not reliably pronounce a bare Latin remnant, so that
    inconsistency reaches the listener.
  * Technical lecture material in Indian classrooms is genuinely spoken
    code-mixed ("weight सही है", "derivative की गणना करें"). `codemix` is
    closer to the source register than a fully re-scripted translation.

Pass `target_script="native"` explicitly to get the old behaviour.
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

from backend.sanitize import sanitize_mt_output

from .base import BaseMT, MTRequest, MTResult, MTSegment

log = logging.getLogger(__name__)


DEFAULT_URL = "https://api.bodhan.ai/v1"
DEFAULT_MODEL = "indic-translate"
# Applied when neither the request nor the call names a script. See the module
# docstring for why this is `codemix` rather than the server default.
DEFAULT_TARGET_SCRIPT = "codemix"


class BodhanMT(BaseMT):
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
        self.base_url = (base_url or os.getenv("BODHAN_MT_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or os.getenv("BODHAN_MT_KEY")
        self.model = model or os.getenv("BODHAN_MT_MODEL") or DEFAULT_MODEL
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
                "BODHAN_MT_KEY is not set. Add it to .env or pass api_key=…"
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
        log.info("bodhan mt: base_url=%s model=%s", self.base_url, self.model)

    # ------------------------------------------------------------------

    def translate(
        self,
        requests: List[MTRequest],
        *,
        target_language: str,
        target_script: Optional[str] = None,
        source_language: Optional[str] = None,
    ) -> MTResult:
        self._ensure_loaded()

        segments: List[MTSegment] = []
        for req in requests:
            text = (req.text or "").strip()
            if not text:
                segments.append(MTSegment(
                    id=req.id, source_text=text, text="",
                    source_language=req.source_language or source_language,
                    target_language=target_language,
                    target_script=req.target_script or target_script,
                ))
                continue

            script = (req.target_script or target_script
                      or DEFAULT_TARGET_SCRIPT)
            translated = self._post_one(
                text,
                target_language=target_language,
                target_script=script,
            )
            segments.append(MTSegment(
                id=req.id,
                source_text=text,
                text=sanitize_mt_output(translated, seg_id=req.id),
                source_language=req.source_language or source_language,
                target_language=target_language,
                target_script=script,
            ))

        return MTResult(
            model_name=self.model,
            source_language=source_language,
            target_language=target_language,
            target_script=target_script or DEFAULT_TARGET_SCRIPT,
            segments=segments,
            extra={"endpoint": self.base_url},
        )

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------

    def _post_one(
        self,
        text: str,
        *,
        target_language: str,
        target_script: Optional[str],
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "target_language_code": target_language,
        }
        if target_script:
            payload["target_script"] = target_script

        last_err_sig: Optional[tuple] = None
        for attempt in range(self.max_retries):
            self._pace()
            resp = self._session.post(url, json=payload, timeout=self.timeout_s)
            self._last_post_ts = time.monotonic()

            pairs = [(k, v) for k, v in resp.headers.items()
                     if k.lower().startswith("x-ratelimit")]
            if pairs:
                log.debug("bodhan mt rate-limit headers: %s",
                          ", ".join(f"{k}={v}" for k, v in pairs))

            if resp.status_code == 429:
                wait = self._backoff(attempt, resp.headers.get("retry-after"))
                log.warning("bodhan mt: 429 on attempt %d, sleeping %.1fs",
                            attempt + 1, wait)
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                sig = (resp.status_code, resp.text[:200])
                if last_err_sig == sig and attempt >= 1:
                    raise RuntimeError(
                        f"Bodhan MT failing persistently (HTTP {resp.status_code}, "
                        f"same body across {attempt+1} attempts): {resp.text[:400]}"
                    )
                last_err_sig = sig
                wait = self._backoff(attempt, None)
                log.warning("bodhan mt: HTTP %d on attempt %d, backoff %.1fs",
                            resp.status_code, attempt + 1, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Bodhan MT failed: HTTP {resp.status_code} — {resp.text[:400]}"
                )
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(
                    f"Bodhan MT response missing choices[0].message.content: {data}"
                ) from e
        raise RuntimeError(
            f"Bodhan MT failed after {self.max_retries} retries at {url}"
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
