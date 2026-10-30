"""Bodhan Indic-Translate adapter.

OpenAI-chat-shaped endpoint — POST /v1/chat/completions with
`model=indic-translate`, `target_language_code` and `target_script`. One call
per segment, since the API is single-utterance.

Env (loaded from `.env`):
  BODHAN_MT_URL    (default: https://api.bodhan.ai/v1)
  BODHAN_MT_KEY    (required)
  BODHAN_MT_MODEL  (default: indic-translate)

`target_script` is "native", "roman", "codemix", or None for the server
default. `codemix` is the default here: `native` re-scripts a technical term
inconsistently across cues, and TTS does not reliably pronounce a bare Latin
remnant. Pass `target_script="native"` for a fully re-scripted translation.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from backend.sanitize import sanitize_mt_output

from .._bodhan import BodhanClientMixin
from .base import BaseMT, MTRequest, MTResult, MTSegment

log = logging.getLogger(__name__)


DEFAULT_MODEL = "indic-translate"
# None means "let the server decide". A caller that wants a specific script
# (UI toggle, CLI flag) passes it through; nothing is substituted here.
DEFAULT_TARGET_SCRIPT: Optional[str] = None


class BodhanMT(BodhanClientMixin, BaseMT):
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
        self._init_bodhan(
            env_prefix="MT", default_model=DEFAULT_MODEL,
            base_url=base_url, api_key=api_key, model=model,
            timeout_s=timeout_s, max_retries=max_retries,
            min_interval_s=min_interval_s, max_backoff_s=max_backoff_s,
        )

    def _ensure_loaded(self) -> None:
        if not self._ensure_bodhan_session(
            missing_key_message=(
                "BODHAN_MT_KEY is not set. Add it to .env or pass api_key=…"
            ),
        ):
            return
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

            script = req.target_script or target_script or DEFAULT_TARGET_SCRIPT
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
            target_script=target_script or DEFAULT_TARGET_SCRIPT,  # may be None
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

