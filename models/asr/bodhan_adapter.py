"""Bodhan Indic-Transcribe adapter.

OpenAI-compatible endpoint — POST /v1/audio/transcriptions with multipart file
upload. One endpoint, one key, chosen per service as the plan calls out
(§5 "Credentials are per service"). Rate limits come back in
`x-ratelimit-*` headers; the adapter adapts concurrency from them rather than
pinning a worker count.

Env (loaded from `.env`):
  BODHAN_ASR_URL    (default: https://api.bodhan.ai/v1)
  BODHAN_ASR_KEY    (required)
  BODHAN_ASR_MODEL  (default: indic-transcribe-core; Flex via `--asr-flex`)

Render mode (plan §6 — a knob, not a fixed upstream): native | mixed | roman.
Pass through `render_mode` kwarg; the adapter sends it as `response_format`
style parameter named `render_mode` in the multipart form. If a future API
spec names it differently, change this one line.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import ASRResult, ASRSegment, BaseASR, Word

log = logging.getLogger(__name__)


DEFAULT_URL = "https://api.bodhan.ai/v1"
DEFAULT_MODEL = "indic-transcribe-core"


class BodhanASR(BaseASR):
    name = "bodhan"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        render_mode: Optional[str] = None,
        timeout_s: float = 120.0,
        max_retries: int = 10,
        min_interval_s: float = 1.2,
        max_backoff_s: float = 60.0,
    ):
        super().__init__(device=device)
        self.base_url = (base_url or os.getenv("BODHAN_ASR_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or os.getenv("BODHAN_ASR_KEY")
        self.model = model or os.getenv("BODHAN_ASR_MODEL") or DEFAULT_MODEL
        self.render_mode = render_mode   # None → server default
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
                "BODHAN_ASR_KEY is not set. Add it to .env or pass api_key=…"
            )
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("`pip install requests` for the Bodhan adapter.") from e
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        })
        self._loaded = True
        log.info("bodhan asr: base_url=%s model=%s render_mode=%s",
                 self.base_url, self.model, self.render_mode or "default")

    # ------------------------------------------------------------------
    # transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        segments: Optional[List[tuple[str, float, float, Optional[str]]]] = None,
    ) -> ASRResult:
        self._ensure_loaded()
        audio_path = Path(audio_path)

        if segments is None:
            seg = self._post_file(audio_path, language)
            return ASRResult(
                audio_path=audio_path,
                model_name=self.model,
                language=seg.get("language") or language,
                segments=[ASRSegment(
                    id="SEG_0000",
                    start=0.0,
                    end=float(seg.get("duration", 0.0)),
                    text=(seg.get("text") or "").strip(),
                    speaker=None,
                    language=seg.get("language"),
                )],
                extra={"raw": seg},
            )

        import soundfile as sf
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        import tempfile
        import io
        out_segments: List[ASRSegment] = []
        detected_lang: Optional[str] = None
        for sid, start_s, end_s, speaker in segments:
            s0 = max(0, int(round(start_s * sr)))
            s1 = min(len(audio), int(round(end_s * sr)))
            clip = audio[s0:s1]
            if clip.size == 0:
                out_segments.append(ASRSegment(
                    id=sid, start=start_s, end=end_s, text="",
                    speaker=speaker, language=language,
                ))
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip, sr)
                r = self._post_file(Path(tmp.name), language)
            detected_lang = detected_lang or r.get("language")
            out_segments.append(ASRSegment(
                id=sid, start=start_s, end=end_s,
                text=(r.get("text") or "").strip(),
                speaker=speaker,
                language=r.get("language") or language,
            ))

        return ASRResult(
            audio_path=audio_path,
            model_name=self.model,
            language=detected_lang or language,
            segments=out_segments,
            extra={"render_mode": self.render_mode, "endpoint": self.base_url},
        )

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------

    def _post_file(self, path: Path, language: Optional[str]) -> dict:
        """POST one audio file with pacing + exponential backoff on 429/5xx.

        Enforces a `min_interval_s` floor between consecutive requests
        (per-adapter instance) so serial per-turn calls don't burst past
        the server's per-second budget, and backs off exponentially up to
        `max_backoff_s` when it still sees 429, respecting any
        `Retry-After` header the server sends.
        """
        url = f"{self.base_url}/audio/transcriptions"
        last_err_sig: Optional[tuple] = None
        for attempt in range(self.max_retries):
            self._pace()
            with open(path, "rb") as fh:
                files = {"file": (path.name, fh, "audio/wav")}
                data: dict = {"model": self.model, "vad": True}
                if language:
                    data["language"] = language
                if self.render_mode:
                    data["render_mode"] = self.render_mode
                resp = self._session.post(url, files=files, data=data, timeout=self.timeout_s)
                self._last_post_ts = time.monotonic()

            pairs = [(k, v) for k, v in resp.headers.items()
                     if k.lower().startswith("x-ratelimit")]
            if pairs:
                log.info("bodhan asr rate-limit headers: %s",
                         ", ".join(f"{k}={v}" for k, v in pairs))

            if resp.status_code == 429:
                wait = self._backoff(attempt, resp.headers.get("retry-after"))
                log.warning("bodhan asr: 429 on attempt %d, sleeping %.1fs",
                            attempt + 1, wait)
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                sig = (resp.status_code, resp.text[:200])
                if last_err_sig == sig and attempt >= 1:
                    raise RuntimeError(
                        f"Bodhan ASR failing persistently (HTTP {resp.status_code}, "
                        f"same body across {attempt+1} attempts): {resp.text[:400]}"
                    )
                last_err_sig = sig
                wait = self._backoff(attempt, None)
                log.warning("bodhan asr: HTTP %d on attempt %d, backoff %.1fs",
                            resp.status_code, attempt + 1, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Bodhan ASR failed: HTTP {resp.status_code} — {resp.text[:400]}"
                )
            return resp.json()
        raise RuntimeError(
            f"Bodhan ASR failed after {self.max_retries} retries at {url}"
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
