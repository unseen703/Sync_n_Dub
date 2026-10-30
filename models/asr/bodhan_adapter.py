"""Bodhan Indic-Transcribe ASR adapter.

POST /v1/audio/transcriptions, multipart. Detects the language once from the
longest turn and pins it for the rest, since a 1-5 s turn is too short to
identify reliably.

Env: BODHAN_ASR_{URL,KEY,MODEL}.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .._bodhan import BodhanClientMixin
from .base import ASRResult, ASRSegment, BaseASR, Word

log = logging.getLogger(__name__)


DEFAULT_MODEL = "indic-transcribe-core"


class BodhanASR(BodhanClientMixin, BaseASR):
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
        self.render_mode = render_mode   # None → server default
        self._init_bodhan(
            env_prefix="ASR", default_model=DEFAULT_MODEL,
            base_url=base_url, api_key=api_key, model=model,
            timeout_s=timeout_s, max_retries=max_retries,
            min_interval_s=min_interval_s, max_backoff_s=max_backoff_s,
        )

    def _ensure_loaded(self) -> None:
        # No Content-Type: this posts multipart, so `requests` must set the
        # boundary itself.
        if not self._ensure_bodhan_session(
            missing_key_message=(
                "BODHAN_ASR_KEY is not set. Add it to .env or pass api_key=…"
            ),
            content_type=None,
        ):
            return
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

        # ONE language for the whole file, detected once and then pinned.
        #
        # Each turn is its own POST and with no `language` the server
        # re-detects per turn. A 1-5 s turn is far too little for reliable
        # identification, so an English lecture comes back written in several
        # different Indic scripts and MT then translates each as if it really
        # were that language.
        #
        # The probe turn is the LONGEST one — identification improves with
        # duration and the first turn is often a two-word fragment. It costs no
        # extra call, since that turn is in the list anyway.
        order = list(range(len(segments)))
        if language is None and len(segments) > 1:
            longest = max(order, key=lambda i: segments[i][2] - segments[i][1])
            order = [longest] + [i for i in order if i != longest]
        results: dict = {}

        for _idx in order:
            sid, start_s, end_s, speaker = segments[_idx]
            s0 = max(0, int(round(start_s * sr)))
            s1 = min(len(audio), int(round(end_s * sr)))
            clip = audio[s0:s1]
            if clip.size == 0:
                results[_idx] = ASRSegment(
                    id=sid, start=start_s, end=end_s, text="",
                    speaker=speaker, language=language or detected_lang,
                )
                continue
            # An explicit `language` wins; otherwise the probe turn's answer
            # is pinned for every turn after it.
            use = language or detected_lang
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip, sr)
                r = self._post_file(Path(tmp.name), use)
            if detected_lang is None and r.get("language"):
                detected_lang = r["language"]
                log.info("bodhan asr: language detected as %r from the "
                         "longest turn (%.1fs); pinning it for the "
                         "remaining %d turn(s)",
                         detected_lang, end_s - start_s, len(segments) - 1)
            results[_idx] = ASRSegment(
                id=sid, start=start_s, end=end_s,
                text=(r.get("text") or "").strip(),
                speaker=speaker,
                language=use or r.get("language"),
            )

        # Back into timeline order: the probe turn was processed first.
        out_segments = [results[i] for i in range(len(segments)) if i in results]

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
        """POST one audio file, with pacing and backoff on 429/5xx.

        Holds a `min_interval_s` floor between requests so serial per-turn
        calls do not burst past the server's per-second budget, then backs off
        up to `max_backoff_s`, respecting any `Retry-After` header.
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
            if resp.status_code == 422 and "render_mode" in resp.text:
                # The one 4xx this adapter can diagnose outright: the field
                # name is this module's own guess and the live endpoint
                # rejects it, failing the whole job at stage 7 behind a
                # wrapper message that never names the cause.
                raise RuntimeError(
                    f"Bodhan ASR rejected `render_mode="
                    f"{self.render_mode!r}` (HTTP 422). The hosted "
                    f"indic-transcribe endpoint does not accept that field "
                    f"— drop --render-mode and let the server default "
                    f"stand. Server said: {resp.text[:240]}"
                )
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

