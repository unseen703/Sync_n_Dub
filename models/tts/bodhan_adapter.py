"""Bodhan Indic-Speak adapter.

OpenAI-audio-shaped endpoint — POST /v1/audio/speech. Payload:
  {"model": "indic-speak", "input": "<text>", "voice": "<voice id>",
   "instructions": "{\"lang\": \"<iso>\"}", "speed": 1.0}

Returns the WAV body directly (binary); the response is NOT JSON. We
save it under `out_dir/<segment_id>.wav` and let the mixer pick up the
file.

Speaking rate
-------------
`supports_rate = True`: the engine re-speaks at a different pace rather than
the caller stretching the waveform afterwards, which is the whole distinction
`models/tts/base.py` draws between method 1 and method 2.

**The field name is not in the published docs.** console.bodhan.ai/api-docs
lists only `model`, `input`, `voice` and `instructions`, so `speed` here is
taken from the OpenAI `/v1/audio/speech` shape this endpoint otherwise
follows exactly. That is an inference, and this file does not pretend
otherwise — but it is a *checkable* one rather than a silent one: the API
answers an unknown or out-of-range field with 422, and `_post_one` turns a
422 on a rated request into an error naming `BODHAN_TTS_RATE_FIELD`, which
overrides the name without a code change. Set it to `rate`, or to
`instructions.speed` to nest it inside the instructions blob instead.

The rate is sent ONLY when it differs from 1.0, so a run that never asks for
one is byte-identical to the payload this adapter has always sent and cannot
be broken by the inference above.

Env (loaded from `.env`):
  BODHAN_TTS_URL        (default: https://api.bodhan.ai/v1)
  BODHAN_TTS_KEY        (required)
  BODHAN_TTS_MODEL      (default: indic-speak)
  BODHAN_TTS_RATE_FIELD (default: speed; or `rate`, or `instructions.<name>`)
  BODHAN_TTS_RATE_RANGE (default: 0.25,4.0 — OpenAI's bounds)
"""
from __future__ import annotations

import json as _json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from .._bodhan import BodhanClientMixin
from .base import BaseTTS, TTSRequest, TTSResult, TTSSegment

log = logging.getLogger(__name__)



#: The 14 delivery styles indic-speak accepts, verbatim from the hosted API
#: docs (console.bodhan.ai/api-docs). Case and spacing are significant and
#: an unknown value is a 422, not a silent fallback -- so the CLI pins
#: `--tts-style` to this list and a typo fails at argparse instead of
#: costing an API round trip. Omit style entirely for a neutral reading.
#:
#: The six Ekman emotions here trace to AI4Bharat's Rasa corpus
#: (INTERSPEECH 2024); that paper reports `fear` and `surprise` as the
#: hardest to synthesize, so weak differentiation on those two is expected
#: rather than evidence the knob is broken.
INDIC_SPEAK_STYLES = (
    "AIR style news",
    "Customer Care",
    "TV style news",
    "advertisements",
    "anger",
    "children's stories",
    "disgust",
    "educational lecture",
    "fear",
    "happy",
    "news",
    "sad",
    "single person narration audiobook",
    "surprise",
)
DEFAULT_MODEL = "indic-speak"


#: Where the speaking rate goes in the payload, and the bounds it is clamped
#: to. Both are env-overridable because neither is documented; see the module
#: docstring. `instructions.<name>` nests it inside the instructions JSON.
RATE_FIELD = os.getenv("BODHAN_TTS_RATE_FIELD") or "speed"


def _rate_range() -> tuple:
    raw = os.getenv("BODHAN_TTS_RATE_RANGE") or "0.25,4.0"
    lo, hi = (float(x) for x in raw.split(","))
    return lo, hi


class BodhanTTS(BodhanClientMixin, BaseTTS):
    name = "bodhan"
    #: The engine takes a speaking rate, so `speed_control` is real here --
    #: it re-synthesises at the new pace instead of resampling a finished
    #: clip. Edge was the only other adapter that could do this and it is
    #: gone, so without this flag `--tts-speed-control` would no-op on every
    #: engine the install has.
    supports_rate = True

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 120.0,
        max_retries: int = 10,
        min_interval_s: float = 1.2,
        max_backoff_s: float = 60.0,
    ):
        super().__init__(device=device)
        self._init_bodhan(
            env_prefix="TTS", default_model=DEFAULT_MODEL,
            base_url=base_url, api_key=api_key, model=model,
            timeout_s=timeout_s, max_retries=max_retries,
            min_interval_s=min_interval_s, max_backoff_s=max_backoff_s,
        )

    def _ensure_loaded(self) -> None:
        # `Accept: audio/wav` — this is the one Bodhan service that answers
        # with bytes rather than JSON.
        if not self._ensure_bodhan_session(
            missing_key_message=(
                "BODHAN_TTS_KEY is not set. Add it to .env or pass api_key=…"
            ),
            accept="audio/wav",
        ):
            return
        log.info("bodhan tts: base_url=%s model=%s", self.base_url, self.model)

    # ------------------------------------------------------------------

    def synthesize(
        self,
        requests: List[TTSRequest],
        *,
        out_dir: Path,
        language: str,
        sample_rate: int = 24000,
    ) -> TTSResult:
        self._ensure_loaded()
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        segments: List[TTSSegment] = []
        for req in requests:
            text = (req.text or "").strip()
            dest = out_dir / f"{req.id}.wav"
            if not text:
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=sample_rate, voice=req.voice,
                    speaker=req.speaker, error="empty text",
                ))
                continue
            try:
                audio_bytes = self._post_one(
                    text, voice=req.voice, language=req.language or language,
                    style=req.style,
                    rate=float(getattr(req, "rate", 1.0) or 1.0),
                )
                dest.write_bytes(audio_bytes)
                dur, sr = _wav_duration(dest)
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=dur,
                    sample_rate=sr, voice=req.voice, speaker=req.speaker,
                    # Evidence of the pace actually asked of the engine, so
                    # a closed-loop run can be audited after the fact.
                    extra={"rate_factor": float(getattr(req, "rate", 1.0) or 1.0)},
                ))
            except Exception as e:
                log.warning("bodhan tts: synth failed for %s: %s", req.id, e)
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=sample_rate, voice=req.voice,
                    speaker=req.speaker, error=str(e),
                ))
        return TTSResult(
            model_name=self.model, out_dir=out_dir, language=language,
            segments=segments, extra={"endpoint": self.base_url},
        )

    # ------------------------------------------------------------------

    def _post_one(
        self, text: str, *,
        voice: Optional[str], language: str, style: Optional[str],
        rate: float = 1.0,
    ) -> bytes:
        url = f"{self.base_url}/audio/speech"
        instructions = {"lang": language}
        if style:
            instructions["style"] = style
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice or "Amit",
        }
        # Only ever sent when a rate was actually asked for: at 1.0 the
        # payload stays exactly what it has always been, so the undocumented
        # field name cannot break a run that does not use it.
        rated = abs(rate - 1.0) > 1e-9
        if rated:
            lo, hi = _rate_range()
            clamped = max(lo, min(hi, rate))
            if clamped != rate:
                log.warning("bodhan tts: rate %.3f outside [%s, %s]; clamped "
                            "to %.3f", rate, lo, hi, clamped)
            field = RATE_FIELD
            if field.startswith("instructions."):
                instructions[field.split(".", 1)[1]] = clamped
            else:
                payload[field] = clamped
        payload["instructions"] = _json.dumps(instructions)
        last_err_sig: Optional[tuple] = None
        for attempt in range(self.max_retries):
            self._pace()
            resp = self._session.post(url, json=payload, timeout=self.timeout_s)
            self._last_post_ts = time.monotonic()

            # Surface the rate-limit headers whenever the server sends them —
            # Bodhan only ships them on MT today, but logging on sight means
            # future ASR/TTS rollouts show up without an adapter change.
            self._log_ratelimit_headers(resp)

            if resp.status_code == 429:
                wait = self._backoff(attempt, resp.headers.get("retry-after"))
                log.warning("bodhan tts: 429 on attempt %d, sleeping %.1fs",
                            attempt + 1, wait)
                time.sleep(wait); continue
            if 500 <= resp.status_code < 600:
                # Fail fast when the service is clearly dead: identical status
                # + body across two attempts means retrying is just burning
                # wall-clock (observed with Bodhan TTS "internal_server_error"
                # outages, where every retry returns the same 500).
                sig = (resp.status_code, resp.text[:200])
                if last_err_sig == sig and attempt >= 1:
                    raise RuntimeError(
                        f"Bodhan TTS failing persistently (HTTP {resp.status_code}, "
                        f"same body across {attempt+1} attempts): {resp.text[:400]}"
                    )
                last_err_sig = sig
                wait = self._backoff(attempt, None)
                log.warning("bodhan tts: HTTP %d on attempt %d, backoff %.1fs",
                            resp.status_code, attempt + 1, wait)
                time.sleep(wait); continue
            if resp.status_code >= 400:
                # A 422 on a request that carries a rate is the one failure
                # this adapter can diagnose outright: the rate field name is
                # not in the published docs (see the module docstring), so
                # it is the first thing to suspect and the message says how
                # to change it rather than leaving the reader to find this
                # file. Unrated requests fall through to the generic error.
                if resp.status_code == 422 and rated:
                    raise RuntimeError(
                        f"Bodhan TTS rejected the speaking rate (HTTP 422). "
                        f"The rate was sent as "
                        f"{'instructions.' if RATE_FIELD.startswith('instructions.') else ''}"
                        f"{RATE_FIELD.split('.')[-1]}={clamped}, a field the "
                        f"published API docs do not list — set "
                        f"BODHAN_TTS_RATE_FIELD to the right name (or run "
                        f"without --tts-speed-control). Server said: "
                        f"{resp.text[:300]}"
                    )
                raise RuntimeError(
                    f"Bodhan TTS failed: HTTP {resp.status_code} — {resp.text[:400]}"
                )
            return resp.content
        raise RuntimeError(
            f"Bodhan TTS failed after {self.max_retries} retries at {url}"
        )

    def _log_ratelimit_headers(self, resp) -> None:
        super()._log_ratelimit_headers(resp, stage="tts")


def _wav_duration(path: Path) -> tuple[float, int]:
    import soundfile as sf
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate)
