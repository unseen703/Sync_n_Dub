"""Shared types + abstract base for TTS adapters.

A TTS adapter takes a list of `TTSRequest` (segment id, text, voice,
optional reference audio for cloning) and writes one WAV per segment
under `out_dir`. The returned `TTSResult.segments[i].audio_path` is the
file the downstream mixer reads.

The adapter does NOT time-stretch — duration adjustment is the caller's
job (`backend.duration.wsola_stretch_cap`) so the stretch logic stays
engine-agnostic and the cap is enforced centrally.

`TTSRequest.rate` is the one exception, and it is NOT a stretch: it is a
SYNTHESIS parameter handed to the engine so it *speaks* faster or slower,
re-deciding phoneme durations and prosody from scratch. Post-hoc stretching
instead resamples an already-spoken waveform. The two produce audibly
different artefacts at the same ratio, and telling them apart is the entire
point of comparing method 1 (stretch the clip) against method 2 (ask the
engine for a shorter clip) — so an adapter must implement `rate` by passing it
to the engine, never by stretching its own output. An adapter that cannot do
that leaves `supports_rate = False` and the caller skips the whole mechanism.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TTSRequest:
    id: str
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    reference_audio: Optional[Path] = None    # for zero-shot cloning engines
    reference_text: Optional[str] = None      # IndicF5 needs it
    speaker: Optional[str] = None             # for logging + filename
    style: Optional[str] = None               # engine-specific hint
    rate: float = 1.0                         # 1.0 = engine default; 1.1 = 10% faster
    extra: dict = field(default_factory=dict)


@dataclass
class TTSSegment:
    id: str
    audio_path: Path
    duration_s: float
    sample_rate: int
    voice: Optional[str] = None
    speaker: Optional[str] = None
    error: Optional[str] = None               # set when synthesis failed
    extra: dict = field(default_factory=dict)


@dataclass
class TTSResult:
    model_name: str
    out_dir: Path
    language: str
    segments: List[TTSSegment]
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "model_name": self.model_name,
            "out_dir": str(self.out_dir),
            "language": self.language,
            "segments": [
                {
                    "id": s.id,
                    "audio_path": str(s.audio_path),
                    "duration_s": s.duration_s,
                    "sample_rate": s.sample_rate,
                    "voice": s.voice,
                    "speaker": s.speaker,
                    "error": s.error,
                    "extra": s.extra,
                }
                for s in self.segments
            ],
            "extra": self.extra,
        }


class BaseTTS(ABC):
    """Every TTS adapter subclasses this."""

    name: str = "base"
    # True only when the engine itself accepts a speaking-rate parameter.
    # Adapters that would have to fake it with a time-stretch leave it False;
    # the caller then no-ops its speed control instead of silently measuring a
    # stretch while believing it measured synthesis.
    supports_rate: bool = False

    def __init__(self, device: Optional[str] = None, **kwargs):
        from models.diarization.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def synthesize(
        self,
        requests: List[TTSRequest],
        *,
        out_dir: Path,
        language: str,
        sample_rate: int = 24000,
    ) -> TTSResult:
        """Synthesise each request; return paths in input order."""
