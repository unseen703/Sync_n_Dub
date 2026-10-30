"""Shared types and abstract base for TTS adapters.

An adapter takes `TTSRequest`s and writes one WAV per segment under
`out_dir`; `TTSResult.segments[i].audio_path` is what the mixer reads.

An adapter does NOT time-stretch — duration adjustment is the caller's job, so
the stretch logic stays engine-agnostic and the cap is enforced centrally.

`TTSRequest.rate` is the exception and is NOT a stretch: it is a synthesis
parameter the engine uses to re-decide phoneme durations and prosody, which
sounds different from resampling a finished waveform. An adapter must
implement it by passing it through, never by stretching its own output; one
that cannot leaves `supports_rate = False` and the caller skips the mechanism.
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
    # True only when the engine itself accepts a speaking-rate parameter. An
    # adapter that would fake it with a time-stretch leaves it False, and the
    # caller no-ops its speed control rather than measuring the wrong thing.
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
