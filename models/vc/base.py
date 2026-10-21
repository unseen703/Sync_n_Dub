"""Shared types + abstract base for voice-conversion adapters.

A VC adapter takes a source WAV (the TTS output) plus a target speaker's
reference WAV, and writes a converted WAV. Content stays the same; the
timbre moves toward the reference.

Length is preserved — the duration cap has already been enforced on the
source, so the converted clip inherits it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class VCRequest:
    id: str
    source_audio: Path           # TTS output
    reference_audio: Path        # target speaker's clip
    speaker: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class VCSegment:
    id: str
    audio_path: Path
    duration_s: float
    sample_rate: int
    speaker: Optional[str] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class VCResult:
    model_name: str
    out_dir: Path
    segments: List[VCSegment]
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "model_name": self.model_name,
            "out_dir": str(self.out_dir),
            "segments": [
                {
                    "id": s.id, "audio_path": str(s.audio_path),
                    "duration_s": s.duration_s, "sample_rate": s.sample_rate,
                    "speaker": s.speaker, "error": s.error, "extra": s.extra,
                }
                for s in self.segments
            ],
            "extra": self.extra,
        }


class BaseVC(ABC):
    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from models.diarization.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def convert(
        self,
        requests: List[VCRequest],
        *,
        out_dir: Path,
        language: str,
    ) -> VCResult:
        """Convert each request; return in input order."""
