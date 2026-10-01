"""Shared types + abstract base for ASR adapters.

An ASR adapter takes a mono 16 kHz audio clip (optionally restricted to a
start/end window) and returns `ASRResult` — text, segments, and word-level
timestamps when the adapter can produce them. Adapters that cannot do word
timings leave `words` empty; the forced-alignment stage (`stages/align`)
produces them later from the final text.

Bodhan Indic-Transcribe deliberately omits word timestamps (see plan §5
"What Bodhan does not provide"). We model that as the empty case rather
than inventing fake timings.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union


@dataclass
class Word:
    text: str
    start: float          # seconds
    end: float            # seconds
    score: Optional[float] = None


@dataclass
class ASRSegment:
    """A contiguous chunk of transcript — typically one diarized turn."""
    id: str
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[Word] = field(default_factory=list)
    language: Optional[str] = None     # ISO-639 code, when the ASR returns one


@dataclass
class ASRResult:
    audio_path: Path
    model_name: str
    language: Optional[str]
    segments: List[ASRSegment]
    extra: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text).strip()

    def to_jsonable(self) -> dict:
        return {
            "audio_path": str(self.audio_path),
            "model_name": self.model_name,
            "language": self.language,
            "text": self.text,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker,
                    "language": s.language,
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end, "score": w.score}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
            "extra": self.extra,
        }


def transcript_from_asr_json(src: Union[str, Path, dict]) -> str:
    """Build the end-to-end transcript from an ASR JSON payload.

    `src` may be a path to a `*.asr.json` file, the already-decoded dict, or
    a JSON string. Segments are joined in file order with single spaces; each
    segment's text is stripped first, and empty segments are skipped.
    """
    if isinstance(src, (str, Path)):
        p = Path(src)
        data: Any = json.loads(p.read_text()) if p.exists() else json.loads(str(src))
    else:
        data = src
    segments = data.get("segments", []) if isinstance(data, dict) else []
    return " ".join(
        s["text"].strip() for s in segments if s.get("text") and s["text"].strip()
    ).strip()


class BaseASR(ABC):
    """Every ASR adapter subclasses this."""

    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from models.diarization.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        segments: Optional[List[tuple[str, float, float, Optional[str]]]] = None,
    ) -> ASRResult:
        """Transcribe `audio_path`.

        Args:
          language: ISO-639 code or `None` to let the model detect.
          segments: optional pre-sliced windows `(id, start_s, end_s, speaker?)`.
            When supplied, the adapter transcribes each window separately and
            returns one `ASRSegment` per input — the plan's "speaker-aware
            timed dialogue" is built this way. Without segments, the adapter
            runs full-file ASR and returns whatever segmentation it picks.
        """
