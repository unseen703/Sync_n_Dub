"""Shared types + abstract base for forced-alignment adapters.

A forced aligner takes an audio clip plus the text that was spoken, and
returns per-word (start, end) time boundaries. It does NOT recognise speech
from scratch; it refines boundaries using text that an ASR already produced.

The pipeline uses it to turn Bodhan's text-only transcript into the
word-timed segments the duration-adjustment and lip-sync stages need
(plan §3 `words`, §7).

Input shape: a list of `AlignRequest`, one per diarized/ASR segment:
    (id, start_s, end_s, text, language?)

Output: a list of `AlignedSegment` in the same order, carrying either the
word boundaries the aligner produced, or an empty `words` list when the
aligner declined that segment (empty text, unsupported language, too short).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from models.asr.base import Word


@dataclass
class AlignRequest:
    id: str
    start: float
    end: float
    text: str
    language: Optional[str] = None
    speaker: Optional[str] = None


@dataclass
class AlignedSegment:
    id: str
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None
    language: Optional[str] = None


@dataclass
class AlignResult:
    audio_path: Path
    model_name: str
    language: Optional[str]
    segments: List[AlignedSegment]
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "audio_path": str(self.audio_path),
            "model_name": self.model_name,
            "language": self.language,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "speaker": s.speaker,
                    "language": s.language,
                    "words": [
                        {"text": w.text, "start": w.start, "end": w.end,
                         "score": w.score}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
            "extra": self.extra,
        }


class BaseAligner(ABC):
    """Every forced-alignment adapter subclasses this."""

    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from models.diarization.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def align(
        self,
        audio_path: Path,
        requests: List[AlignRequest],
        *,
        language: Optional[str] = None,
    ) -> AlignResult:
        """Return word-level timings for each request. The returned list is
        parallel to `requests`; a segment the aligner declined gets an empty
        `words` list rather than being dropped.
        """
