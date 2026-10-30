"""Shared types and abstract base for forced-alignment adapters.

An aligner takes audio plus the text that was spoken and returns per-word
(start, end) boundaries; it does not recognise speech, it refines boundaries
for text an ASR already produced. This is what turns a text-only transcript
into the word-timed segments the duration and lip-sync stages need.

Takes one `AlignRequest` per segment and returns `AlignedSegment`s in the same
order, with an empty `words` list where the aligner declined (empty text,
unsupported language, too short).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from backend.asr.base import Word


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
        """Return word-level timings for each request.

        The result is parallel to `requests`; a declined segment gets an empty
        `words` list rather than being dropped.
        """
