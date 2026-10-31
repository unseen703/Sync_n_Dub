"""Shared types and abstract base for transliteration adapters.

A script change, not a translation: `"gradient"` -> `"ग्रेडिएंट"` is the same
word in different glyphs. The code-mix stage uses it to re-script retained
English spans and to normalise mixed-script output to one script.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranslitRequest:
    id: str
    text: str
    source_language: Optional[str] = None     # ISO-639 code of the input word
    source_script: Optional[str] = None       # "roman" | "native" | None
    target_script: str = "native"             # "native" | "roman"


@dataclass
class TranslitSegment:
    id: str
    source_text: str
    text: str                                 # the re-scripted output
    source_language: Optional[str] = None
    source_script: Optional[str] = None
    target_script: str = "native"
    extra: dict = field(default_factory=dict)


@dataclass
class TranslitResult:
    model_name: str
    target_script: str
    segments: List[TranslitSegment]
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "model_name": self.model_name,
            "target_script": self.target_script,
            "segments": [
                {
                    "id": s.id,
                    "source_text": s.source_text,
                    "text": s.text,
                    "source_language": s.source_language,
                    "source_script": s.source_script,
                    "target_script": s.target_script,
                    "extra": s.extra,
                }
                for s in self.segments
            ],
            "extra": self.extra,
        }


class BaseTransliterator(ABC):
    """Every transliteration adapter subclasses this."""

    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from backend.diarize.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def transliterate(
        self,
        requests: List[TranslitRequest],
        *,
        target_language: str,
        target_script: str = "native",
    ) -> TranslitResult:
        """Re-script each request, preserving order."""
