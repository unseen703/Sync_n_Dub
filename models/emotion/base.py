"""Shared types and abstract base for speech-emotion adapters.

An adapter reads audio clips and returns a label plus a confidence. What to do
with that label lives in `backend.emotion`, so the policy stays in one place.

Two rules for writing another adapter. Label vocabularies differ and the
published docs are unreliable, so `EmotionBatch.labels` carries whatever the
model actually reported and callers map by normalised NAME, never by index.
And emotion is read from the SOURCE audio, not the dub — an adapter only ever
sees the source language, so no Indic coverage is needed to dub into Hindi.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EmotionResult:
    """One clip's prediction.

    `label` is the model's own string, lowercased and stripped of any language
    prefix (emotion2vec returns e.g. "生气/angry"). `scores` keeps the full
    distribution; `confidence` is the winning score.
    """
    clip_id: str
    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EmotionBatch:
    model_name: str
    results: List[EmotionResult]
    labels: List[str] = field(default_factory=list)   # vocabulary as reported
    extra: dict = field(default_factory=dict)

    def by_id(self) -> Dict[str, EmotionResult]:
        return {r.clip_id: r for r in self.results}


class BaseEmotion(ABC):
    """Adapters implement `score`; everything else is the caller's policy."""

    name: str = "base"
    #: Sample rate the model expects. `backend.emotion` resamples to this.
    sample_rate: int = 16_000
    #: Free-text note for the manifest, e.g. which languages were validated.
    coverage: str = "unknown"

    def __init__(self, device: Optional[str] = None, **kwargs):
        self.device = device

    @abstractmethod
    def score(self, clips: Dict[str, Path]) -> EmotionBatch:
        """Score `{clip_id: wav_path}` and return one result per clip.

        Every id must come back — set `error` and leave `confidence` at 0.0
        rather than dropping the entry, so the caller can tell "model said
        nothing" from "model was never asked".
        """
        raise NotImplementedError
