"""Shared types + abstract base for speech-emotion adapters.

An emotion adapter reads one audio clip (or a list of them) and returns a
label plus a confidence. It never decides what to *do* with that label --
mapping an emotion onto a TTS delivery style, and deciding whether the
model is confident enough to act on, both live in `backend.emotion` so the
policy stays in one place and the adapters stay swappable.

Two things worth knowing before writing another adapter.

**Label vocabularies differ and are not trustworthy from the docs.**
emotion2vec's own pages disagree with each other -- the HF card for
`emotion2vec_plus_large` lists 5 classes while the GitHub README claims 9.
So `EmotionResult.labels` carries whatever the model actually reported, and
`backend.emotion` maps by *normalised name* rather than by index. Never key
off a label index.

**Emotion is read from the SOURCE audio, not the dub.** The pipeline detects
how the original speaker delivered a line and re-applies that to the
synthesized target-language line. That matters for model choice: the adapter
only ever sees the source language, so a model with no Indic coverage is
still fine for dubbing an English lecture into Hindi.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EmotionResult:
    """One clip's prediction.

    `label` is the model's own string, lowercased and stripped of any
    language prefix (emotion2vec returns e.g. "生气/angry"); `scores` keeps
    the full distribution so a caller can apply its own threshold or
    inspect the runner-up. `confidence` is the winning score.
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

        An adapter must return a result for every id it was given -- set
        `error` and leave `confidence` at 0.0 rather than dropping the
        entry, so the caller can tell "model said nothing" apart from
        "model was never asked".
        """
        raise NotImplementedError
