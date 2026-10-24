"""Speech-emotion adapter registry.

  - emotion2vec : iic/emotion2vec_plus_large — utterance-level SER via FunASR

An emotion adapter reads the SOURCE speaker's audio per segment and reports
how that line was delivered. `backend.emotion` turns those labels into
per-segment `tts_style` values, which `backend/tts.py` already consumes
(`seg.get("tts_style") or info.get("style")`).

Adding a model is one adapter plus one line in `_REGISTRY_SPEC` below. The
lazy-import pattern matches `models.faces.detect`: an env missing FunASR can
still `from models.emotion import available_models` without blowing up, and
only pays the import cost when a model is actually requested.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseEmotion, EmotionBatch, EmotionResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "emotion2vec": (".emotion2vec_adapter", "Emotion2VecEmotion"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="emotion model",
    base=BaseEmotion,
    import_noun="Emotion model",
    import_detail="is not installed in this environment (`pip install "
                  "funasr`).",
    verbose_warning=True,
)

available_models = _registry.available
get_emotion = _registry.get

__all__ = [
    "BaseEmotion",
    "EmotionBatch",
    "EmotionResult",
    "available_models",
    "get_emotion",
]
