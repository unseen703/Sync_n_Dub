"""Speech-emotion adapter registry.

  - emotion2vec : iic/emotion2vec_plus_large — utterance-level SER via FunASR

An adapter reads the SOURCE speaker's audio per segment and reports how that
line was delivered; `backend.emotion` turns the labels into `tts_style`
values. Adapters import lazily, so an env without FunASR can still list the
registry.
"""
from __future__ import annotations

from backend.common.registry import make_registry
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
