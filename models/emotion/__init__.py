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

import importlib
import inspect
from typing import Optional, Type

from .base import BaseEmotion, EmotionBatch, EmotionResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "emotion2vec": (".emotion2vec_adapter", "Emotion2VecEmotion"),
}


def _load(key: str) -> Type[BaseEmotion]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseEmotion]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_emotion(
    name: str, *, device: Optional[str] = None, **kwargs
) -> BaseEmotion:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown emotion model {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Emotion model {key!r} is not installed in this environment "
            f"(`pip install funasr`). Original error: {e}"
        ) from e
    accepted = _accepted_kwargs(cls)
    if accepted:
        unknown = set(kwargs) - accepted
        if unknown:
            import warnings
            warnings.warn(
                f"{cls.__name__} does not accept kwargs {sorted(unknown)!r}; "
                f"dropping. Accepted: {sorted(accepted - {'device'})!r}."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(device=device, **kwargs)


__all__ = [
    "BaseEmotion",
    "EmotionBatch",
    "EmotionResult",
    "available_models",
    "get_emotion",
]
