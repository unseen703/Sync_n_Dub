"""Text-to-speech adapter registry.

  - bodhan   : Bodhan Indic-Speak via REST (POST /v1/audio/speech)
  - indicf5  : AI4Bharat IndicF5 — zero-shot cloning TTS (local transformers)

Edge (Microsoft Edge-Read-Aloud) was removed on 2026-10-17. It spoke 11 of the
23 languages this project offers and had exactly one voice per gender per
language, so it could neither cover the catalog nor tell two same-gender
speakers apart; the voice-assignment stage had a whole second code path for
it that did not match on gender at all. Its table also carried a voice id
that the service does not have (`pa-IN-OjasNeural`, verified absent from all
322 voices `edge_tts.list_voices()` returns), which made every Punjabi run a
silent Hindi substitution. Bodhan's indic-speak covers all 22 scheduled
languages with 45 voices and is now the only stock-voice engine.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseTTS, TTSRequest, TTSResult, TTSSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanTTS"),
    "indicf5": (".indicf5_adapter", "IndicF5TTS"),
}


def _load(key: str) -> Type[BaseTTS]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseTTS]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_tts(name: str, *, device: Optional[str] = None, **kwargs) -> BaseTTS:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown TTS model {name!r}. Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"TTS model {key!r} is not installed. Original error: {e}"
        ) from e
    accepted = _accepted_kwargs(cls)
    if accepted:
        unknown = set(kwargs) - accepted
        if unknown:
            import warnings
            warnings.warn(
                f"{cls.__name__} does not accept kwargs {sorted(unknown)!r}; dropping."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(device=device, **kwargs)


__all__ = [
    "BaseTTS",
    "TTSRequest",
    "TTSResult",
    "TTSSegment",
    "available_models",
    "get_tts",
]
