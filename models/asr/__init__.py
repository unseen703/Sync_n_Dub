"""ASR adapter registry.

  - bodhan               : Bodhan Indic-Transcribe (Core / Flex) via REST
  - whisperx             : WhisperX (whisper + CTC alignment)
  - indicconformer       : AI4Bharat IndicConformer (Conformer-CTC)
  - indictranscribe_core : bodhan-ai/indic-transcribe-core (Canary, local HF)
  - indictranscribe_flex : bodhan-ai/indic-transcribe-flex (Canary, local HF)
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import ASRResult, ASRSegment, BaseASR, Word, transcript_from_asr_json

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanASR"),
    "whisperx": (".whisperx_adapter", "WhisperXASR"),
    "indicconformer": (".indicconformer_adapter", "IndicConformerASR"),
    "indictranscribe_core": (".indic_transcribe_adapter", "IndicTranscribeCoreASR"),
    "indictranscribe_flex": (".indic_transcribe_adapter", "IndicTranscribeFlexASR"),
}


def _load(key: str) -> Type[BaseASR]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models(language: Optional[str] = None) -> list[str]:
    """Registered ASR adapter names, optionally filtered by source language.

    When `language` is set, adapters whose `supported_languages` class
    attribute declares a fixed inventory and does not include the code are
    omitted. Adapters that leave `supported_languages = None` (whisperx,
    bodhan) always pass through — they handle any language the caller asks
    for, either because the model is language-agnostic or because they
    delegate language detection downstream.
    """
    names = sorted(_REGISTRY_SPEC)
    if language is None:
        return names
    out = []
    for key in names:
        try:
            cls = _load(key)
        except ImportError:
            # Can't import → can't know its language set → leave it in so the
            # user gets the ImportError (actionable) instead of silent drop.
            out.append(key); continue
        if cls.supports_language(language):
            out.append(key)
    return out


def _accepted_kwargs(cls: Type[BaseASR]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_asr(name: str, *, device: Optional[str] = None, **kwargs) -> BaseASR:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown ASR model {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"ASR model {key!r} is not installed in this environment. "
            f"Original error: {e}"
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
    "ASRResult",
    "ASRSegment",
    "BaseASR",
    "Word",
    "available_models",
    "get_asr",
    "transcript_from_asr_json",
]
