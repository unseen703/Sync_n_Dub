"""ASR adapter registry.

  - bodhan        : Bodhan Indic-Transcribe (Core / Flex) via REST
  - whisperx      : WhisperX (whisper + CTC alignment)
  - indicconformer: AI4Bharat IndicConformer (Conformer-CTC)
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
}


def _load(key: str) -> Type[BaseASR]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


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
