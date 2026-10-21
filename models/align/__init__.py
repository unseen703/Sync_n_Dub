"""Forced-alignment adapter registry.

  - mms_fa    : torchaudio MMS forced aligner (multilingual, char-level CTC)
  - whisperx  : WhisperX's wav2vec2 CTC aligner (same code path WhisperX
                uses internally, exposed here so a non-WhisperX ASR can reuse
                it after the fact)
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import AlignRequest, AlignResult, AlignedSegment, BaseAligner

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "mms_fa": (".mms_fa_adapter", "MMSForcedAligner"),
    "whisperx": (".whisperx_adapter", "WhisperXAligner"),
}


def _load(key: str) -> Type[BaseAligner]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseAligner]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_aligner(name: str, *, device: Optional[str] = None, **kwargs) -> BaseAligner:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown aligner {name!r}. Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Aligner {key!r} is not installed in this environment. "
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
    "AlignRequest",
    "AlignResult",
    "AlignedSegment",
    "BaseAligner",
    "available_models",
    "get_aligner",
]
