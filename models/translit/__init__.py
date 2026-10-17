"""Transliteration adapter registry.

  - bodhan    : Bodhan Indic-Transliterate via REST (POST /v1/chat/completions)
  - indicxlit : AI4Bharat IndicXlit (en ↔ 21 Indic langs; local transformers)

Both do script-only conversion (same language in, same language out); the
code-mix stage calls them to render retained English spans in the target
script, plus to normalise mixed-script segments to a single script.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseTransliterator, TranslitRequest, TranslitResult, TranslitSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanTransliterate"),
    "indicxlit": (".indicxlit_adapter", "IndicXlitTransliterate"),
}


def _load(key: str) -> Type[BaseTransliterator]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseTransliterator]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_transliterator(name: str, *, device: Optional[str] = None, **kwargs) -> BaseTransliterator:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown transliterator {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Transliterator {key!r} is not installed. Original error: {e}"
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
    "BaseTransliterator",
    "TranslitRequest",
    "TranslitResult",
    "TranslitSegment",
    "available_models",
    "get_transliterator",
]
