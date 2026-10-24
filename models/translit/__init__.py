"""Transliteration adapter registry.

  - bodhan    : Bodhan Indic-Transliterate via REST (POST /v1/chat/completions)
  - indicxlit : AI4Bharat IndicXlit (en ↔ 21 Indic langs; local transformers)

Both do script-only conversion (same language in, same language out); the
code-mix stage calls them to render retained English spans in the target
script, plus to normalise mixed-script segments to a single script.
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseTransliterator, TranslitRequest, TranslitResult, TranslitSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanTransliterate"),
    "indicxlit": (".indicxlit_adapter", "IndicXlitTransliterate"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="transliterator",
    base=BaseTransliterator,
    import_noun="Transliterator",
)

available_models = _registry.available
get_transliterator = _registry.get

__all__ = [
    "BaseTransliterator",
    "TranslitRequest",
    "TranslitResult",
    "TranslitSegment",
    "available_models",
    "get_transliterator",
]
