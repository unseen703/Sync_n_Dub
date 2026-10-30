"""Transliteration adapter registry — Bodhan Indic-Transliterate only.

  - bodhan : Bodhan Indic-Transliterate via REST (POST /v1/chat/completions)

Script-only conversion, same language in and out. The code-mix stage uses it
to render retained English spans in the target script and to normalise
mixed-script segments to one script.
"""
from __future__ import annotations

from backend.common.registry import make_registry
from .base import BaseTransliterator, TranslitRequest, TranslitResult, TranslitSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanTransliterate"),
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
