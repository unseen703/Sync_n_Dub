"""ASR adapter registry — Bodhan Indic-Transcribe only.

  - bodhan : Bodhan Indic-Transcribe (Core / Flex) via REST
"""
from __future__ import annotations

from typing import Optional

from .._registry import make_registry
from .base import ASRResult, ASRSegment, BaseASR, Word, transcript_from_asr_json

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanASR"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="ASR model",
    base=BaseASR,
    import_noun="ASR model",
    import_detail="is not installed in this environment.",
)

get_asr = _registry.get


def available_models(language: Optional[str] = None) -> list[str]:
    """Registered ASR adapter names.

    `language` is accepted and ignored: no registered adapter limits its
    language inventory, so the filter can never drop anything. The parameter
    stays because the CLI and UI pass it, and filtering returns the moment a
    language-limited adapter is added.
    """
    return sorted(_REGISTRY_SPEC)


__all__ = [
    "ASRResult",
    "ASRSegment",
    "BaseASR",
    "Word",
    "available_models",
    "get_asr",
    "transcript_from_asr_json",
]
