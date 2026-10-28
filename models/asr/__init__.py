"""ASR adapter registry — Bodhan Indic-Transcribe only.

  - bodhan : Bodhan Indic-Transcribe (Core / Flex) via REST

`whisperx`, `indicconformer`, `indictranscribe_core` and
`indictranscribe_flex` were removed on 2026-10-28. In git at
b823223.
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

    `language` is accepted and ignored. bodhan leaves it None and
    delegates language detection downstream, so with only bodhan registered
    the filter can never drop anything. The parameter stays because `cli.py`
    and the UI both pass it, and because the filtering comes back the moment
    a second, language-limited adapter does.
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
