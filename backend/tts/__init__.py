"""Text-to-speech adapter registry — Bodhan Indic-Speak only.

  - bodhan : Bodhan Indic-Speak via REST (POST /v1/audio/speech)
"""
from __future__ import annotations

from typing import Optional

from backend.common.registry import make_registry
from .base import BaseTTS, TTSRequest, TTSResult, TTSSegment

#: "Read it plainly" — the one delivery value that is NOT an engine style.
#: Bodhan has no `neutral`; a neutral reading is what sending no style gives.
#: The sentinel is carried through the pipeline and recorded like any other
#: style, then resolved to None at synthesis by `resolve_style`.
NEUTRAL_STYLE = "neutral"

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanTTS"),
}


_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="TTS model",
    base=BaseTTS,
    import_noun="TTS model",
)

available_models = _registry.available
get_tts = _registry.get

def __getattr__(name):
    if name in {"synthesize_segments", "resolve_style"}:
        import importlib
        return getattr(importlib.import_module(".tts", __name__), name)
    raise AttributeError(name)


def resolve_style(segment_style: Optional[str],
                  document_style: Optional[str]) -> Optional[str]:
    """The style one segment is actually synthesised with.

    A per-segment `tts_style` beats the document-level register, and
    `NEUTRAL_STYLE` resolves to None (no style in the payload). The sentinel is
    needed because None already means "no override", so `seg or doc` would fall
    back to the register the operator asked to be rid of.
    """
    style = document_style if segment_style in (None, "") else segment_style
    if isinstance(style, str) and style.strip().lower() == NEUTRAL_STYLE:
        return None
    return style


__all__ = [
    "BaseTTS",
    "NEUTRAL_STYLE",
    "TTSRequest",
    "TTSResult",
    "TTSSegment",
    "available_models",
    "get_tts",
    "resolve_style",
]
