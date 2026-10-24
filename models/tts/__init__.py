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

from typing import Optional

from .._registry import make_registry
from .base import BaseTTS, TTSRequest, TTSResult, TTSSegment

#: "Read it plainly" — the one delivery value that is NOT an engine style.
#:
#: Bodhan has no `neutral` in `INDIC_SPEAK_STYLES`; a neutral reading is what
#: you get by sending no style at all (`bodhan_adapter._post_one`: `if style:
#: instructions["style"] = style`). There was no way to ASK for that, though:
#: the new-job form's "auto" resolves to stage 9b1 classifying a register, and
#: the per-speaker picker offered the 14 engine styles and nothing else. So a
#: dub could be given a register and never taken off one.
#:
#: This is the sentinel that means it. It is carried through the pipeline and
#: recorded in the manifest like any other style — the operator's choice is
#: worth keeping — and resolved to None at the moment of synthesis by
#: `resolve_style` below. Defined here, in the registry both `backend/` and
#: `ui/` already import, so there is one spelling of it.
NEUTRAL_STYLE = "neutral"

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanTTS"),
    "indicf5": (".indicf5_adapter", "IndicF5TTS"),
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


def resolve_style(segment_style: Optional[str],
                  document_style: Optional[str]) -> Optional[str]:
    """The style one segment is actually synthesised with.

    Two rules, in this order:

      1. A per-segment `tts_style` beats the document-level register. That is
         long-standing behaviour — `backend/edits.py` writes the per-segment
         key and stage 9b1 writes the document one — and is preserved exactly.
      2. `NEUTRAL_STYLE` resolves to None, i.e. no style in the payload.

    Rule 2 cannot be expressed as "set it to None", because None is what a
    segment with no override looks like, and `seg or doc` would then fall back
    to the very register the operator just asked to be rid of.
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
