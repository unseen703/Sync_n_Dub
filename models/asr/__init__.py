"""ASR adapter registry.

  - bodhan               : Bodhan Indic-Transcribe (Core / Flex) via REST
  - whisperx             : WhisperX (whisper + CTC alignment)
  - indicconformer       : AI4Bharat IndicConformer (Conformer-CTC)
  - indictranscribe_core : bodhan-ai/indic-transcribe-core (Canary, local HF)
  - indictranscribe_flex : bodhan-ai/indic-transcribe-flex (Canary, local HF)
"""
from __future__ import annotations

from typing import Optional

from .._registry import make_registry
from .base import ASRResult, ASRSegment, BaseASR, Word, transcript_from_asr_json

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanASR"),
    "whisperx": (".whisperx_adapter", "WhisperXASR"),
    "indicconformer": (".indicconformer_adapter", "IndicConformerASR"),
    "indictranscribe_core": (".indic_transcribe_adapter", "IndicTranscribeCoreASR"),
    "indictranscribe_flex": (".indic_transcribe_adapter", "IndicTranscribeFlexASR"),
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
_load = _registry.load


def available_models(language: Optional[str] = None) -> list[str]:
    """Registered ASR adapter names, optionally filtered by source language.

    The one stage whose lister is not the shared `sorted(_REGISTRY_SPEC)`.

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


__all__ = [
    "ASRResult",
    "ASRSegment",
    "BaseASR",
    "Word",
    "available_models",
    "get_asr",
    "transcript_from_asr_json",
]
