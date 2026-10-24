"""Forced-alignment adapter registry.

  - mms_fa    : torchaudio MMS forced aligner (multilingual, char-level CTC)
  - whisperx  : WhisperX's wav2vec2 CTC aligner (same code path WhisperX
                uses internally, exposed here so a non-WhisperX ASR can reuse
                it after the fact)
"""
from __future__ import annotations

from .._registry import make_registry
from .base import AlignRequest, AlignResult, AlignedSegment, BaseAligner

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "mms_fa": (".mms_fa_adapter", "MMSForcedAligner"),
    "whisperx": (".whisperx_adapter", "WhisperXAligner"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="aligner",
    base=BaseAligner,
    import_noun="Aligner",
    import_detail="is not installed in this environment.",
)

available_models = _registry.available
get_aligner = _registry.get

__all__ = [
    "AlignRequest",
    "AlignResult",
    "AlignedSegment",
    "BaseAligner",
    "available_models",
    "get_aligner",
]
