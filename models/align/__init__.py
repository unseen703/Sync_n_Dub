"""Forced-alignment adapter registry — MMS only.

  - mms_fa : torchaudio MMS forced aligner (multilingual, char-level CTC)
"""
from __future__ import annotations

from backend.common.registry import make_registry
from .base import AlignRequest, AlignResult, AlignedSegment, BaseAligner

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "mms_fa": (".mms_fa_adapter", "MMSForcedAligner"),
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
