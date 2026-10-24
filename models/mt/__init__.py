"""Machine-translation adapter registry.

  - bodhan      : Bodhan Indic-Translate via REST (POST /v1/chat/completions)
  - nllb        : facebook/nllb-200-* via local transformers
"""
from __future__ import annotations

from .._registry import make_registry
from .base import BaseMT, MTRequest, MTResult, MTSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanMT"),
    "nllb": (".nllb_adapter", "NLLBMT"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="MT model",
    base=BaseMT,
    import_noun="MT model",
)

available_models = _registry.available
get_mt = _registry.get

__all__ = [
    "BaseMT",
    "MTRequest",
    "MTResult",
    "MTSegment",
    "available_models",
    "get_mt",
]
