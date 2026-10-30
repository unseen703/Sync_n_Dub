"""Machine-translation adapter registry — Bodhan Indic-Translate only.

  - bodhan : Bodhan Indic-Translate via REST (POST /v1/chat/completions)
"""
from __future__ import annotations

from backend.common.registry import make_registry
from .base import BaseMT, MTRequest, MTResult, MTSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanMT"),
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

def __getattr__(name):
    if name in {"codemix_segments", "translate_asr", "write_mt_outputs"}:
        import importlib
        return getattr(importlib.import_module(".translate", __name__), name)
    raise AttributeError(name)

__all__ = [
    "BaseMT",
    "MTRequest",
    "MTResult",
    "MTSegment",
    "available_models",
    "get_mt",
]
