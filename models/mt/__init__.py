"""Machine-translation adapter registry.

  - bodhan      : Bodhan Indic-Translate via REST (POST /v1/chat/completions)
  - nllb        : facebook/nllb-200-* via local transformers
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseMT, MTRequest, MTResult, MTSegment

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bodhan": (".bodhan_adapter", "BodhanMT"),
    "nllb": (".nllb_adapter", "NLLBMT"),
}


def _load(key: str) -> Type[BaseMT]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseMT]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_mt(name: str, *, device: Optional[str] = None, **kwargs) -> BaseMT:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown MT model {name!r}. Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"MT model {key!r} is not installed. Original error: {e}"
        ) from e
    accepted = _accepted_kwargs(cls)
    if accepted:
        unknown = set(kwargs) - accepted
        if unknown:
            import warnings
            warnings.warn(
                f"{cls.__name__} does not accept kwargs {sorted(unknown)!r}; dropping."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(device=device, **kwargs)


__all__ = [
    "BaseMT",
    "MTRequest",
    "MTResult",
    "MTSegment",
    "available_models",
    "get_mt",
]
