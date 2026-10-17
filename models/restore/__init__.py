"""Face-restoration adapter registry.

  - codeformer : sczhou/CodeFormer — codebook-lookup blind face
                 restoration (non-commercial)

A restoration adapter runs *after* lip-sync, re-rendering the face the
lip-sync model left soft. `backend.lipsync.run_lipsync(..., restore=...)`
chains it automatically; `scripts/run_lipsync_only.py --restore` does the
same for a one-off.

Adding a model is one adapter plus one line in `_REGISTRY_SPEC` below —
GFPGAN and RestoreFormer would both drop in the same way.

Each adapter wraps an external repo clone under `third_party/`.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from .base import BaseRestore, RestoreRequest, RestoreResult

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "codeformer": (".codeformer_adapter", "CodeFormerRestore"),
}


def _load(key: str) -> Type[BaseRestore]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_models() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseRestore]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_restore(name: str, *, device: Optional[str] = None, **kwargs) -> BaseRestore:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown restoration model {name!r}. "
            f"Available: {', '.join(available_models())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Restoration model {key!r} is not installed. Original error: {e}"
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
    "BaseRestore",
    "RestoreRequest",
    "RestoreResult",
    "available_models",
    "get_restore",
]
