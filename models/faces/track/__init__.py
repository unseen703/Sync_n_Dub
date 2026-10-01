"""Face-tracker registry — currently ByteTrack only (plan.md default)."""
from __future__ import annotations

import importlib
import inspect
from typing import Type

from ..base import BaseFaceTracker, FaceTracks

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "bytetrack": (".bytetrack_adapter", "ByteTrackTracker"),
}


def _load(key: str) -> Type[BaseFaceTracker]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_trackers() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseFaceTracker]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_tracker(name: str, **kwargs) -> BaseFaceTracker:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown face tracker {name!r}. "
            f"Available: {', '.join(available_trackers())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Face tracker {key!r} is not installed. "
            f"Install with `pip install supervision`. Original error: {e}"
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
    return cls(**kwargs)


__all__ = [
    "BaseFaceTracker",
    "FaceTracks",
    "available_trackers",
    "get_tracker",
]
