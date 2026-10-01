"""Face-detector registry.

Add a new detector by adding its spec to `_REGISTRY_SPEC`. The lazy-import
pattern matches `models.diarization.__init__`: environments that only have
one adapter installed can still `from models.faces.detect import ...`.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

from ..base import BaseFaceDetector, FaceBox, FaceDetections

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "retinaface": (".retinaface_adapter", "RetinaFaceDetector"),
    "mediapipe": (".mediapipe_adapter", "MediaPipeDetector"),
}


def _load(key: str) -> Type[BaseFaceDetector]:
    module_path, cls_name = _REGISTRY_SPEC[key]
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, cls_name)


def available_detectors() -> list[str]:
    return sorted(_REGISTRY_SPEC)


def _accepted_kwargs(cls: Type[BaseFaceDetector]) -> set[str]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set()
    return {
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def get_detector(
    name: str, *, device: Optional[str] = None, **kwargs
) -> BaseFaceDetector:
    key = name.strip().lower()
    if key not in _REGISTRY_SPEC:
        raise ValueError(
            f"Unknown face detector {name!r}. "
            f"Available: {', '.join(available_detectors())}"
        )
    try:
        cls = _load(key)
    except ImportError as e:
        raise ImportError(
            f"Face detector {key!r} is not installed in this environment. "
            f"Install the matching extras (e.g. `pip install retina-face` or "
            f"`pip install mediapipe`). Original error: {e}"
        ) from e
    accepted = _accepted_kwargs(cls)
    if accepted:
        unknown = set(kwargs) - accepted
        if unknown:
            import warnings
            warnings.warn(
                f"{cls.__name__} does not accept kwargs {sorted(unknown)!r}; dropping. "
                f"Accepted: {sorted(accepted - {'device'})!r}."
            )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(device=device, **kwargs)


__all__ = [
    "BaseFaceDetector",
    "FaceBox",
    "FaceDetections",
    "available_detectors",
    "get_detector",
]
