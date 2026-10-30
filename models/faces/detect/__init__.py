"""Face-detector registry.

  - retinaface : InsightFace SCRFD, or the `retina-face` package
  - mediapipe  : CPU-only, bbox-oriented

Adapters import lazily, so an env with only one of them installed can still
list the registry.
"""
from __future__ import annotations

from ..._registry import make_registry
from ..base import BaseFaceDetector, FaceBox, FaceDetections

_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    "retinaface": (".retinaface_adapter", "RetinaFaceDetector"),
    "mediapipe": (".mediapipe_adapter", "MediaPipeDetector"),
}

_registry = make_registry(
    package=__name__,
    spec=_REGISTRY_SPEC,
    noun="face detector",
    base=BaseFaceDetector,
    import_noun="Face detector",
    import_detail="is not installed in this environment. Install the "
                  "matching extras (e.g. `pip install retina-face` or "
                  "`pip install mediapipe`).",
    verbose_warning=True,
)

available_detectors = _registry.available
get_detector = _registry.get

__all__ = [
    "BaseFaceDetector",
    "FaceBox",
    "FaceDetections",
    "available_detectors",
    "get_detector",
]
