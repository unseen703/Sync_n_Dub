"""Face stack: detection, tracking, and the shared data types.

Submodules:
  - models.faces.detect  (RetinaFace, MediaPipe)
  - models.faces.track   (ByteTrack)

Each submodule exposes `available_models()` and `get_<X>(name, ...)` the same
way `models.diarization` does. Downstream code never imports an adapter class
directly — the registry drops unknown kwargs with a warning so one call site
can pass the union of what any adapter wants.
"""
from .base import (
    BaseFaceDetector,
    BaseFaceTracker,
    FaceBox,
    FaceDetections,
    FaceTrack,
    FaceTracks,
    tracks_from_detections_array,
)

__all__ = [
    "BaseFaceDetector",
    "BaseFaceTracker",
    "FaceBox",
    "FaceDetections",
    "FaceTrack",
    "FaceTracks",
    "tracks_from_detections_array",
]
