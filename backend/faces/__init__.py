"""Face stack: detection, tracking and the shared data types.

  - models.faces.detect  (RetinaFace, MediaPipe)
  - models.faces.track   (ByteTrack)

Each submodule exposes `available_models()` and `get_<X>(name, ...)`. Callers
never import an adapter class directly: the registry drops unknown kwargs with
a warning, so one call site can pass the union of what any adapter wants.
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
_STAGE_EXPORTS = {
    "SegmentAssignment", "assign_faces_to_turns", "assignments_to_jsonable",
    "detect_and_track", "detect_faces", "preimport_face_stack",
    "speaker_face_tracks", "track_faces",
}


def __getattr__(name):
    if name in _STAGE_EXPORTS:
        import importlib
        return getattr(importlib.import_module(".faces", __name__), name)
    raise AttributeError(name)

__all__ = [
    "BaseFaceDetector",
    "BaseFaceTracker",
    "FaceBox",
    "FaceDetections",
    "FaceTrack",
    "FaceTracks",
    "SegmentAssignment",
    "assign_faces_to_turns",
    "assignments_to_jsonable",
    "detect_and_track",
    "detect_faces",
    "preimport_face_stack",
    "speaker_face_tracks",
    "track_faces",
    "tracks_from_detections_array",
]
