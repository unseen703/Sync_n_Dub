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
from .faces import (
    SegmentAssignment,
    assign_faces_to_turns,
    assignments_to_jsonable,
    detect_and_track,
    detect_faces,
    preimport_face_stack,
    speaker_face_tracks,
    track_faces,
)

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
