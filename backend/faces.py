"""Backend-facing entry point for face detection + tracking.

Keeps the detect/track registries behind one function so the pipeline / CLI
have one place to import from, same way `backend.diarize` fronts the
diarizer registry.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from models.faces import FaceDetections, FaceTracks
from models.faces.detect import available_detectors, get_detector
from models.faces.track import available_trackers, get_tracker

log = logging.getLogger(__name__)


def detect_faces(
    video: Path,
    *,
    detector: str = "retinaface",
    device: Optional[str] = None,
    every_n: int = 1,
    min_score: float = 0.5,
    **detector_kwargs,
) -> FaceDetections:
    """Run `detector` over `video`."""
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    if detector not in available_detectors():
        raise ValueError(
            f"Unknown face detector {detector!r}. "
            f"Available: {', '.join(available_detectors())}"
        )
    log.info("face detect: %s with %s on %s (every_n=%d, min_score=%.2f)",
             video.name, detector, device or "auto", every_n, min_score)
    d = get_detector(detector, device=device, **detector_kwargs)
    return d.detect(video, every_n=every_n, min_score=min_score)


def track_faces(
    detections: FaceDetections,
    *,
    tracker: str = "bytetrack",
    **tracker_kwargs,
) -> FaceTracks:
    """Associate per-frame detections into stable tracks."""
    if tracker not in available_trackers():
        raise ValueError(
            f"Unknown face tracker {tracker!r}. "
            f"Available: {', '.join(available_trackers())}"
        )
    log.info("face track: %s on %d frames of detections",
             tracker, len(detections.detections))
    t = get_tracker(tracker, **tracker_kwargs)
    return t.track(detections)


def detect_and_track(
    video: Path,
    *,
    detector: str = "retinaface",
    tracker: str = "bytetrack",
    device: Optional[str] = None,
    every_n: int = 1,
    min_score: float = 0.5,
    tracks_out: Optional[Path] = None,
    detector_kwargs: Optional[dict] = None,
    tracker_kwargs: Optional[dict] = None,
) -> FaceTracks:
    """One call: detect, track, optionally save raw tracks to .npz."""
    detections = detect_faces(
        video,
        detector=detector, device=device,
        every_n=every_n, min_score=min_score,
        **(detector_kwargs or {}),
    )
    tracks = track_faces(detections, tracker=tracker, **(tracker_kwargs or {}))
    if tracks_out is not None:
        tracks.save_npz(Path(tracks_out))
        log.info("face track: wrote %s", tracks_out)
    return tracks
