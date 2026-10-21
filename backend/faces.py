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


def preimport_face_stack() -> bool:
    """Load insightface BEFORE anything else claims the shared C libraries.

    Third instance of the import-order trap in CLAUDE.md section 1, and the
    first one `LD_PRELOAD` does not cover, because the clashing pair is not
    libstdc++:

        import pyannote.audio
        from insightface.app import FaceAnalysis
        ImportError: .../cv2/.../libtiff.so.6: undefined symbol:
                     jpeg12_write_raw_data, version LIBJPEG_8.0

    `pyannote.audio` pulls in a libjpeg without the 12-bit symbols that cv2's
    bundled `libtiff.so.6` wants, and whichever lands first wins for the life
    of the process. Reversing the order fixes it outright — verified both
    ways on 2026-10-17.

    This hid for weeks because `--lipsync auto` runs the face-size probe
    *before* diarization, so insightface was already in `sys.modules` by
    stage 4. Name a lip-sync model explicitly and the probe short-circuits
    (`ui/runner.py::_probe_and_pick`), diarization imports first, and stage 5
    dies with "RetinaFace needs either `insightface` ... or `retina-face`" —
    a message that points at a missing package when the package is installed
    and imports fine on its own.

    Returns whether the import succeeded. Never raises: a host genuinely
    without insightface must still reach the adapter's own error, which is
    the one that explains the fix.
    """
    try:
        import insightface.app                              # noqa: F401
    except Exception as e:                                  # noqa: BLE001
        log.debug("face stack pre-import skipped (%s: %s)", type(e).__name__, e)
        return False
    return True


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
