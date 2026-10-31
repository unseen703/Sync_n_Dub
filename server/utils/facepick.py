"""Suggest a lip-sync model from how large the faces are on screen.

Samples frames, takes the median height of the largest face box as a fraction
of frame height, and compares it to a threshold. Detection pulls in
cv2/insightface, so the caller must set `LD_PRELOAD` for libstdc++.
"""
from __future__ import annotations

import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("server.facepick")

#: The empty gap between the two measured clusters (0.105 vs 0.401/0.419).
DEFAULT_THRESHOLD = 0.25

THRESHOLD_ENV = "DUBBING_UI_FACE_THRESHOLD"

#: The three clips the threshold was drawn from, kept beside the constant so
#: the two cannot drift apart.
CALIBRATION: Dict[str, Tuple[float, str]] = {
    "svwrTOm55y4_1400to1420": (0.105, "wav2lip"),
    "svwrTOm55y4_240to270":   (0.419, "musetalk"),
    "svwrTOm55y4_closeup3s":  (0.401, "musetalk"),
}


def configured_threshold() -> float:
    """The configured face-height threshold, or the calibrated default."""
    raw = os.environ.get(THRESHOLD_ENV)
    if not raw:
        return DEFAULT_THRESHOLD
    try:
        val = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %.2f",
                    THRESHOLD_ENV, raw, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD
    if not 0.0 < val < 1.0:
        log.warning("%s=%s is not a fraction of frame height; using %.2f",
                    THRESHOLD_ENV, val, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD
    return val


# ---------------------------------------------------------------------------
# measuring
# ---------------------------------------------------------------------------

def face_size_stats(video: Path, *, sample_frames: int = 40,
                    detector: str = "retinaface",
                    min_score: float = 0.5) -> Dict[str, Any]:
    """Median/p90 largest-face height as a fraction of frame height.

    Always returns a dict, so the caller never branches on an exception.
    `median` is None only when no sampled frame had a face.
    """
    from backend.faces.faces import detect_faces           # lazy: pulls in cv2/onnx

    video = Path(video)
    total = _frame_count(video)
    every_n = max(1, int(total // max(1, sample_frames))) if total else 1

    det = detect_faces(video, detector=detector, every_n=every_n,
                       min_score=min_score)
    height = float(getattr(det, "height", 0) or 0)
    fracs = _largest_face_fractions(det, height)

    # `det.detections` holds only frames that had a face, so the number
    # sampled has to come from the spacing.
    out: Dict[str, Any] = {
        "n_frames_sampled": -(-total // every_n) if total else len(det.detections),
        "n_frames_with_face": len(fracs),
        "every_n": every_n,
        "frame_height": int(height),
        "median": None,
        "p90": None,
        "max": None,
    }
    if fracs:
        ordered = sorted(fracs)
        out["median"] = round(statistics.median(ordered), 4)
        out["p90"] = round(ordered[min(len(ordered) - 1,
                                       int(round(0.9 * (len(ordered) - 1))))], 4)
        out["max"] = round(ordered[-1], 4)
    log.info("face size: median=%s p90=%s over %d/%d sampled frames "
             "(every_n=%d)", out["median"], out["p90"],
             out["n_frames_with_face"], out["n_frames_sampled"], every_n)
    return out


def _largest_face_fractions(det, frame_height: float) -> List[float]:
    if frame_height <= 0:
        return []
    out: List[float] = []
    for boxes in det.detections.values():
        if not boxes:
            continue
        tallest = max(float(b.y2) - float(b.y1) for b in boxes)
        if tallest > 0:
            out.append(tallest / frame_height)
    return out


def _frame_count(video: Path) -> int:
    """Total frames, for spacing the sample. 0 when it cannot be determined."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        if n > 0:
            return n
    except Exception as e:                                  # pragma: no cover
        log.debug("cv2 frame count failed: %s", e)
    try:
        # Last resort: duration at an assumed 25 fps. Only the sample spacing
        # depends on it, so being wrong costs frames, not correctness.
        from backend.common.media_io import probe_duration
        return int(round(probe_duration(video) * 25.0))
    except Exception as e:                                  # pragma: no cover
        log.warning("could not determine frame count for %s: %s", video, e)
        return 0


# ---------------------------------------------------------------------------
# deciding
# ---------------------------------------------------------------------------

def pick_lipsync_model(median_frac: Optional[float],
                       threshold: Optional[float] = None
                       ) -> Tuple[str, str]:
    """Return `(model, reason)`; the reason is shown to the operator verbatim.

    Keep it to one clause — the panel already prints the measured size and the
    per-model estimates above it. Suggestion language, not decision language:
    the operator still picks, and the list beside this has nothing selected.
    """
    thr = configured_threshold() if threshold is None else float(threshold)

    if not median_frac:
        return ("wav2lip",
                "No faces were found in the sampled frames — Wav2Lip is the "
                "safer fit.")
    if median_frac >= thr:
        return ("musetalk",
                "Larger faces in frame — MuseTalk performs better here.")
    return ("wav2lip",
            "Smaller faces in frame — Wav2Lip works better here.")
