"""Pick a lip-sync model from how big the faces actually are.

Why there is a choice to make
-----------------------------
Measured on this host (`.claude/skills/clip-run`): wav2lip runs at **3-5x**
clip length, MuseTalk at **8-21x**, LatentSync at ~83x. Wav2Lip repaints a
96x96 mouth crop; MuseTalk works at 256x256. On a close-up the extra
resolution is the whole point; on a wide lecture shot where the speaker's
face is a hundred pixels tall, MuseTalk spends four times the GPU hours
rendering detail the frame cannot show.

The statistic
-------------
Per sampled frame, the height of the **largest** face box as a fraction of
frame height; the **median** over frames. The median, not the max: a wide
shot that cuts to a close-up once has a p90 of 40% while being a wide shot
throughout.

Calibration — read this before trusting it
------------------------------------------
Measured on **three clips from one source video** (`svwrTOm55y4`):

    svwrTOm55y4_1400to1420  (wide, 9 face tracks)   median 0.105  -> wav2lip
    svwrTOm55y4_240to270    (close-up)              median 0.419  -> musetalk
    svwrTOm55y4_closeup3s   (close-up)              median 0.401  -> musetalk

The threshold of **0.25** sits in the empty gap between those two clusters.
That makes it a defensible default with a wide margin, **not a validated
classifier** — three clips from one video is not a calibration set. It is
surfaced in the UI with its reason, it is overridable per job, and
`DUBBING_UI_FACE_THRESHOLD` moves it without a code change.

`LD_PRELOAD` is the caller's problem: detection pulls in cv2/insightface, and
whichever `libstdc++` loads first decides whether the import works at all
(CLAUDE.md section 1). `scripts/ui_run_job.sh` exports it.
"""
from __future__ import annotations

import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("ui.facepick")

#: The empty gap between the two measured clusters (0.105 vs 0.401/0.419).
DEFAULT_THRESHOLD = 0.25

THRESHOLD_ENV = "DUBBING_UI_FACE_THRESHOLD"

#: The clips the threshold was drawn from, kept next to the number so the
#: evidence and the constant cannot drift apart. `tests/test_ui_create.py`
#: asserts the picker still classifies these as recorded.
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

    Returns a dict even when nothing is found, so the caller logs a number
    rather than branching on an exception. `median` is `None` only when no
    frame had a face at all.
    """
    from backend.faces import detect_faces           # lazy: pulls in cv2/onnx

    video = Path(video)
    total = _frame_count(video)
    every_n = max(1, int(total // max(1, sample_frames))) if total else 1

    det = detect_faces(video, detector=detector, every_n=every_n,
                       min_score=min_score)
    height = float(getattr(det, "height", 0) or 0)
    fracs = _largest_face_fractions(det, height)

    # `det.detections` holds only the frames that HAD a face, so the number
    # of frames actually looked at has to be derived from the spacing.
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
        # Last resort: duration at an assumed 25 fps. Only the *spacing* of
        # the sample depends on this, so being wrong costs a few extra or a
        # few fewer sampled frames, never a wrong answer.
        from backend.media_io import probe_duration
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
    """`(model, reason)` — the reason is shown to the operator verbatim.

    The reason says what was measured, what the threshold was, and that the
    threshold is calibrated on three clips from one video. An operator who
    disagrees overrides the model; an operator who is not told the basis
    cannot.
    """
    thr = configured_threshold() if threshold is None else float(threshold)
    caveat = ("This rule is calibrated on three clips from one video, so it "
              "is a default, not a verdict — override it if the shot is "
              "unusual.")

    if not median_frac:
        return ("wav2lip",
                "No faces were found in the sampled frames, so there is no "
                "close-up to justify MuseTalk's 8-21x render time. Using "
                "Wav2Lip (3-5x). " + caveat)

    pct = 100.0 * float(median_frac)
    thr_pct = 100.0 * float(thr)
    if median_frac >= thr:
        return ("musetalk",
                f"Faces fill about {pct:.0f}% of the frame height (threshold "
                f"{thr_pct:.0f}%), so this is a close-up and the extra "
                f"resolution shows. Using MuseTalk (256x256 crop, 8-21x clip "
                f"length). " + caveat)
    return ("wav2lip",
            f"Faces fill about {pct:.0f}% of the frame height (threshold "
            f"{thr_pct:.0f}%), so this is a wide shot and MuseTalk's extra "
            f"resolution would not be visible. Using Wav2Lip (96x96 mouth "
            f"crop, 3-5x clip length instead of 8-21x). " + caveat)
