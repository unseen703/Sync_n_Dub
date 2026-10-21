"""ASD-gated per-frame lip-sync targets (Route A).

MuseTalk has NO idea which face is speaking. `get_detections_for_batch` takes
the first S3FD detection (`d = d[0]`, musetalk/utils/face_detection/api.py:80 --
the same line CLAUDE.md records for Wav2Lip, because both vendor the same
face_detection lineage), and the 68 landmarks that actually shape the crop come
from DWPose `keypoints[0]`: instance 0 of a WHOLE-FRAME pose estimate that is
never matched to that box. On a multi-face shot the box and the landmarks can
describe different people, and neither is necessarily the one talking.

This module answers both halves on our side and hands MuseTalk the result
through its own `--use_saved_coord` pickle, so nothing upstream is forked:

  1. **Who** -- `speaking_track_per_frame` folds TalkNet's per-frame scores
     over the tracker's face tracks into one chosen track per frame.
  2. **Where** -- `musetalk_box` reproduces `get_landmark_and_bbox`'s
     arithmetic exactly, from InsightFace's `landmark_3d_68` instead of
     DWPose, run on the chosen track's own box.

**The landmark swap is verified, not assumed.** MuseTalk indexes
`face_land_mark[28..30]` -- the nose bridge in iBUG-68 order -- out of DWPose's
`keypoints[23:91]`. InsightFace's `landmark_3d_68` was checked against the same
convention on this host: points 27->30 descend the bridge monotonically, point
8 is the lowest point (the chin) and 48/54 are the mouth corners. The indices
therefore transfer 1:1. `2d106det` is a DIFFERENT topology and would need an
explicit mapping; it is deliberately not used here.

**A frame with no confirmed speaker becomes `(0, 0, 0, 0)`.** That is
MuseTalk's `coord_placeholder`, and since the 2026-10-18 patch
(scripts/inference.py:253) such a frame is WRITTEN UNCHANGED rather than
dropped -- so "nobody on screen is talking, leave this alone" is expressible.
Before that patch a placeholder truncated the render, so this module is only
safe against a MuseTalk carrying that fix.

Not handled: two people talking in the same frame. MuseTalk animates exactly
one box per frame, so the higher ASD score wins and the other is left alone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: `face_land_mark[29]` upstream -- the pivot `bbox_shift` moves.
PIVOT_IDX = 29

#: MuseTalk's "no face here" sentinel, and now also our "do not touch".
PLACEHOLDER: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

#: Per-frame ASD score at or above which a face counts as speaking. Matches
#: the repo's `asd_on_screen_threshold` default so one number means one thing.
ASD_SPEAK_THRESHOLD = 0.5

#: A speaking run shorter than this many frames is noise, not a line, and is
#: discarded. At 25 fps this is ~0.2 s.
MIN_SPEAK_RUN = 5

#: A sub-threshold gap SHORTER than this inside a speaking run is filled.
#: Without it a two-frame ASD dip mid-sentence leaves two un-animated frames
#: between animated ones, which reads as a mouth stutter -- worse than the
#: wrong-face problem being fixed. At 25 fps this is ~0.5 s.
MAX_GAP_FILL = 12


def musetalk_box(
    landmarks68,
    *,
    bbox_shift: int = 0,
    fallback: Optional[Sequence[float]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """The crop box MuseTalk would derive from these landmarks.

    Mirrors `get_landmark_and_bbox` (preprocessing.py:113-129) line for line,
    including its degenerate-case fallback to the detector box. Reproduced
    rather than approximated because the model was trained on crops framed
    this way: a box that merely looks reasonable is a different distribution.
    """
    import numpy as np

    lm = np.asarray(landmarks68, dtype=np.float64)[:, :2].astype(np.int32)
    pivot = lm[PIVOT_IDX].copy()
    if bbox_shift:
        # Upstream adds the shift to the pivot's y, which moves the TOP edge.
        # It does not touch the bottom -- that is `extra_margin`, applied by
        # inference.py itself at blend time and so deliberately absent here.
        pivot[1] = int(pivot[1]) + int(bbox_shift)
    half_face_dist = int(lm[:, 1].max()) - int(pivot[1])
    upper_bond = max(0, int(pivot[1]) - half_face_dist)

    x1, y1 = int(lm[:, 0].min()), int(upper_bond)
    x2, y2 = int(lm[:, 0].max()), int(lm[:, 1].max())
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        if fallback is None:
            return None
        f = [int(v) for v in fallback]
        return (f[0], f[1], f[2], f[3])
    return (x1, y1, x2, y2)


def speaking_track_per_frame(
    tracks,
    asd,
    *,
    threshold: float = ASD_SPEAK_THRESHOLD,
    min_run: int = MIN_SPEAK_RUN,
    max_gap: int = MAX_GAP_FILL,
) -> Dict[int, str]:
    """frame index -> the track_id speaking in it, for frames that have one.

    `ASDScore.frames` / `.scores` are parallel and omit frames outside the
    track, so alignment is explicit rather than positional.
    """
    import numpy as np

    # Highest-scoring track per frame, before any temporal cleanup.
    best: Dict[int, Tuple[float, str]] = {}
    for tr in tracks.tracks:
        sc = asd.per_track.get(tr.track_id)
        if sc is None:
            continue
        frames = np.asarray(sc.frames).astype(int)
        scores = np.asarray(sc.scores, dtype=float)
        n = min(len(frames), len(scores))
        for f, s in zip(frames[:n], scores[:n]):
            if s < threshold:
                continue
            prev = best.get(int(f))
            if prev is None or s > prev[0]:
                best[int(f)] = (float(s), tr.track_id)

    chosen = {f: tid for f, (_, tid) in best.items()}
    return _smooth_runs(chosen, min_run=min_run, max_gap=max_gap)


def _smooth_runs(chosen: Dict[int, str], *, min_run: int,
                 max_gap: int) -> Dict[int, str]:
    """Fill short gaps inside a run, then drop runs too short to be a line."""
    if not chosen:
        return {}
    out = dict(chosen)

    # Gap fill, per track, so one speaker's dip is not filled by another.
    by_track: Dict[str, List[int]] = {}
    for f, tid in chosen.items():
        by_track.setdefault(tid, []).append(f)
    for tid, frames in by_track.items():
        frames.sort()
        for a, b in zip(frames, frames[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for f in range(a + 1, b):
                    # Only fill a frame nobody else claimed.
                    out.setdefault(f, tid)

    # Drop short runs.
    frames_sorted = sorted(out)
    runs: List[List[int]] = []
    for f in frames_sorted:
        if runs and f == runs[-1][-1] + 1 and out[f] == out[runs[-1][-1]]:
            runs[-1].append(f)
        else:
            runs.append([f])
    for run in runs:
        if len(run) < min_run:
            for f in run:
                out.pop(f, None)
    return out


def build_coord_list(
    video: Path,
    tracks,
    asd,
    *,
    n_frames: Optional[int] = None,
    bbox_shift: int = 0,
    threshold: float = ASD_SPEAK_THRESHOLD,
    min_run: int = MIN_SPEAK_RUN,
    max_gap: int = MAX_GAP_FILL,
    device: str = "cpu",
) -> Tuple[List[Tuple], Dict]:
    """Per-frame MuseTalk coord list, gated on who is actually speaking.

    Returns `(coord_list, stats)`. `coord_list` is exactly what MuseTalk
    pickles: one 4-tuple per frame in frame order, `PLACEHOLDER` where the
    frame must be left alone. `stats` records how many frames each branch
    produced, because "it ran" and "it animated the right face" are different
    claims and only the second one matters.
    """
    import cv2
    import numpy as np

    from .gpu import ensure_onnxruntime_cuda

    video = Path(video)
    speaking = speaking_track_per_frame(
        tracks, asd, threshold=threshold, min_run=min_run, max_gap=max_gap)

    by_id = {t.track_id: t for t in tracks.tracks}
    # frame -> bbox, for the chosen track only.
    box_at: Dict[int, Tuple[int, ...]] = {}
    for f, tid in speaking.items():
        tr = by_id.get(tid)
        if tr is None:
            continue
        frames = np.asarray(tr.frames).astype(int)
        hit = np.nonzero(frames == f)[0]
        if hit.size:
            box_at[f] = tuple(int(v) for v in np.asarray(tr.bboxes)[hit[0]][:4])

    lmk = _landmark_model(device=device, ensure=ensure_onnxruntime_cuda)

    cap = cv2.VideoCapture(str(video))
    total = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    coords: List[Tuple] = []
    stats = {"frames": 0, "speaking": 0, "landmarked": 0,
             "fallback_box": 0, "placeholder": 0, "no_speaker": 0,
             "track_ids": sorted({t for t in speaking.values()})}

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total and idx >= total:
            break
        stats["frames"] += 1
        box = box_at.get(idx)
        if box is None:
            coords.append(PLACEHOLDER)
            stats["no_speaker"] += 1
            stats["placeholder"] += 1
            idx += 1
            continue
        stats["speaking"] += 1
        lm = _landmarks_for_box(lmk, frame, box)
        if lm is None:
            coords.append(tuple(int(v) for v in box))
            stats["fallback_box"] += 1
            idx += 1
            continue
        mb = musetalk_box(lm, bbox_shift=bbox_shift, fallback=box)
        if mb is None:
            coords.append(PLACEHOLDER)
            stats["placeholder"] += 1
        else:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = mb
            mb = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            coords.append(mb)
            stats["landmarked"] += 1
        idx += 1
    cap.release()

    # The list must be exactly as long as the frame list MuseTalk reads, or
    # `zip(coord_list, frame_list)` silently truncates to the shorter one and
    # the tail of the video is never considered.
    if total and len(coords) < total:
        coords.extend([PLACEHOLDER] * (total - len(coords)))
        stats["placeholder"] += total - stats["frames"]
    log.info("lipsync targets: %d frames, %d with a speaking face "
             "(%d landmarked, %d detector-box fallback), %d left untouched",
             len(coords), stats["speaking"], stats["landmarked"],
             stats["fallback_box"], stats["placeholder"])
    return coords, stats


_LMK = None


def _landmark_model(*, device: str, ensure):
    """InsightFace's 68-point landmark head, loaded once."""
    global _LMK
    if _LMK is not None:
        return _LMK
    try:
        ensure()
    except Exception:                                   # noqa: BLE001
        log.debug("lipsync targets: ensure_onnxruntime_cuda failed",
                  exc_info=True)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=("detection", "landmark_3d_68"))
    app.prepare(ctx_id=0 if device != "cpu" else -1, det_size=(640, 640))
    _LMK = app.models["landmark_3d_68"]
    return _LMK


def _landmarks_for_box(lmk, frame, box):
    """68 landmarks for ONE given box -- not whichever face is biggest.

    The landmark head takes the box it is handed, which is the whole point:
    it is how the chosen (speaking) face gets its own landmarks instead of
    inheriting a whole-frame pose estimate's first instance.
    """
    import numpy as np
    from insightface.app.common import Face

    try:
        face = Face(bbox=np.asarray(box[:4], dtype=np.float32), det_score=1.0)
        lmk.get(frame, face)
        lm = getattr(face, "landmark_3d_68", None)
        return None if lm is None else np.asarray(lm)
    except Exception:                                   # noqa: BLE001
        log.debug("lipsync targets: landmark failed on box %s", box,
                  exc_info=True)
        return None
