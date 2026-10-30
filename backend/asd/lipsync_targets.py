"""ASD-gated per-frame lip-sync targets.

Lip-sync models do not pick the speaking face: MuseTalk takes the first S3FD
detection (`d = d[0]`) and shapes its crop from DWPose `keypoints[0]`, an
unrelated whole-frame instance. This module picks the face and the box on our
side and hands the result over through MuseTalk's own `--use_saved_coord`
pickle, so nothing upstream is forked.

  * `speaking_track_per_frame` folds ASD scores over the tracker's face
    tracks into one chosen track per frame.
  * `musetalk_box` reproduces `get_landmark_and_bbox`'s arithmetic from
    InsightFace `landmark_3d_68` run on that track's box.

`landmark_3d_68` follows the same iBUG-68 order MuseTalk indexes, so the
indices transfer 1:1. `2d106det` is a different topology and is not used here.

A frame with no confirmed speaker becomes `(0, 0, 0, 0)`, MuseTalk's
`coord_placeholder`. Such a frame is written through unchanged only on a
MuseTalk carrying the 2026-10-18 patch (scripts/inference.py:253); before it,
a placeholder truncated the render.

Two people talking in one frame is not handled -- MuseTalk animates one box,
so the higher ASD score wins.
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

#: Speaking runs shorter than this are dropped as noise (~0.2 s at 25 fps).
MIN_SPEAK_RUN = 5

#: Sub-threshold gaps shorter than this inside a run are filled (~0.5 s at
#: 25 fps). Without it a brief ASD dip mid-sentence reads as a mouth stutter.
MAX_GAP_FILL = 12


def musetalk_box(
    landmarks68,
    *,
    bbox_shift: int = 0,
    fallback: Optional[Sequence[float]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """The crop box MuseTalk would derive from these landmarks.

    Mirrors `get_landmark_and_bbox` (preprocessing.py:113-129) exactly,
    including its fallback to the detector box: the model was trained on
    crops framed this way.
    """
    import numpy as np

    lm = np.asarray(landmarks68, dtype=np.float64)[:, :2].astype(np.int32)
    pivot = lm[PIVOT_IDX].copy()
    if bbox_shift:
        # The shift moves the TOP edge only. The bottom is `extra_margin`,
        # applied by inference.py at blend time and not handled here.
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
    track, so frames are matched by value, not by position.
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
    """Per-frame MuseTalk coord list, gated on who is speaking.

    Returns `(coord_list, stats)`. `coord_list` is what MuseTalk pickles: one
    4-tuple per frame in frame order, `PLACEHOLDER` where the frame is left
    alone. `stats` counts how many frames each branch produced.
    """
    import cv2
    import numpy as np

    from backend.common.gpu import ensure_onnxruntime_cuda

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

    # Must be exactly as long as MuseTalk's frame list: `zip` truncates to
    # the shorter one and the tail of the video would go unconsidered.
    if total and len(coords) < total:
        coords.extend([PLACEHOLDER] * (total - len(coords)))
        stats["placeholder"] += total - stats["frames"]
    log.info("lipsync targets: %d frames, %d with a speaking face "
             "(%d landmarked, %d detector-box fallback), %d left untouched",
             len(coords), stats["speaking"], stats["landmarked"],
             stats["fallback_box"], stats["placeholder"])
    return coords, stats


#: One loaded head per kind. `musetalk_box` needs the 68-point layout;
#: LatentSync indexes fixed positions in the 106-point one. Same detector and
#: weights, different output head -- they do not substitute.
_LMK: Dict[str, object] = {}

LMK_68 = "landmark_3d_68"
LMK_106 = "landmark_2d_106"


def _landmark_model(*, device: str, ensure, kind: str = LMK_68):
    """One InsightFace landmark head from buffalo_l, loaded once per kind."""
    global _LMK
    if kind in _LMK:
        return _LMK[kind]
    try:
        ensure()
    except Exception:                                   # noqa: BLE001
        log.debug("lipsync targets: ensure_onnxruntime_cuda failed",
                  exc_info=True)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=("detection", kind))
    app.prepare(ctx_id=0 if device != "cpu" else -1, det_size=(640, 640))
    _LMK[kind] = app.models[kind]
    return _LMK[kind]


def _landmarks_for_box(lmk, frame, box, *, kind: str = LMK_68):
    """Landmarks for ONE given box, not whichever face is biggest.

    The head takes the box it is handed and never re-detects, so this also
    works on a face below a detector's own size floor.

    Coordinates come back in the frame's pixel space. `AlignRestore` scales
    its template by `resolution / 256`, so the same points drive LatentSync
    1.5 at 256 and 1.6 at 512 unchanged.
    """
    import numpy as np
    from insightface.app.common import Face

    try:
        face = Face(bbox=np.asarray(box[:4], dtype=np.float32), det_score=1.0)
        lmk.get(frame, face)
        lm = getattr(face, kind, None)
        return None if lm is None else np.asarray(lm)
    except Exception:                                   # noqa: BLE001
        log.debug("lipsync targets: landmark failed on box %s", box,
                  exc_info=True)
        return None


# ---------------------------------------------------------------------------
# LatentSync targets
# ---------------------------------------------------------------------------
#
# LatentSync takes no box: `image_processor.affine_transform` builds its warp
# from three means over 106 landmarks. So the return type differs from
# `build_coord_list`'s, but the gating must not -- hence the shared calls to
# `speaking_track_per_frame` and `_smooth_runs` rather than a second copy of
# the threshold/min-run/gap-fill rules.

#: Written into the artifact so a stale targets file is identifiable.
TARGETS_PRODUCER = "buffalo_l_landmark_2d_106+talknet"


def _runs_from_tracks(frame_track) -> "np.ndarray":
    """`run_start[i]` -- True where frame i begins a new contiguous run.

    A run is a maximal stretch of consecutive frames held by one track.
    `AlignRestore.p_bias` is an EMA on the warp translation carried across
    calls, so without a reset at these boundaries it damps one person's frame
    toward another's across a cut, a skipped gap or a shard edge.
    """
    import numpy as np

    ft = np.asarray(frame_track)
    out = np.zeros(ft.shape[0], dtype=bool)
    for i in range(ft.shape[0]):
        if ft[i] < 0:
            continue
        # First frame overall, first after a gap, or first after a switch.
        if i == 0 or ft[i - 1] != ft[i]:
            out[i] = True
    return out


def build_targets(
    video: Path,
    tracks,
    asd,
    *,
    n_frames: Optional[int] = None,
    threshold: float = ASD_SPEAK_THRESHOLD,
    min_run: int = MIN_SPEAK_RUN,
    max_gap: int = MAX_GAP_FILL,
    device: str = "cpu",
) -> Tuple[Dict[str, "np.ndarray"], Dict]:
    """Per-frame LatentSync targets. Returns `(payload, stats)`.

    `payload` is ready for `np.savez_compressed` and is exactly what
    `latentsync/utils/targets.py::load_targets` reads back:

        width, height, fps, num_frames, producer
        frame_track  (N,)       int32    track index, -1 = leave frame alone
        frame_box    (N, 4)     float32  pixel bbox,      NaN where -1
        frame_lmk    (N, 106, 2) float32 pixel landmarks, NaN where -1
        run_start    (N,)       bool

    Coordinates are in `video`'s frame geometry, recorded as `width`/`height`
    because the render often is not that size -- a 4K source is capped to
    1080p first, and oversized landmarks place the mouth off-screen without
    erroring. The consumer scales.

    Detection runs at native resolution even when the render will be smaller:
    the landmark head crops to a fixed 192x192 either way, so detecting small
    throws pixels away while scaling coordinates down afterwards is exact.
    """
    import cv2
    import numpy as np

    from backend.common.gpu import ensure_onnxruntime_cuda

    video = Path(video)
    speaking = speaking_track_per_frame(
        tracks, asd, threshold=threshold, min_run=min_run, max_gap=max_gap)

    by_id = {t.track_id: t for t in tracks.tracks}
    # Stable string -> small int, so the artifact carries an index per frame
    # rather than a string. Sorted to keep the mapping reproducible.
    track_order = sorted({t for t in speaking.values()})
    track_index = {tid: i for i, tid in enumerate(track_order)}

    box_at: Dict[int, Tuple[float, ...]] = {}
    for f, tid in speaking.items():
        tr = by_id.get(tid)
        if tr is None:
            continue
        frames = np.asarray(tr.frames).astype(int)
        hit = np.nonzero(frames == f)[0]
        if hit.size:
            box_at[f] = tuple(float(v)
                              for v in np.asarray(tr.bboxes)[hit[0]][:4])

    lmk = _landmark_model(device=device, ensure=ensure_onnxruntime_cuda,
                          kind=LMK_106)

    cap = cv2.VideoCapture(str(video))
    total = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_track: List[int] = []
    frame_box: List[Tuple[float, float, float, float]] = []
    frame_lmk: List["np.ndarray"] = []
    nan_box = (float("nan"),) * 4
    nan_lmk = np.full((106, 2), np.nan, dtype=np.float32)

    stats = {"frames": 0, "speaking": 0, "landmarked": 0,
             "no_speaker": 0, "landmark_failed": 0,
             "track_ids": track_order}

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
            frame_track.append(-1)
            frame_box.append(nan_box)
            frame_lmk.append(nan_lmk)
            stats["no_speaker"] += 1
            idx += 1
            continue
        stats["speaking"] += 1
        lm = _landmarks_for_box(lmk, frame, box, kind=LMK_106)
        if lm is None or np.asarray(lm).shape[:2] != (106, 2):
            # Someone is speaking but the head could not place the points.
            # LatentSync needs landmarks, not a box, so leave the frame.
            frame_track.append(-1)
            frame_box.append(nan_box)
            frame_lmk.append(nan_lmk)
            stats["landmark_failed"] += 1
            idx += 1
            continue
        frame_track.append(track_index[speaking[idx]])
        frame_box.append(tuple(float(v) for v in box))
        frame_lmk.append(np.asarray(lm, dtype=np.float32)[:, :2])
        stats["landmarked"] += 1
        idx += 1
    cap.release()

    # Must be exactly as long as the clip the renderer reads:
    # `lipsync_pipeline` compares lengths and drops a mismatch.
    if total and len(frame_track) < total:
        pad = total - len(frame_track)
        frame_track.extend([-1] * pad)
        frame_box.extend([nan_box] * pad)
        frame_lmk.extend([nan_lmk] * pad)
        stats["no_speaker"] += pad

    ft = np.asarray(frame_track, dtype=np.int32)
    payload = {
        "width": np.array(width), "height": np.array(height),
        "fps": np.array(fps), "num_frames": np.array(len(ft)),
        "producer": np.array(TARGETS_PRODUCER),
        "frame_track": ft,
        "frame_box": np.asarray(frame_box, dtype=np.float32),
        "frame_lmk": np.asarray(frame_lmk, dtype=np.float32),
        "run_start": _runs_from_tracks(ft),
    }
    stats["runs"] = int(payload["run_start"].sum())
    stats["geometry"] = [width, height]
    log.info("latentsync targets: %d frames, %d targeted (%d landmarked, "
             "%d landmark-failed), %d left alone, %d run(s), built at %dx%d",
             len(ft), int((ft >= 0).sum()), stats["landmarked"],
             stats["landmark_failed"], int((ft < 0).sum()), stats["runs"],
             width, height)
    return payload, stats


def save_targets(payload: Dict[str, "np.ndarray"], path: Path) -> Path:
    """Write what `build_targets` produced, where the adapter will read it."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def slice_targets(payload: Dict[str, "np.ndarray"], start: int,
                  stop: int) -> Dict[str, "np.ndarray"]:
    """The window `[start, stop)` of a targets payload, renumbered from zero.

    Sharding, segmenting and window renders all renumber frames from zero, so
    an array indexed against the whole video lands wrong in every piece but
    the first. The targets equivalent of `backend.lipsync._split_coords`.

    `run_start` is re-marked on the first kept frame: the warp smoother's
    carry-over does not survive into a new process anyway.

    Geometry is carried through untouched -- slicing changes which frames are
    covered, never the resolution they were measured in.
    """
    import numpy as np

    n = int(np.asarray(payload["frame_track"]).shape[0])
    start = max(0, int(start))
    stop = min(n, int(stop))
    if stop <= start:
        raise ValueError(f"empty targets slice [{start}, {stop}) of {n}")

    rs = np.asarray(payload["run_start"])[start:stop].copy()
    ft = np.asarray(payload["frame_track"])[start:stop].copy()
    if rs.size and ft[0] >= 0:
        rs[0] = True

    out = dict(payload)
    out["frame_track"] = ft
    out["frame_box"] = np.asarray(payload["frame_box"])[start:stop].copy()
    out["frame_lmk"] = np.asarray(payload["frame_lmk"])[start:stop].copy()
    out["run_start"] = rs
    out["num_frames"] = np.array(stop - start)
    return out


def load_targets_payload(path: Path) -> Optional[Dict[str, "np.ndarray"]]:
    """Read a targets .npz back as a plain payload dict, or None.

    The vendored reader (`latentsync/utils/targets.py`) is in another conda
    env and cannot be imported here, so the schema lives in two places.
    """
    import numpy as np

    path = Path(path)
    if not path.is_file():
        return None
    try:
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    except Exception as e:                               # noqa: BLE001
        log.warning("targets: %s unreadable (%s); ignoring", path, e)
        return None
