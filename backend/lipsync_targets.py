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


#: One loaded head per kind. Keyed because the two consumers want DIFFERENT
#: heads and neither substitutes for the other: `musetalk_box` reads the
#: 68-point layout, while LatentSync indexes fixed positions in the 106-point
#: one (43/48/49/51/50 for the left eyebrow, 101:106 for the right, 74/77/83/86
#: for the nose -- see `image_processor.affine_transform`). Same detector,
#: same weights on disk, different output head.
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
    """Landmarks for ONE given box -- not whichever face is biggest.

    The landmark head takes the box it is handed, which is the whole point:
    it is how the chosen (speaking) face gets its own landmarks instead of
    inheriting a whole-frame pose estimate's first instance. It is also the
    reason this works on a face too small for a detector's own floor --
    nothing re-detects, so nothing re-filters.

    Coordinates come back in the FRAME's pixel space, which is what makes
    them portable: `AlignRestore` scales its template by `resolution / 256`,
    so the same points drive LatentSync 1.5 at 256 and 1.6 at 512 unchanged.
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
# `build_coord_list` above answers "which BOX does MuseTalk paint into".
# LatentSync does not take a box: its only input from the detection stage is
# 106 landmarks, from which `image_processor.affine_transform` takes three
# means and builds the warp. So the two cannot share a return type -- but
# they MUST share the gating, which is why `speaking_track_per_frame` and
# `_smooth_runs` are called here rather than reimplemented. A second copy of
# the threshold/min-run/gap-fill rules would drift from the first, and the
# drift would show up as "the wrong person is animated" months later.

#: Producer string written into the artifact, so a stale targets file can be
#: told apart from one built by a later version of this code.
TARGETS_PRODUCER = "buffalo_l_landmark_2d_106+talknet"


def _runs_from_tracks(frame_track) -> "np.ndarray":
    """`run_start[i]` -- True where frame i begins a new contiguous run.

    A run is a maximal stretch of consecutive frames held by one track. The
    flag exists because `AlignRestore.p_bias` is an EMA on the warp
    translation carried ACROSS calls: without a reset it damps frame i toward
    frame i-1 even when the two are different people (a cut), seconds apart
    (a gap this gating skipped), or either side of a shard boundary. Marking
    the boundaries here puts that discipline in the artifact, where it can be
    audited, instead of leaving each consumer to re-derive it.
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

    Coordinates are in the frame geometry of `video`, and `width`/`height`
    record which that was, because the render is routinely NOT that size --
    a 4K source is capped to 1080p before rendering, and landmarks twice the
    size of the frame they warp place the mouth off-screen without erroring.
    The consumer scales; this function only has to be honest about the
    geometry it measured in.

    Detect at native resolution even when the render will be smaller. The
    landmark head crops to a fixed 192x192 whatever it is given, so a face
    that is 160x200 in 4K arrives near-native while the same face at 1080p
    arrives 2x interpolated. Scaling coordinates down afterwards is exact;
    re-detecting smaller throws the pixels away for good.
    """
    import cv2
    import numpy as np

    from .gpu import ensure_onnxruntime_cuda

    video = Path(video)
    speaking = speaking_track_per_frame(
        tracks, asd, threshold=threshold, min_run=min_run, max_gap=max_gap)

    by_id = {t.track_id: t for t in tracks.tracks}
    # Stable string -> small int, so the artifact carries an index rather than
    # a variable-length string per frame. Sorted so the mapping is reproducible
    # across runs of the same job.
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
            # The gating says someone is speaking but the head could not
            # place the points. Animating from a box alone is not possible
            # here -- LatentSync needs the landmarks -- so the frame is left
            # untouched rather than warped from a guess.
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

    # The arrays must be exactly as long as the clip the renderer reads, or
    # the consumer refuses them -- `lipsync_pipeline` compares lengths and
    # drops a mismatch rather than animating the wrong frames.
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

    Every path that cuts a clip -- sharding across GPUs, segmenting for
    memory, a window render -- renumbers frames from zero, so an array
    indexed against the whole video lands on the wrong frames in every piece
    but the first. This is the targets equivalent of
    `backend.lipsync._split_coords`, and it is deliberately the same shape:
    cut by MEASURED frame counts, and hand back nothing rather than
    something misaligned.

    `run_start` is re-marked on the first kept frame. The warp smoother's
    carry-over does not survive into a new process either, so the first frame
    of a shard is unsmoothed whether we say so or not -- saying so makes the
    render match what actually happens instead of damping toward a frame that
    was never drawn.

    The GEOMETRY is carried through untouched: slicing changes which frames
    are covered, never the resolution the coordinates were measured in.
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

    The vendored reader (`latentsync/utils/targets.py`) lives in a DIFFERENT
    conda env and cannot be imported from here, so the schema is necessarily
    known in two places. `tests/test_latentsync_targets.py` pins them to each
    other; change one and the test will tell you about the other.
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
