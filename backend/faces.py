"""Face detection + tracking entry point, plus speaker <-> face assignment.

Two concerns on the same `FaceTracks` object:

* The detect/track registries are hidden behind `detect_faces`, `track_faces`
  and `detect_and_track`, same way `backend.speaker_id.diarize` fronts the
  diarizer registry.
* `assign_faces_to_turns` and `speaker_face_tracks` turn diarized turns +
  face tracks + ASD scores into one `SegmentAssignment` per turn, by
  Hungarian matching -- see that function's docstring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.faces import FaceDetections, FaceTracks
from models.faces.detect import available_detectors, get_detector
from models.faces.track import available_trackers, get_tracker

log = logging.getLogger(__name__)


def preimport_face_stack() -> bool:
    """Load insightface BEFORE anything else claims the shared C libraries.

    An import-order trap `LD_PRELOAD` does not cover: `pyannote.audio` pulls in
    a libjpeg without the 12-bit symbols cv2's bundled `libtiff.so.6` wants,
    and whichever loads first wins for the life of the process. Importing in
    this order fixes it. The failure surfaces at stage 5 as "RetinaFace needs
    either `insightface` ... or `retina-face`", which points at a missing
    package that is in fact installed.

    Returns whether the import succeeded, and never raises — a host genuinely
    without insightface must still reach the adapter's own error.
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


# ---------------------------------------------------------------------------
# Speaker <-> face assignment (formerly backend/assign_faces.py)
#
# Input:
#   - Diarization turns (speaker, start_s, end_s).
#   - FaceTracks (per-frame bboxes grouped into stable identities).
#   - ASDResult (per-track, per-frame speaking scores).
#
# Output per segment:
#   { "face_track": "FACE_02" | None,
#     "asd_score":  0.91,
#     "on_screen":  True }
#
# Hungarian rather than per-turn argmax: when two turns overlap in time and
# both candidate faces are on screen, an independent argmax can hand the same
# face to both. Solving the whole matrix at once keeps assignments unique.
# ---------------------------------------------------------------------------


@dataclass
class SegmentAssignment:
    """One decision per diarized turn."""
    seg_id: str
    speaker: str
    start_s: float
    end_s: float
    face_track: Optional[str]
    asd_score: float          # -inf when no track covered the window
    on_screen: bool
    extra: dict = field(default_factory=dict)


def assign_faces_to_turns(
    turns,                            # List[models.diarization.base.Turn]
    tracks,                           # FaceTracks
    asd,                              # ASDResult
    *,
    fps: Optional[float] = None,
    on_screen_threshold: float = 0.3,
    group_window_s: float = 0.5,
) -> List[SegmentAssignment]:
    """Resolve one `SegmentAssignment` per turn.

    `fps` defaults to the FPS on `tracks`. Below `on_screen_threshold` a
    segment is marked off-screen rather than bound to a face, so narration is
    never animated. `group_window_s` is how close in time turns must be to be
    solved in the same Hungarian problem.
    """
    fps = fps or tracks.fps
    if not fps or fps <= 0:
        raise ValueError("Cannot assign without a positive FPS")

    groups = _group_turns(turns, window_s=group_window_s)
    track_ids = [t.track_id for t in tracks.tracks]

    assignments: Dict[str, SegmentAssignment] = {}
    for g_idx, group in enumerate(groups):
        _assign_group(
            group, g_idx, track_ids, asd, fps,
            on_screen_threshold, assignments,
        )

    ordered = []
    for i, turn in enumerate(turns):
        seg_id = f"SEG_{i:04d}"
        ordered.append(assignments[seg_id])
    log.info(
        "assign_faces: %d turns -> %d on-screen (%d off-screen or no face)",
        len(ordered),
        sum(1 for a in ordered if a.on_screen),
        sum(1 for a in ordered if not a.on_screen),
    )
    return ordered


def _group_turns(turns, *, window_s: float) -> List[List[Tuple[int, object]]]:
    """Chain turns whose intervals touch (gap <= window_s) into one group.

    Returns `[(turn_index, turn), ...]`, keeping the original index for seg_id
    generation.
    """
    indexed = sorted(enumerate(turns), key=lambda it: it[1].start)
    groups: List[List[Tuple[int, object]]] = []
    cur: List[Tuple[int, object]] = []
    cur_end: float = -1.0
    for i, t in indexed:
        if cur and t.start - cur_end > window_s:
            groups.append(cur)
            cur = []
            cur_end = -1.0
        cur.append((i, t))
        cur_end = max(cur_end, t.end)
    if cur:
        groups.append(cur)
    return groups


def _assign_group(
    group, g_idx, track_ids, asd, fps, on_screen_threshold, out,
):
    from scipy.optimize import linear_sum_assignment  # defer heavy import

    n_turns, n_tracks = len(group), len(track_ids)
    if n_tracks == 0:
        for i, turn in group:
            out[f"SEG_{i:04d}"] = SegmentAssignment(
                seg_id=f"SEG_{i:04d}",
                speaker=turn.speaker,
                start_s=turn.start, end_s=turn.end,
                face_track=None, asd_score=float("-inf"),
                on_screen=False,
                extra={"reason": "no face tracks in video"},
            )
        return

    scores = np.full((n_turns, n_tracks), -np.inf, dtype=np.float32)
    for ti, (_, turn) in enumerate(group):
        s_frame = int(round(turn.start * fps))
        e_frame = int(round(turn.end * fps))
        for tj, trk_id in enumerate(track_ids):
            s = asd.mean_score_over_window(trk_id, s_frame, e_frame)
            scores[ti, tj] = s

    cost = np.where(np.isfinite(scores), -scores, 1e9)
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_cols = set()
    for r, c in zip(row_ind, col_ind):
        assigned_cols.add(c)
        turn_i, turn = group[r]
        trk_id = track_ids[c]
        score = float(scores[r, c])
        on_screen = np.isfinite(score) and score >= on_screen_threshold
        out[f"SEG_{turn_i:04d}"] = SegmentAssignment(
            seg_id=f"SEG_{turn_i:04d}",
            speaker=turn.speaker,
            start_s=turn.start, end_s=turn.end,
            face_track=trk_id if on_screen else None,
            asd_score=score if np.isfinite(score) else float("-inf"),
            on_screen=bool(on_screen),
            extra={"group_idx": g_idx},
        )
    paired_rows = set(row_ind.tolist())
    for r, (turn_i, turn) in enumerate(group):
        if r in paired_rows:
            continue
        out[f"SEG_{turn_i:04d}"] = SegmentAssignment(
            seg_id=f"SEG_{turn_i:04d}",
            speaker=turn.speaker,
            start_s=turn.start, end_s=turn.end,
            face_track=None,
            asd_score=float("-inf"),
            on_screen=False,
            extra={"group_idx": g_idx, "reason": "unpaired by Hungarian"},
        )


def assignments_to_jsonable(assignments: List[SegmentAssignment]) -> list:
    """Round-trippable through json.dumps. -inf becomes null."""
    out = []
    for a in assignments:
        score = a.asd_score if np.isfinite(a.asd_score) else None
        out.append({
            "seg_id": a.seg_id,
            "speaker": a.speaker,
            "start_s": a.start_s,
            "end_s": a.end_s,
            "face_track": a.face_track,
            "asd_score": score,
            "on_screen": a.on_screen,
            "extra": a.extra,
        })
    return out


def speaker_face_tracks(
    assignments,                      # List[SegmentAssignment] | list[dict]
    tracks,                           # FaceTracks
    *,
    min_on_screen_s: float = 1.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return `{speaker: (frames, bboxes)}` for the face each speaker owns.

    One speaker can land on several tracks (a cut to a reverse angle starts a
    new one), so the winner is the track with the most on-screen SECONDS, not
    the most turns -- otherwise reaction shots outvote a long close-up.
    Off-screen turns are skipped; they are narration or off-camera lines where
    the face in frame belongs to someone else.

    A speaker under `min_on_screen_s` is left out rather than guessed at, which
    reads as "no face evidence" instead of "no faces found".
    """
    by_id = tracks.by_id() if callable(getattr(tracks, "by_id", None)) \
        else {t.track_id: t for t in tracks.tracks}

    seconds: Dict[str, Dict[str, float]] = {}
    for a in assignments:
        if isinstance(a, dict):
            spk, trk = a.get("speaker"), a.get("face_track")
            on, t0, t1 = a.get("on_screen"), a.get("start_s"), a.get("end_s")
        else:
            spk, trk, on, t0, t1 = (a.speaker, a.face_track, a.on_screen,
                                    a.start_s, a.end_s)
        if not spk or not trk or not on:
            continue
        seconds.setdefault(spk, {})
        seconds[spk][trk] = seconds[spk].get(trk, 0.0) + max(0.0, (t1 or 0) - (t0 or 0))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spk, per_track in seconds.items():
        trk, secs = max(per_track.items(), key=lambda kv: kv[1])
        if secs < min_on_screen_s or trk not in by_id:
            log.info("faces: %s has %.1f s on screen on %s; no face evidence",
                     spk, secs, trk)
            continue
        t = by_id[trk]
        out[spk] = (np.asarray(t.frames), np.asarray(t.bboxes))
        log.info("faces: %s -> %s (%.1f s on screen, %d frames)",
                 spk, trk, secs, len(t.frames))
    return out
