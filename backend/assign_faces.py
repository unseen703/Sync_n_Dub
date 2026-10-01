"""Speaker ↔ face assignment via Hungarian matching over ASD scores.

Input:
  - Diarization turns (speaker, start_s, end_s).
  - FaceTracks (per-frame bboxes grouped into stable identities).
  - ASDResult (per-track, per-frame speaking scores).

Output per segment:
  { "face_track": "FACE_02" | None,
    "asd_score":  0.91,
    "on_screen":  True }

Why Hungarian: for an interval where two diarized turns overlap in time (or
near-overlap, e.g. interview back-and-forth) and both candidate faces are
on-screen, an independent per-turn argmax can hand the same face to both.
Hungarian solves the whole matrix in one shot and keeps assignments unique.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

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

    `fps` defaults to the FPS carried on `tracks`. `on_screen_threshold` is
    the ASD score below which we mark the segment off-screen rather than
    binding a face to it — narration must never be animated.

    `group_window_s` controls how turns are batched into Hungarian problems:
    turns whose time intervals come within this many seconds of another in
    the group are solved together. A plain argmax per isolated turn is fine;
    the Hungarian step only matters when turns time-overlap.
    """
    fps = fps or tracks.fps
    if not fps or fps <= 0:
        raise ValueError("Cannot assign without a positive FPS")

    # Group turns that could compete for the same face.
    groups = _group_turns(turns, window_s=group_window_s)
    track_ids = [t.track_id for t in tracks.tracks]

    assignments: Dict[str, SegmentAssignment] = {}
    for g_idx, group in enumerate(groups):
        _assign_group(
            group, g_idx, track_ids, asd, fps,
            on_screen_threshold, assignments,
        )

    # Preserve input turn order.
    ordered = []
    for i, turn in enumerate(turns):
        seg_id = f"SEG_{i:04d}"
        ordered.append(assignments[seg_id])
    log.info(
        "assign_faces: %d turns → %d on-screen (%d off-screen or no face)",
        len(ordered),
        sum(1 for a in ordered if a.on_screen),
        sum(1 for a in ordered if not a.on_screen),
    )
    return ordered


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _group_turns(turns, *, window_s: float) -> List[List[Tuple[int, object]]]:
    """Chain turns whose intervals touch (gap <= window_s) into one group.

    Returned as `[(turn_index, turn), ...]` to preserve the original index
    for seg_id generation.
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

    # Build (|turns| x |tracks|) score matrix. -inf for tracks absent in the
    # window, so Hungarian's maximisation never selects an impossible pair.
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

    # scipy minimises; negate the finite scores and keep -inf as +inf cost.
    cost = np.where(np.isfinite(scores), -scores, 1e9)
    # Hungarian requires a rectangular matrix; scipy handles non-square.
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
    # Any turns the solver could not pair (more turns than tracks) fall out
    # as off-screen.
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


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

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
