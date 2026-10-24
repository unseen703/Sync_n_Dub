"""Backend-facing entry point for ASD + speaker↔face assignment.

Call order the pipeline follows:
  1. `score_asd(video, audio, tracks, model=..., reference_audio={...})`
  2. `assign_faces_to_turns(turns, tracks, asd_result)`
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from models.asd import ASDResult, available_models, get_asd
from models.faces import FaceTracks

from .assign_faces import (
    SegmentAssignment,
    assign_faces_to_turns,
    assignments_to_jsonable,
)

log = logging.getLogger(__name__)


def score_asd(
    video: Path,
    audio: Path,
    tracks: FaceTracks,
    *,
    model: str = "talknet",
    device: Optional[str] = None,
    reference_audio: Optional[Dict[str, Path]] = None,
    **model_kwargs,
) -> ASDResult:
    video = Path(video); audio = Path(audio)
    if not video.exists():
        raise FileNotFoundError(video)
    if not audio.exists():
        raise FileNotFoundError(audio)
    if model not in available_models():
        raise ValueError(
            f"Unknown ASD model {model!r}. Available: {', '.join(available_models())}"
        )
    log.info(
        "asd: %s on %d tracks (model=%s, device=%s)",
        video.name, len(tracks.tracks), model, device or "auto",
    )
    m = get_asd(model, device=device, **model_kwargs)
    result = m.score(video, audio, tracks, reference_audio=reference_audio)
    log.info("asd: scored %d tracks", len(result.per_track))
    return result


#: Per-frame scores, beside the per-segment `assignments.json`.
#:
#: Stage 6 has always computed these and then thrown them away, persisting
#: only the per-SEGMENT summary. That cost twice over: `ui/bbox_video.py`
#: went looking for a `scores.npz` that was never written (so every box in
#: the review video came out grey), and `backend/lipsync_targets.py` needs
#: per-frame scores to decide which face stage 9h should animate — with
#: nothing on disk, the only way to get them was to re-run TalkNet, which
#: no caller was ever going to do mid-pipeline.
SCORES_SUFFIX = ".scores.npz"


def save_scores_npz(result: ASDResult, path: Path) -> Path:
    """Write one npz holding every track's frame indices and scores.

    Flat arrays plus an index, not one array per track: `np.savez` with a
    key per track makes the file's schema depend on the video, and a reader
    then has to guess which keys are tracks and which are metadata.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, frames, scores, offsets = [], [], [], [0]
    for tid, sc in sorted(result.per_track.items()):
        ids.append(str(tid))
        frames.append(np.asarray(sc.frames, dtype=np.int32))
        scores.append(np.asarray(sc.scores, dtype=np.float32))
        offsets.append(offsets[-1] + len(sc.frames))
    np.savez_compressed(
        path,
        track_ids=np.asarray(ids, dtype=object),
        frames=(np.concatenate(frames) if frames
                else np.zeros(0, dtype=np.int32)),
        scores=(np.concatenate(scores) if scores
                else np.zeros(0, dtype=np.float32)),
        offsets=np.asarray(offsets, dtype=np.int64),
        fps=np.float32(result.fps or 0.0),
        model_name=str(result.model_name or ""),
        video_path=str(result.video_path or ""),
    )
    log.info("asd: wrote per-frame scores for %d track(s) -> %s",
             len(ids), path)
    return path


def load_scores_npz(path: Path) -> ASDResult:
    """Read back what `save_scores_npz` wrote."""
    import numpy as np

    from models.asd import ASDScore

    with np.load(Path(path), allow_pickle=True) as z:
        ids = [str(t) for t in z["track_ids"].tolist()]
        frames, scores, offs = z["frames"], z["scores"], z["offsets"]
        per_track = {
            tid: ASDScore(track_id=tid,
                          frames=np.asarray(frames[offs[i]:offs[i + 1]],
                                            dtype=np.int32),
                          scores=np.asarray(scores[offs[i]:offs[i + 1]],
                                            dtype=np.float32))
            for i, tid in enumerate(ids)
        }
        return ASDResult(video_path=Path(str(z["video_path"])),
                         model_name=str(z["model_name"]),
                         fps=float(z["fps"]),
                         per_track=per_track)


__all__ = [
    "ASDResult",
    "SCORES_SUFFIX",
    "SegmentAssignment",
    "assign_faces_to_turns",
    "assignments_to_jsonable",
    "load_scores_npz",
    "save_scores_npz",
    "score_asd",
]
