"""Backend-facing entry point for ASD + speaker↔face assignment.

Call order the pipeline follows:
  1. `score_asd(video, audio, tracks, model=..., reference_audio={...})`
  2. `assign_faces_to_turns(turns, tracks, asd_result)`
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from backend.asd import ASDResult, available_models, get_asd
from backend.faces.faces import FaceTracks

from .faces import (
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
    model: str = "loconet",
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
#: `backend/lipsync_targets.py` needs these to choose which face stage 9h
#: animates, and re-running ASD to get them is not an option mid-pipeline.
SCORES_SUFFIX = ".scores.npz"


def save_scores_npz(result: ASDResult, path: Path) -> Path:
    """Write one npz holding every track's frame indices and scores.

    Flat arrays plus an index, not one array per track — a key per track makes
    the file's schema depend on the video, leaving a reader to guess which
    keys are tracks and which are metadata.
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

    from backend.asd import ASDScore

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


#: Which model's scores win when a job holds several `*.scores.npz`, best
#: first. Consulted by `find_scores_npz`, because a bare glob sorts
#: alphabetically and would pick the wrong file.
ASD_MODEL_PREFERENCE = ("loconet", "lr_asd", "light_asd", "talknet")


def find_scores_npz(asd_dir, stem=None, *, prefer: Optional[str] = None):
    """The per-frame scores file to feed the lip-sync target builder.

    Picks by MODEL, not by filename sort: `prefer` first, then
    `ASD_MODEL_PREFERENCE`, then any remaining candidate. The one place that
    decides which model's ASD drives who-is-speaking, so every caller goes
    through it and they agree.
    """
    asd_dir = Path(asd_dir)
    if not asd_dir.is_dir():
        return None
    cands = sorted(asd_dir.glob(f"{stem}.*{SCORES_SUFFIX}")) if stem else []
    if not cands:
        cands = sorted(asd_dir.glob(f"*{SCORES_SUFFIX}"))
    if not cands:
        return None
    order = ([prefer] if prefer else []) + [
        m for m in ASD_MODEL_PREFERENCE if m != prefer]
    for m in order:
        for c in cands:
            if c.name.endswith(f".{m}{SCORES_SUFFIX}") or f".{m}." in c.name:
                return c
    return cands[0]


__all__ = [
    "ASDResult",
    "ASD_MODEL_PREFERENCE",
    "SCORES_SUFFIX",
    "SegmentAssignment",
    "assign_faces_to_turns",
    "assignments_to_jsonable",
    "find_scores_npz",
    "load_scores_npz",
    "save_scores_npz",
    "score_asd",
]
