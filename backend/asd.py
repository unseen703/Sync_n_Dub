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


__all__ = [
    "ASDResult",
    "SegmentAssignment",
    "assign_faces_to_turns",
    "assignments_to_jsonable",
    "score_asd",
]
