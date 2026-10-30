"""Shared types for active speaker detection.

An ASD model takes the video, the audio and a `FaceTracks`, and scores per
track per frame whether that face is speaking. Scores are bounded to [0, 1] —
normalising is the adapter's job, so the Hungarian assignment downstream has
one metric to compare on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ASDScore:
    """Per-face-track, per-frame ASD scores for a single video.

    `frames` and `scores` are parallel and sorted by frame. Frames outside the
    track are omitted rather than zero-filled, so a consumer must align
    explicitly.
    """
    track_id: str
    frames: np.ndarray            # (N,) int32
    scores: np.ndarray            # (N,) float32 in [0, 1]


@dataclass
class ASDResult:
    """ASD output for one video."""
    video_path: Path
    model_name: str
    fps: float
    per_track: Dict[str, ASDScore] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def mean_score_over_window(
        self,
        track_id: str,
        start_frame: int,
        end_frame: int,
    ) -> float:
        """Average ASD score for `track_id` on frames [start_frame, end_frame].

        Returns -inf when the track did not appear in that interval, which the
        Hungarian solver reads as "cannot assign". Returning 0 would rank an
        absent face above a quiet but present one.
        """
        score = self.per_track.get(track_id)
        if score is None or score.frames.size == 0:
            return float("-inf")
        mask = (score.frames >= start_frame) & (score.frames <= end_frame)
        if not np.any(mask):
            return float("-inf")
        return float(score.scores[mask].mean())


class BaseASD(ABC):
    """Every ASD adapter subclasses this."""

    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from models.diarization.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def score(
        self,
        video_path: Path,
        audio_path: Path,
        tracks,                          # FaceTracks — not imported to avoid cycle
        *,
        reference_audio: Optional[Dict[str, Path]] = None,
    ) -> ASDResult:
        """Score every face track against the audio.

        `reference_audio` maps `{speaker_id: wav_path}` and is read only by
        target-speaker-aware adapters; passing it to one that cannot use it
        simply changes nothing.
        """
