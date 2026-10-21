"""Shared types + abstract bases for the face stack.

Two interfaces:
  - `BaseFaceDetector`: video → per-frame face boxes.
  - `BaseFaceTracker` : per-frame boxes → stable track identities over time.

The data classes are what every downstream stage (ASD, speaker↔face
assignment, lip-sync) reads. They are JSON-serializable through plain dicts
and `numpy.ndarray` → list so the manifest can carry them without pickling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FaceBox:
    """A single face detection, in a single frame.

    Coordinates are pixel-space of the ORIGINAL frame (not any letterboxed
    or resized tensor the detector consumed). `score` is the detector's own
    confidence, kept raw so ByteTrack's association step can read it.
    """
    frame_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    # 5-point landmarks (eyes, nose, mouth corners) when the detector provides
    # them. None for detectors that do not (e.g. a bare bbox-only mode).
    landmarks: Optional[np.ndarray] = None   # shape (5, 2), xy

    @property
    def wh(self) -> Tuple[float, float]:
        return (max(0.0, self.x2 - self.x1), max(0.0, self.y2 - self.y1))

    @property
    def area(self) -> float:
        w, h = self.wh
        return w * h

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class FaceDetections:
    """Per-frame detections for one video."""
    video_path: Path
    fps: float
    width: int
    height: int
    num_frames: int
    detector: str
    detections: Dict[int, List[FaceBox]] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class FaceTrack:
    """A stable identity across a span of frames.

    `frames` and `bboxes` are parallel: `bboxes[i]` is the box at frame
    `frames[i]`. `score[i]` is the detector score for that frame; the tracker
    carries it through so later stages (ASD, assignment) can weight by it.

    Missing frames (brief occlusions the tracker interpolated through) are
    represented as gaps in `frames`, not by zero rows — a consumer that wants
    a dense per-frame signal should resample.
    """
    track_id: str
    frames: np.ndarray            # shape (N,) int, strictly increasing
    bboxes: np.ndarray            # shape (N, 4) float xyxy
    scores: np.ndarray            # shape (N,) float
    # Diagnostics that the lip-sync stage reads to decide whether to skip:
    mean_pose_yaw: Optional[float] = None
    occlusion_ratio: Optional[float] = None

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def frame_range(self) -> Tuple[int, int]:
        if self.num_frames == 0:
            return (-1, -1)
        return (int(self.frames[0]), int(self.frames[-1]))

    def mean_bbox(self) -> np.ndarray:
        """(4,) float — mean of xyxy. Useful as a track summary row."""
        if self.num_frames == 0:
            return np.zeros(4, dtype=np.float32)
        return self.bboxes.mean(axis=0)

    def to_jsonable(self) -> dict:
        return {
            "track_id": self.track_id,
            "frame_range": list(self.frame_range),
            "num_frames": self.num_frames,
            "mean_bbox_xyxy": self.mean_bbox().tolist(),
            "mean_pose_yaw": self.mean_pose_yaw,
            "occlusion_ratio": self.occlusion_ratio,
        }


@dataclass
class FaceTracks:
    """All tracks for one video, with a stable id scheme (`FACE_01`, …)."""
    video_path: Path
    fps: float
    width: int
    height: int
    num_frames: int
    detector: str
    tracker: str
    tracks: List[FaceTrack] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def by_id(self) -> Dict[str, FaceTrack]:
        return {t.track_id: t for t in self.tracks}

    def save_npz(self, path: Path) -> Path:
        """Persist a compact binary form. The manifest keeps the summary; this
        file holds the raw per-frame boxes a downstream stage might rescore."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "video_path": np.array(str(self.video_path)),
            "fps": np.array(self.fps),
            "width": np.array(self.width),
            "height": np.array(self.height),
            "num_frames": np.array(self.num_frames),
            "detector": np.array(self.detector),
            "tracker": np.array(self.tracker),
            "track_ids": np.array([t.track_id for t in self.tracks]),
        }
        for t in self.tracks:
            payload[f"{t.track_id}.frames"] = t.frames
            payload[f"{t.track_id}.bboxes"] = t.bboxes
            payload[f"{t.track_id}.scores"] = t.scores
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path: Path) -> "FaceTracks":
        """Read back what `save_npz` wrote.

        The writer shipped without a reader, so every consumer that wanted the
        per-frame boxes had to re-derive this key scheme by hand -- which is
        how `save_npz`'s own promise ("a downstream stage might rescore") went
        unkept. Keep the two together.
        """
        path = Path(path)
        d = np.load(path, allow_pickle=True)
        tracks = []
        for tid in (str(x) for x in d["track_ids"]):
            tracks.append(FaceTrack(
                track_id=tid,
                frames=d[f"{tid}.frames"],
                bboxes=d[f"{tid}.bboxes"],
                scores=d[f"{tid}.scores"],
            ))
        return cls(
            video_path=Path(str(d["video_path"])),
            fps=float(d["fps"]), width=int(d["width"]), height=int(d["height"]),
            num_frames=int(d["num_frames"]),
            detector=str(d["detector"]), tracker=str(d["tracker"]),
            tracks=tracks,
        )


# ---------------------------------------------------------------------------
# bases
# ---------------------------------------------------------------------------

class BaseFaceDetector(ABC):
    """Takes a video, returns per-frame face boxes.

    Implementations must be safe to construct without weights — register load
    in `_ensure_loaded()` and call it from `detect()`. The CLI enumerates
    adapters at startup; a greedy constructor would download gigabytes before
    the user picked one.
    """

    name: str = "base"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from models.diarization.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def detect(
        self,
        video_path: Path,
        *,
        every_n: int = 1,
        min_score: float = 0.5,
    ) -> FaceDetections:
        """Run detection over the video.

        Args:
          video_path: any OpenCV/FFmpeg-readable video.
          every_n: skip rate; `>1` detects on every nth frame and the tracker
            interpolates. Lecture footage with slow motion tolerates `every_n=2`.
          min_score: detector-confidence floor. ByteTrack handles low-score
            boxes itself, so keep this permissive (0.3–0.5) at detect time.
        """


class BaseFaceTracker(ABC):
    """Associates per-frame boxes into stable track identities.

    The input is one video's `FaceDetections`; the output is `FaceTracks`
    with canonical ids `FACE_01`, `FACE_02`, … in first-appearance order.
    """

    name: str = "base"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def track(self, detections: FaceDetections) -> FaceTracks: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tracks_from_detections_array(
    per_frame_tracks: Dict[int, List[Tuple[int, np.ndarray, float]]],
    *,
    detections: FaceDetections,
    tracker_name: str,
) -> FaceTracks:
    """Fold a tracker's per-frame output into `FaceTrack` objects.

    `per_frame_tracks[frame_idx]` is a list of `(raw_track_id, xyxy, score)`.
    We rewrite raw ids to `FACE_01`, `FACE_02`, … in first-appearance order so
    downstream manifests are stable across tracker swaps.
    """
    id_map: Dict[int, str] = {}
    buckets: Dict[str, List[Tuple[int, np.ndarray, float]]] = {}
    for frame_idx in sorted(per_frame_tracks):
        for raw_id, xyxy, score in per_frame_tracks[frame_idx]:
            if raw_id not in id_map:
                id_map[raw_id] = f"FACE_{len(id_map) + 1:02d}"
            canon = id_map[raw_id]
            buckets.setdefault(canon, []).append((frame_idx, xyxy, score))

    tracks: List[FaceTrack] = []
    for canon_id, rows in buckets.items():
        rows.sort(key=lambda r: r[0])
        frames = np.array([r[0] for r in rows], dtype=np.int32)
        bboxes = np.stack([r[1] for r in rows]).astype(np.float32)
        scores = np.array([r[2] for r in rows], dtype=np.float32)
        tracks.append(FaceTrack(
            track_id=canon_id,
            frames=frames,
            bboxes=bboxes,
            scores=scores,
        ))

    return FaceTracks(
        video_path=detections.video_path,
        fps=detections.fps,
        width=detections.width,
        height=detections.height,
        num_frames=detections.num_frames,
        detector=detections.detector,
        tracker=tracker_name,
        tracks=tracks,
    )
