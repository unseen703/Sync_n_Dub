"""ByteTrack tracker — wraps `supervision`'s pure-Python ByteTrack.

The reference ByteTrack impl ships with YOLOX; `supervision.ByteTrack` is the
same algorithm (Kalman motion model + IoU association over detections,
including low-score boxes) without the YOLOX/mmcv dependency chain. Suitable
for face boxes as long as we treat every detection as class 0 — ByteTrack is
class-agnostic internally.

Install with: `pip install supervision>=0.21`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from ..base import (
    BaseFaceTracker,
    FaceDetections,
    FaceTracks,
    tracks_from_detections_array,
)

log = logging.getLogger(__name__)


class ByteTrackTracker(BaseFaceTracker):
    name = "bytetrack"

    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 25,
        minimum_consecutive_frames: int = 1,
    ):
        super().__init__()
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        self.minimum_consecutive_frames = minimum_consecutive_frames

    def _new_tracker(self, fps: float):
        import supervision as sv
        return sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=int(round(fps)) or self.frame_rate,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
        )

    def track(self, detections: FaceDetections) -> FaceTracks:
        try:
            import supervision as sv
        except ImportError as e:
            raise RuntimeError(
                "ByteTrack adapter requires `pip install supervision>=0.21`."
            ) from e

        tracker = self._new_tracker(detections.fps)

        # ByteTrack is stateful; feed frames in strict order even if some are
        # empty, so the Kalman step advances correctly between hits.
        per_frame: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
        last_frame = max(detections.detections.keys(), default=-1)
        if detections.num_frames:
            last_frame = max(last_frame, detections.num_frames - 1)

        for frame_idx in range(last_frame + 1):
            frame_boxes = detections.detections.get(frame_idx, [])
            if frame_boxes:
                xyxy = np.array([b.as_xyxy() for b in frame_boxes], dtype=np.float32)
                conf = np.array([b.score for b in frame_boxes], dtype=np.float32)
                cls = np.zeros((len(frame_boxes),), dtype=int)
                sv_det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
            else:
                sv_det = sv.Detections.empty()

            tracked = tracker.update_with_detections(sv_det)
            if tracked.tracker_id is None or len(tracked) == 0:
                continue
            rows = []
            for i, tid in enumerate(tracked.tracker_id):
                if tid is None:
                    continue
                conf_i = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                rows.append((int(tid), tracked.xyxy[i].astype(np.float32), conf_i))
            if rows:
                per_frame[frame_idx] = rows

        tracks = tracks_from_detections_array(
            per_frame, detections=detections, tracker_name=self.name,
        )
        log.info(
            "bytetrack: %d tracks across %d frames (detector=%s, fps=%.2f)",
            len(tracks.tracks), detections.num_frames, detections.detector, detections.fps,
        )
        return tracks
