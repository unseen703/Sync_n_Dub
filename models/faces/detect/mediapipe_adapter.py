"""MediaPipe face detector: fast, CPU-only, bbox-oriented.

`model_selection=1` ("full range") handles faces a few metres from the camera,
which suits lecture and interview footage.

MediaPipe emits SIX landmarks (two eyes, nose, mouth centre, two ear tragions)
and they are projected onto the 5-point convention the rest of the stack
expects, with both mouth corners duplicated from the mouth centre. That is
fine for IoU tracking but is not a real localisation — use RetinaFace where
landmarks matter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..base import BaseFaceDetector, FaceBox, FaceDetections

log = logging.getLogger(__name__)


class MediaPipeDetector(BaseFaceDetector):
    name = "mediapipe"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        model_selection: int = 1,
        min_detection_confidence: float = 0.5,
    ):
        super().__init__(device=device)
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self._mp = None
        self._detector = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "mediapipe is required for the MediaPipe detector. "
                "`pip install mediapipe`."
            ) from e
        self._mp = mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence,
        )
        self._loaded = True
        log.info(
            "mediapipe: loaded model_selection=%d min_conf=%.2f (CPU — device=%s ignored)",
            self.model_selection, self.min_detection_confidence, self.device,
        )

    def detect(
        self,
        video_path: Path,
        *,
        every_n: int = 1,
        min_score: float = 0.5,
    ) -> FaceDetections:
        self._ensure_loaded()
        video_path = Path(video_path)

        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "OpenCV is required for video decoding. `pip install opencv-python`."
            ) from e

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        detections: dict[int, List[FaceBox]] = {}
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame_idx % every_n) == 0:
                    boxes = self._detect_frame(frame, frame_idx, min_score, width, height)
                    if boxes:
                        detections[frame_idx] = boxes
                frame_idx += 1
        finally:
            cap.release()

        log.info(
            "mediapipe: %d frames scanned, %d with faces, %d total boxes",
            frame_idx, len(detections), sum(len(v) for v in detections.values()),
        )

        return FaceDetections(
            video_path=video_path,
            fps=fps,
            width=width,
            height=height,
            num_frames=total or frame_idx,
            detector=self.name,
            detections=detections,
        )

    def _detect_frame(
        self,
        bgr: np.ndarray,
        frame_idx: int,
        min_score: float,
        width: int,
        height: int,
    ) -> List[FaceBox]:
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._detector.process(rgb)
        if not result.detections:
            return []

        out: List[FaceBox] = []
        for det in result.detections:
            score = float(det.score[0]) if det.score else 0.0
            if score < min_score:
                continue
            rel = det.location_data.relative_bounding_box
            # Clip to frame: MediaPipe returns boxes past the edge on partial
            # faces.
            x1 = max(0.0, rel.xmin) * width
            y1 = max(0.0, rel.ymin) * height
            x2 = min(1.0, rel.xmin + rel.width) * width
            y2 = min(1.0, rel.ymin + rel.height) * height
            if x2 <= x1 or y2 <= y1:
                continue

            lm = None
            kps = det.location_data.relative_keypoints
            if kps:
                def pt(i):
                    p = kps[i]
                    return (p.x * width, p.y * height)
                # MediaPipe order: 0=right eye, 1=left eye, 2=nose, 3=mouth,
                # 4=right ear tragion, 5=left ear tragion.
                re = pt(0); le = pt(1); nose = pt(2); mouth = pt(3)
                lm = np.array([re, le, nose, mouth, mouth], dtype=np.float32)

            out.append(FaceBox(
                frame_idx=frame_idx,
                x1=x1, y1=y1, x2=x2, y2=y2,
                score=score, landmarks=lm,
            ))
        return out
