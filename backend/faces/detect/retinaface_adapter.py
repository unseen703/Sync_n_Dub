"""RetinaFace detector, via InsightFace's SCRFD or the `retina-face` package.

InsightFace's `buffalo_l` pack (SCRFD-10GF plus a 5-point landmark head) is
preferred; the pure-TF `retina-face` is the fallback when it is unavailable.
Weights download on the first `detect()` and run ~250 MB on disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..base import BaseFaceDetector, FaceBox, FaceDetections

log = logging.getLogger(__name__)


class RetinaFaceDetector(BaseFaceDetector):
    name = "retinaface"
    default_pack = "buffalo_l"      # InsightFace preset shipping SCRFD-10GF
    default_input_size = (640, 640)

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        pack: Optional[str] = None,
        input_size: Optional[tuple[int, int]] = None,
        det_thresh: float = 0.5,
    ):
        super().__init__(device=device)
        self.pack = pack or self.default_pack
        self.input_size = input_size or self.default_input_size
        self.det_thresh = det_thresh
        self._impl: Optional[str] = None     # "insightface" | "retinaface_tf"
        self._model = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from insightface.app import FaceAnalysis

            # onnxruntime lists CUDAExecutionProvider even when it cannot
            # dlopen the CUDA runtime, then silently falls back to CPU. This
            # makes the EP genuinely loadable.
            if self.device.startswith("cuda"):
                from backend.common.gpu import ensure_onnxruntime_cuda
                if not ensure_onnxruntime_cuda():
                    log.warning(
                        "retinaface: CUDA requested but onnxruntime cannot "
                        "provide it; detection will run on CPU."
                    )

            provider = (
                "CUDAExecutionProvider"
                if self.device.startswith("cuda")
                else "CPUExecutionProvider"
            )
            app = FaceAnalysis(
                name=self.pack,
                allowed_modules=("detection",),
                providers=[provider],
            )
            # ctx_id: 0+ = GPU index, -1 = CPU.
            ctx_id = 0 if self.device.startswith("cuda") else -1
            app.prepare(ctx_id=ctx_id, det_size=self.input_size, det_thresh=self.det_thresh)
            self._model = app
            self._impl = "insightface"
            self._loaded = True
            log.info("retinaface: loaded insightface pack=%s on %s", self.pack, self.device)
            return
        except ImportError as e:
            # Say WHICH import failed: a bare "insightface unavailable" reads
            # as "not installed", but the interesting case is an ImportError
            # from inside insightface's own lazy imports once torch/cv2/
            # onnxruntime are loaded — the import-order trap in CLAUDE.md §1.
            log.warning("retinaface: insightface did not load (%s: %s); "
                        "trying the retina-face package",
                        type(e).__name__, e)
            log.debug("retinaface: insightface import traceback", exc_info=True)

        try:
            from retinaface import RetinaFace    # type: ignore
            self._model = RetinaFace
            self._impl = "retinaface_tf"
            self._loaded = True
            log.info("retinaface: loaded retina-face (TF) package")
        except ImportError as e:
            raise RuntimeError(
                "RetinaFace needs either `insightface` (preferred: "
                "`pip install insightface onnxruntime-gpu`) or the "
                "`retina-face` TF package."
            ) from e

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

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
                    boxes = self._detect_frame(frame, frame_idx, min_score)
                    if boxes:
                        detections[frame_idx] = boxes
                frame_idx += 1
        finally:
            cap.release()

        log.info(
            "retinaface: %d frames scanned, %d frames with faces, %d boxes total",
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

    # ------------------------------------------------------------------
    # per-frame
    # ------------------------------------------------------------------

    def _detect_frame(self, bgr: np.ndarray, frame_idx: int, min_score: float) -> List[FaceBox]:
        if self._impl == "insightface":
            # InsightFace expects BGR, as OpenCV reads.
            faces = self._model.get(bgr)
            out: List[FaceBox] = []
            for f in faces:
                if float(f.det_score) < min_score:
                    continue
                x1, y1, x2, y2 = [float(v) for v in f.bbox]
                lm = np.asarray(f.kps, dtype=np.float32) if f.kps is not None else None
                out.append(FaceBox(
                    frame_idx=frame_idx,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=float(f.det_score),
                    landmarks=lm,
                ))
            return out

        # retina-face (TF) path
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._model.detect_faces(rgb, threshold=min_score)
        if not isinstance(result, dict):
            return []
        out = []
        for face in result.values():
            score = float(face.get("score", 0.0))
            if score < min_score:
                continue
            x1, y1, x2, y2 = [float(v) for v in face["facial_area"]]
            lm_dict = face.get("landmarks", {}) or {}
            lm = np.array(
                [lm_dict.get(k, (0.0, 0.0)) for k in
                 ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")],
                dtype=np.float32,
            ) if lm_dict else None
            out.append(FaceBox(
                frame_idx=frame_idx,
                x1=x1, y1=y1, x2=x2, y2=y2,
                score=score, landmarks=lm,
            ))
        return out
