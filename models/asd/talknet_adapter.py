"""TalkNet-ASD adapter.

Wraps the TaoRuijie/TalkNet-ASD repo, which is not pip-installable: clone it,
set `TALKNET_REPO`, and this adapter imports its `talkNet` model and
`demoTalkNet.py`-style preprocessing.

The pipeline per face track:
  1. Crop a 112x112 face patch around the track's bbox per frame (25 fps
     target — TalkNet trained on AVA-ActiveSpeaker at 25).
  2. Extract 13-d MFCC from the audio aligned to the same frames.
  3. Run the audio-visual encoder → per-frame speaking logits → sigmoid.
  4. Pack into an `ASDScore` per track.

If your repo clone exposes a different symbol, override the module/class via
`TALKNET_MODEL_MODULE` / `TALKNET_MODEL_CLASS` env vars rather than editing
this file — the TalkNet fork landscape is varied.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..faces.base import FaceTrack, FaceTracks
from .base import ASDResult, ASDScore, BaseASD

log = logging.getLogger(__name__)


class TalkNetASD(BaseASD):
    name = "talknet"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[Path] = None,
        checkpoint: Optional[Path] = None,
        crop_size: int = 112,
        target_fps: float = 25.0,
    ):
        super().__init__(device=device)
        env_repo = os.getenv("TALKNET_REPO")
        self.repo_path = Path(repo_path or env_repo or "").expanduser() if (repo_path or env_repo) else None
        env_ckpt = os.getenv("TALKNET_CHECKPOINT")
        self.checkpoint = Path(checkpoint or env_ckpt).expanduser() if (checkpoint or env_ckpt) else None
        self.crop_size = crop_size
        self.target_fps = target_fps
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.repo_path is None or not self.repo_path.exists():
            raise RuntimeError(
                "TalkNet repo not found. Set TALKNET_REPO to the clone of "
                "https://github.com/TaoRuijie/TalkNet-ASD."
            )
        if self.checkpoint is None or not self.checkpoint.exists():
            raise RuntimeError(
                "TalkNet checkpoint not found. Download pretrain_TalkSet.model "
                "from the TalkNet README and set TALKNET_CHECKPOINT."
            )
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        model_module = os.getenv("TALKNET_MODEL_MODULE", "talkNet")
        model_class = os.getenv("TALKNET_MODEL_CLASS", "talkNet")
        import importlib
        import torch
        mod = importlib.import_module(model_module)
        TalkNet = getattr(mod, model_class)
        model = TalkNet()
        # The upstream checkpoint is saved as a plain state_dict with the
        # `model.` prefix. Repo's own loader strips it; we do the same.
        state = torch.load(str(self.checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        clean = {k.removeprefix("model."): v for k, v in state.items()}
        try:
            model.load_state_dict(clean, strict=True)
        except Exception:
            model.load_state_dict(state, strict=False)
        model.eval()
        if self.device.startswith("cuda"):
            model = model.cuda()
        self._model = model
        self._loaded = True
        log.info("talknet: loaded from %s", self.checkpoint)

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score(
        self,
        video_path: Path,
        audio_path: Path,
        tracks: FaceTracks,
        *,
        reference_audio: Optional[Dict[str, Path]] = None,
    ) -> ASDResult:
        self._ensure_loaded()

        import torch
        import cv2
        import soundfile as sf
        from python_speech_features import mfcc as mfcc_fn  # upstream dep

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(
                f"TalkNet expects 16 kHz audio; got {sr}. Resample upstream."
            )

        per_track: Dict[str, ASDScore] = {}
        for tr in tracks.tracks:
            per_track[tr.track_id] = self._score_track(
                tr, video_path, audio, tracks.fps, mfcc_fn, cv2, torch,
            )

        return ASDResult(
            video_path=video_path,
            model_name=self.name,
            fps=tracks.fps,
            per_track=per_track,
        )

    def _score_track(
        self,
        track: FaceTrack,
        video_path: Path,
        audio: np.ndarray,
        fps: float,
        mfcc_fn,
        cv2,
        torch,
    ) -> ASDScore:
        if track.num_frames == 0:
            return ASDScore(track.track_id,
                            frames=np.zeros((0,), dtype=np.int32),
                            scores=np.zeros((0,), dtype=np.float32))

        # Crop faces per frame.
        crops = _crop_track(track, video_path, self.crop_size, cv2)   # (N, H, W)
        # Pull aligned audio: TalkNet uses 4 audio features per video frame.
        n_frames = len(crops)
        samples_per_frame = int(round(16000 / fps))
        start_sample = max(0, int(track.frames[0]) * samples_per_frame)
        end_sample = min(len(audio), start_sample + n_frames * samples_per_frame)
        clip_audio = audio[start_sample:end_sample]
        if clip_audio.size == 0:
            return ASDScore(track.track_id,
                            frames=track.frames.copy(),
                            scores=np.zeros((track.num_frames,), dtype=np.float32))
        feats = mfcc_fn(clip_audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        audio_feat = torch.FloatTensor(np.asarray(feats))
        video_feat = torch.FloatTensor(np.asarray(crops))

        with torch.no_grad():
            if self.device.startswith("cuda"):
                audio_feat = audio_feat.cuda(); video_feat = video_feat.cuda()
            audio_embed = self._model.model.forward_audio_frontend(audio_feat.unsqueeze(0))
            video_embed = self._model.model.forward_visual_frontend(video_feat.unsqueeze(0))
            audio_embed, video_embed = self._model.model.forward_cross_attention(
                audio_embed, video_embed
            )
            out = self._model.model.forward_audio_visual_backend(audio_embed, video_embed)
            logits = self._model.lossAV.FC(out)
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        # Scores are per-frame of the clip (length min(n_frames, len(scores))).
        n = min(len(scores), track.num_frames)
        return ASDScore(
            track_id=track.track_id,
            frames=track.frames[:n].astype(np.int32),
            scores=scores[:n].astype(np.float32),
        )


# ---------------------------------------------------------------------------
# frame cropping — shared with TS-TalkNet
# ---------------------------------------------------------------------------

def _crop_track(track: FaceTrack, video_path: Path, size: int, cv2) -> np.ndarray:
    """Decode the needed frames once and crop around each bbox.

    Returns a `(N, size, size)` grayscale uint8 array. OpenCV's seek is
    approximate on VBR files; we read sequentially and skip, which is slower
    but exact.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    wanted = set(int(f) for f in track.frames)
    frame_to_bbox = {int(f): bb for f, bb in zip(track.frames, track.bboxes)}
    out: List[np.ndarray] = []
    idx = 0
    max_f = int(track.frames[-1])
    try:
        while idx <= max_f:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                x1, y1, x2, y2 = frame_to_bbox[idx]
                # Pad the box to a square and 1.4x out — TalkNet expects
                # the surrounding context, not a tight crop.
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                side = max(x2 - x1, y2 - y1) * 1.4
                xa = int(max(0, cx - side / 2))
                ya = int(max(0, cy - side / 2))
                xb = int(min(frame.shape[1], cx + side / 2))
                yb = int(min(frame.shape[0], cy + side / 2))
                patch = frame[ya:yb, xa:xb]
                if patch.size == 0:
                    out.append(np.zeros((size, size), dtype=np.uint8))
                else:
                    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
                    out.append(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            idx += 1
    finally:
        cap.release()
    if not out:
        return np.zeros((0, size, size), dtype=np.uint8)
    return np.stack(out)
