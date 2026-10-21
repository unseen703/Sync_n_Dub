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

        # TalkNet's MFCC front-end was trained on int16-valued PCM: upstream
        # reads the wav with `scipy.io.wavfile.read` (which yields int16 for
        # 16-bit PCM) and hands that straight to `python_speech_features.mfcc`
        # -- see TalkNet-ASD/dataLoader.py:11,41 and demoTalkNet.py:218-219.
        # soundfile gives us floats in [-1, 1] instead, which shifts every
        # log-energy cepstral term by ~log(32768) and drags all scores toward
        # zero. Measured on one 3 s single-speaker close-up: mean 0.044 /
        # max 0.129 / 0 of 75 frames above 0.5 as floats, versus mean 0.411 /
        # max 0.798 / 31 of 75 frames once rescaled. The stage never errored.
        audio = audio * 32768.0

        # Decode the video ONCE and bucket each cropped face to its track(s).
        # Previously the adapter reopened and resliced the video per track —
        # with N tracks that's O(N * video_len) decode, which exploded on
        # busy footage (262 tracks → hours). Now it's O(video_len).
        crops_by_track = _crop_all_tracks(
            tracks.tracks, video_path, self.crop_size, cv2,
        )

        per_track: Dict[str, ASDScore] = {}
        for tr in tracks.tracks:
            crops = crops_by_track.get(tr.track_id,
                                        np.zeros((0, self.crop_size, self.crop_size),
                                                 dtype=np.uint8))
            per_track[tr.track_id] = self._score_track(
                tr, crops, audio, tracks.fps, mfcc_fn, torch,
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
        crops: np.ndarray,
        audio: np.ndarray,
        fps: float,
        mfcc_fn,
        torch,
    ) -> ASDScore:
        if track.num_frames == 0 or crops.shape[0] == 0:
            return ASDScore(track.track_id,
                            frames=track.frames.copy().astype(np.int32),
                            scores=np.zeros((track.num_frames,), dtype=np.float32))

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
            # Trim both to the common length BEFORE cross-attention.
            #
            # The two frontends disagree by a frame on most clips: the audio
            # one downsamples 100 Hz MFCC by 4 (-> 25 Hz), the visual one
            # emits exactly one embedding per crop, and `n_frames *
            # samples_per_frame` rarely divides into a whole number of MFCC
            # windows. The residual becomes a 17-vs-16 mismatch inside
            # `attentionLayer.forward`'s `src + self.dropout1(src2)`, which
            # raises RuntimeError and takes the whole stage -- and so the
            # whole job -- down with it.
            #
            # Upstream does the same alignment earlier, on durations, in
            # `demoTalkNet.py`:
            #     length = min((audio.shape[0] - audio.shape[0] % 4) / 100,
            #                  video.shape[0] / 25)
            # This adapter builds its tensors itself and never ported that
            # line. Trimming the tail is correct rather than merely safe:
            # the extra frame has no counterpart to attend to.
            t = min(audio_embed.shape[1], video_embed.shape[1])
            if audio_embed.shape[1] != video_embed.shape[1]:
                log.debug("talknet: trimming a=%d v=%d -> %d",
                          audio_embed.shape[1], video_embed.shape[1], t)
            audio_embed = audio_embed[:, :t]
            video_embed = video_embed[:, :t]
            audio_embed, video_embed = self._model.model.forward_cross_attention(
                audio_embed, video_embed
            )
            out = self._model.model.forward_audio_visual_backend(audio_embed, video_embed)
            logits = self._model.lossAV.FC(out)
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        n = min(len(scores), track.num_frames)
        return ASDScore(
            track_id=track.track_id,
            frames=track.frames[:n].astype(np.int32),
            scores=scores[:n].astype(np.float32),
        )


# ---------------------------------------------------------------------------
# frame cropping — shared with TS-TalkNet
# ---------------------------------------------------------------------------

def _crop_all_tracks(
    tracks: List[FaceTrack],
    video_path: Path,
    size: int,
    cv2,
) -> Dict[str, np.ndarray]:
    """Sweep the video once; dispatch each needed frame to every track that
    wants it.

    Returns {track_id: (N, size, size) uint8 grayscale array}, with crops in
    the same frame order as the track's `.frames` array. Any frame a track
    expected but the decoder skipped (e.g. truncated video) is dropped —
    the caller trims scores to `min(len(crops), track.num_frames)` so this
    is safe.
    """
    # Build: frame_idx -> list of (track_id, bbox, position_in_track_output)
    from collections import defaultdict
    buckets: Dict[int, list] = defaultdict(list)
    out: Dict[str, List[Optional[np.ndarray]]] = {}
    for tr in tracks:
        out[tr.track_id] = [None] * len(tr.frames)
        for pos, (f, bb) in enumerate(zip(tr.frames, tr.bboxes)):
            buckets[int(f)].append((tr.track_id, bb, pos))

    if not buckets:
        return {tid: np.zeros((0, size, size), dtype=np.uint8) for tid in out}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    max_needed = max(buckets)
    idx = 0
    try:
        while idx <= max_needed:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in buckets:
                for tid, bb, pos in buckets[idx]:
                    out[tid][pos] = _crop_one(frame, bb, size, cv2)
            idx += 1
    finally:
        cap.release()

    # Compact: drop any None slots (frames the decoder didn't reach) so the
    # returned array lengths match what inference actually consumes.
    packed: Dict[str, np.ndarray] = {}
    for tid, crops in out.items():
        valid = [c for c in crops if c is not None]
        packed[tid] = (np.stack(valid) if valid
                       else np.zeros((0, size, size), dtype=np.uint8))
    return packed


def _crop_one(frame: np.ndarray, bb, size: int, cv2) -> np.ndarray:
    x1, y1, x2, y2 = bb
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.4   # TalkNet wants context, not a tight crop
    xa = int(max(0, cx - side / 2))
    ya = int(max(0, cy - side / 2))
    xb = int(min(frame.shape[1], cx + side / 2))
    yb = int(min(frame.shape[0], cy + side / 2))
    patch = frame[ya:yb, xa:xb]
    if patch.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
