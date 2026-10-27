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
from ._mfcc_av import (
    MFCC_HZ,
    align_pair,
    crop_all_tracks,
    crop_one,
    load_audio_int16,
    mfcc_features,
    window_bounds,
)
from .base import ASDResult, ASDScore, BaseASD

log = logging.getLogger(__name__)

# The cropping and MFCC half of this adapter moved to `_mfcc_av` on
# 2026-10-25 when Light-ASD arrived needing the identical code. These two
# names are kept because they were module-level and importable; the bodies
# are gone, not duplicated.
_crop_all_tracks = crop_all_tracks
_crop_one = crop_one


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
        window_frames: int = 150,
    ):
        super().__init__(device=device)
        # Resolved the same way the lip-sync upstreams are, so TalkNet moved
        # under `models/asd/vendor/` with them on 2026-10-23 and nothing in
        # `models/` reaches outside it. There was no in-code default here at
        # all -- `.env` carried the only path -- so a move would have broken
        # it outright; `resolve_vendor_repo` also treats an env var pointing
        # at a path that no longer exists as a stale pin rather than an
        # instruction.
        from .._vendor import resolve_vendor_repo
        self.repo_path = resolve_vendor_repo(
            "talknet", "TALKNET_REPO",
            str(repo_path) if repo_path else None,
            legacy="third_party/TalkNet-ASD",
            vendor_root="models/asd/vendor")
        env_ckpt = os.getenv("TALKNET_CHECKPOINT")
        if checkpoint or env_ckpt:
            ck = Path(checkpoint or env_ckpt).expanduser()
            # Same stale-pin rule as the repo: `.env` names the checkpoint
            # INSIDE the clone, so the move relocates it and a pin that
            # outlives its target would fail at stage 6 with "checkpoint not
            # found" while the file sits beside the repo we just resolved.
            if not ck.exists():
                moved = self.repo_path / Path(ck).name
                if moved.exists():
                    ck = moved
            self.checkpoint = ck
        else:
            self.checkpoint = self.repo_path / "pretrain_TalkSet.model"
        self.crop_size = crop_size
        self.target_fps = target_fps
        # Frames per forward pass. The visual frontend holds a
        # (1, T, 64, 56, 56) float activation, ~0.77 MiB/frame, and the crops
        # are 112x112 whatever the source resolution -- so T is the whole
        # memory story and a window size is a hard bound, not a guess. A
        # 7-minute single-face track (10575 frames) asked for 7.91 GiB in one
        # `batch_norm` and took job 20261024-155149 down at stage 6.
        #
        # Upstream never runs a whole track either: `demoTalkNet.py`:237-245
        # cuts it into 1-6 s pieces and runs the FULL stack per piece. This
        # adapter simply never ported that. 150 frames is upstream's longest
        # piece (6 s at 25 fps), ~115 MiB peak. 0 restores the single-pass
        # behaviour for anyone who wants it back.
        # `or window_frames`, not `or 0`: `.env` ships this blank, and a
        # blank must mean "the default" -- reading it as 0 would silently
        # restore the single-pass behaviour this exists to end. "0" is a
        # truthy string, so disabling on purpose still works.
        self.window_frames = int(os.getenv("TALKNET_WINDOW_FRAMES")
                                 or window_frames)
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

        # The int16 rescale that keeps these scores off the floor lives in
        # `_mfcc_av.load_audio_int16` now -- see its docstring for the
        # measurement, and for why a third copy of it was not made.
        audio = load_audio_int16(audio_path, model_name="TalkNet")

        crops_by_track = crop_all_tracks(
            tracks.tracks, video_path, self.crop_size, cv2,
        )

        if self.window_frames > 0:
            log.info("talknet: scoring %d track(s) in windows of %d frames",
                     len(tracks.tracks), self.window_frames)

        per_track: Dict[str, ASDScore] = {}
        for tr in tracks.tracks:
            crops = crops_by_track.get(tr.track_id,
                                        np.zeros((0, self.crop_size, self.crop_size),
                                                 dtype=np.uint8))
            per_track[tr.track_id] = self._score_track(
                tr, crops, audio, tracks.fps, torch,
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
        # No `fps=` here, deliberately: TalkNet's upstream resamples the
        # video to 25 fps and keeps the MFCC window fixed, so passing the
        # source rate would change this stage's output on every non-25 fps
        # video. `_mfcc_av.mfcc_features` documents the asymmetry.
        feats = mfcc_features(clip_audio)

        bounds = window_bounds(n_frames, self.window_frames)

        # MFCC is a fixed 100 Hz whatever the video fps, so a window's audio
        # slice is its frame span through 100/fps -- the same ratio upstream
        # hard-codes as `duration*100` against `duration*25`.
        mfcc_per_frame = MFCC_HZ / float(fps or self.target_fps)

        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for f0, f1 in bounds:
                want = f1 - f0
                a0 = int(round(f0 * mfcc_per_frame))
                a1 = min(len(feats), int(round(f1 * mfcc_per_frame)))
                if a1 - a0 < 4:
                    # The audio frontend downsamples by 4; with fewer rows
                    # than that it has nothing to emit, so the window would
                    # contribute no scores however it was run.
                    chunks.append(np.zeros((want,), dtype=np.float32))
                    continue
                sc = self._score_window(np.asarray(feats[a0:a1]),
                                        crops[f0:f1], torch)
                if len(sc) < want:
                    # The two frontends disagree by up to a frame (see
                    # `_score_window`). On a whole-track pass that was a
                    # one-frame trim at the END, which the caller absorbed.
                    # Across ~70 windows an unpadded short window would
                    # shift every LATER frame's score off its own frame, so
                    # hold the last value instead of letting it slide.
                    sc = np.concatenate([sc, np.full(
                        want - len(sc), sc[-1] if len(sc) else 0.0,
                        dtype=np.float32)])
                chunks.append(sc[:want])

        scores = (np.concatenate(chunks) if chunks
                  else np.zeros((0,), dtype=np.float32))

        n = min(len(scores), track.num_frames)
        return ASDScore(
            track_id=track.track_id,
            frames=track.frames[:n].astype(np.int32),
            scores=scores[:n].astype(np.float32),
        )

    def _score_window(
        self,
        feats: np.ndarray,
        crops: np.ndarray,
        torch,
    ) -> np.ndarray:
        """One forward pass over a bounded slice: one score per crop.

        May return FEWER scores than crops when the two frontends disagree
        on length; the caller pads to the window width so the concatenation
        stays aligned to `track.frames`.

        The caller holds the `torch.no_grad()` -- it spans every window.
        """
        audio_feat = torch.FloatTensor(np.asarray(feats))
        video_feat = torch.FloatTensor(np.asarray(crops))

        if self.device.startswith("cuda"):
            audio_feat = audio_feat.cuda(); video_feat = video_feat.cuda()
        audio_embed = self._model.model.forward_audio_frontend(audio_feat.unsqueeze(0))
        video_embed = self._model.model.forward_visual_frontend(video_feat.unsqueeze(0))
        # Trim both to the common length BEFORE cross-attention. Untrimmed,
        # the one-frame disagreement becomes a 17-vs-16 mismatch inside
        # `attentionLayer.forward`'s `src + self.dropout1(src2)`, which
        # raises and takes the whole stage -- and so the whole job -- down.
        # `_mfcc_av.align_pair` carries the full why.
        audio_embed, video_embed = align_pair(
            audio_embed, video_embed, who="talknet")
        audio_embed, video_embed = self._model.model.forward_cross_attention(
            audio_embed, video_embed
        )
        out = self._model.model.forward_audio_visual_backend(audio_embed, video_embed)
        logits = self._model.lossAV.FC(out)
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
