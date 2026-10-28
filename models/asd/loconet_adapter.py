"""LoCoNet-ASD adapter (Wang et al., CVPR 2024).

Wraps the `SJTUwxz/LoCoNet_ASD` clone (vendored under
`models/asd/vendor/loconet`) driven by the `Superxixixi/LoCoNet_ASD` AVA
weights. The only ASD model here that is not a per-face scorer: it models
the speakers in a scene jointly (short-term inter-speaker convolution
interleaved with long-term intra-speaker attention), and its gain is
concentrated in the multi-speaker case (+3% mAP).

Three load-bearing differences from the MFCC adapters that preceded it
(talknet, light_asd, lr_asd -- all removed 2026-10-28):

1. **The audio frontend is VGGish, not MFCC.** `model/loconet_encoder.py`
   builds `torchvggish.vggish.VGGish`, and the val loader feeds it
   `vggish_input.waveform_to_examples` log-mel (64 bands) at 4x the video
   frame rate -- this calls the clone's own `vggish_input` on the
   int16-valued wav. Only the crop half of `_av_frontend` is shared, since
   the visual input was identical; its MFCC half went with those adapters.

2. **Scoring is joint, per scene.** The model takes `NUM_SPEAKERS` (=3)
   faces at once: the target at slot 0 plus up to two co-occurring faces as
   context, padded by repeating the target when fewer co-occur. To score
   every track it runs once per track with that track at slot 0, then
   demuxes back to one `ASDScore` per track, so stage 6 is unchanged.

3. **It is constructed, not imported.** There is no clean `Model()` class.
   Upstream's `loconet.py` imports `xxlib`, which is not in the clone, so
   this builds `locoencoder(cfg)` + the `loss_multi` heads directly with a
   3-field cfg shim. Weights load via the same `model.module.` strip
   upstream's `loadParameters` uses.

VGGish's own weights are not fetched: `locoencoder` builds
`VGGish(pretrained=True)`, which would download at construction, and our
checkpoint overwrites the audio encoder anyway. The download is no-opped so
the load is offline.

AVA-only weights -- upstream ships no TalkSet fine-tune. An AVA-trained
detector fires on ~43% of frames in dubbed footage where the answer is zero
(Kim et al., Interspeech 2021), which is the condition this pipeline
manufactures, so read these scores as a relative ranker of competing faces
rather than an absolute speaking gate.

One `model`-package-shipping ASD model per process: this clone and the Liao
family both ship a top-level `model/` package and would collide on that
name. Stage 6 runs one ASD model per process.

CUDA only -- upstream's VGGish moves itself to cuda at construction.

If a clone exposes different symbols, override via `LOCONET_MODEL_MODULE` /
`LOCONET_MODEL_CLASS` rather than editing this file.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..faces.base import FaceTrack, FaceTracks
from ._av_frontend import crop_all_tracks, window_bounds
from .base import ASDResult, ASDScore, BaseASD

log = logging.getLogger(__name__)


class LoCoNetASD(BaseASD):
    name = "loconet"
    vendor_name = "loconet"
    env_prefix = "LOCONET"
    legacy_path = "third_party/LoCoNet_ASD"
    clone_url = "https://github.com/SJTUwxz/LoCoNet_ASD.git"
    #: Relative to the clone. The Superxixixi HF `pytorch_model.bin`,
    #: fetched by `scripts/fetch_assets.py` to this name beside the repo.
    default_checkpoint = "loconet_ava.model"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[Path] = None,
        checkpoint: Optional[Path] = None,
        crop_size: int = 112,
        target_fps: float = 25.0,
        window_frames: int = 150,
        num_speakers: int = 3,
    ):
        super().__init__(device=device)
        from .._vendor import resolve_vendor_repo
        self.repo_path = resolve_vendor_repo(
            self.vendor_name, f"{self.env_prefix}_REPO",
            str(repo_path) if repo_path else None,
            legacy=self.legacy_path,
            vendor_root="models/asd/vendor")

        env_ckpt = os.getenv(f"{self.env_prefix}_CHECKPOINT")
        if checkpoint or env_ckpt:
            ck = Path(checkpoint or env_ckpt).expanduser()
            # A relative pin is relative to the CLONE, same rule as the
            # Liao adapters.
            if not ck.is_absolute() and not ck.exists():
                ck = self.repo_path / ck
            if not ck.exists():
                moved = self.repo_path / Path(ck).name
                if moved.exists():
                    ck = moved
            self.checkpoint = ck
        else:
            self.checkpoint = self.repo_path / self.default_checkpoint

        self.crop_size = crop_size
        self.target_fps = target_fps
        # Same VRAM bound and blank-means-default rule as the other
        # adapters, but the activation is ~NUM_SPEAKERS x heavier per window
        # because every candidate face goes through the visual frontend
        # together. Upstream's CLIP_LENGTH is 200.
        self.window_frames = int(os.getenv(f"{self.env_prefix}_WINDOW_FRAMES")
                                 or window_frames)
        self.num_speakers = int(os.getenv(f"{self.env_prefix}_NUM_SPEAKERS")
                                or num_speakers)
        # multi.yaml values, overridable without a yaml because the
        # encoder reads only these three fields.
        self.av_layers = int(os.getenv(f"{self.env_prefix}_AV_LAYERS") or 3)
        self.adjust_attention = int(
            os.getenv(f"{self.env_prefix}_ADJUST_ATTENTION") or 0)
        self._enc = None
        self._lossAV = None
        self._vggish_input = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _make_cfg(self):
        """The three fields `locoencoder`/`ConvLayer`/cross-attention read.

        A full `dlhammer` bootstrap reads argv and a yaml; the encoder only
        touches `MODEL.NUM_SPEAKERS`, `MODEL.AV_layers` and
        `MODEL.ADJUST_ATTENTION`, so a nested namespace suffices.
        """
        from types import SimpleNamespace
        return SimpleNamespace(MODEL=SimpleNamespace(
            NUM_SPEAKERS=self.num_speakers,
            AV_layers=self.av_layers,
            ADJUST_ATTENTION=self.adjust_attention,
        ))

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.repo_path is None or not self.repo_path.exists():
            raise RuntimeError(
                f"LoCoNet repo not found at {self.repo_path}. Clone it with:\n"
                f"  git clone {self.clone_url} "
                f"models/asd/vendor/{self.vendor_name}\n"
                f"or set {self.env_prefix}_REPO to an existing clone."
            )
        if self.checkpoint is None or not self.checkpoint.exists():
            raise RuntimeError(
                f"LoCoNet checkpoint not found at {self.checkpoint}. Fetch the "
                "Superxixixi/LoCoNet_ASD weights (pytorch_model.bin) with "
                "scripts/fetch_assets.py, or set "
                f"{self.env_prefix}_CHECKPOINT."
            )

        import importlib
        import torch

        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        # VGGish would download its own weights at construction, but our
        # checkpoint supplies the audio encoder, so no-op it and keep the
        # load offline. strict=False upstream makes an empty dict a clean
        # skip, not an error.
        orig_loader = torch.hub.load_state_dict_from_url
        torch.hub.load_state_dict_from_url = lambda *a, **k: {}
        try:
            enc_module = os.getenv(f"{self.env_prefix}_MODEL_MODULE",
                                   "model.loconet_encoder")
            enc_class = os.getenv(f"{self.env_prefix}_MODEL_CLASS", "locoencoder")
            loco = importlib.import_module(enc_module)
            locoencoder = getattr(loco, enc_class)
            loss_multi = importlib.import_module("loss_multi")
            from torchvggish import vggish_input

            cfg = self._make_cfg()
            # Mirror upstream's `loconet` (rank=None) object: `.model` is
            # the encoder, `.lossAV/.lossA/.lossV` the heads, so upstream's
            # `model.module.` key strip applies unchanged.
            wrapper = torch.nn.Module()
            wrapper.model = locoencoder(cfg)
            wrapper.lossAV = loss_multi.lossAV()
            wrapper.lossA = loss_multi.lossA()
            wrapper.lossV = loss_multi.lossV()
        finally:
            torch.hub.load_state_dict_from_url = orig_loader

        state = torch.load(str(self.checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Upstream `loadParameters` (rank=None branch) strips
        # `model.module.` from the DDP-saved checkpoint; the remaining
        # `model.*` / `lossAV.*` land on the wrapper above. loconet.py:171.
        clean = {k.replace("model.module.", ""): v for k, v in state.items()}
        info = wrapper.load_state_dict(clean, strict=False)
        n_missing = len(getattr(info, "missing_keys", []) or [])
        n_total = len(wrapper.state_dict())
        if n_missing > n_total * 0.1:
            # A mostly-unmatched load scores noise at exit code 0. The
            # usual cause is a checkpoint keyed for the HF
            # `modeling_loconet.py` wrapper rather than the SJTUwxz layout.
            log.warning(
                "loconet: %d/%d state_dict keys did not load from %s -- "
                "scores may be meaningless. Check the checkpoint's key scheme.",
                n_missing, n_total, self.checkpoint)

        wrapper.eval()
        if self.device.startswith("cuda"):
            wrapper = wrapper.to(self.device)
            # VGGish.forward sends its input to self.device, set at its
            # own construction; keep it on ours after the move.
            wrapper.model.audioEncoder.device = torch.device(self.device)

        self._enc = wrapper.model
        self._lossAV = wrapper.lossAV
        self._wrapper = wrapper
        self._vggish_input = vggish_input
        self._loaded = True
        log.info("loconet: loaded from %s (S=%d, window=%d)",
                 self.checkpoint, self.num_speakers, self.window_frames)

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
        """`reference_audio` is ignored -- LoCoNet takes no enrolment."""
        self._ensure_loaded()

        import cv2
        import torch
        from scipy.io import wavfile

        sr, audio = wavfile.read(str(audio_path))
        if sr != 16000:
            raise ValueError(
                f"loconet expects 16 kHz audio; got {sr}. Resample upstream.")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # The val loader hands wavfile.read's int16 straight to
        # vggish_input, so the log-mel the AVA weights saw is on the int16
        # scale. Do NOT normalise to [-1, 1] here.
        audio = np.asarray(audio)

        crops_by_track = crop_all_tracks(
            tracks.tracks, video_path, self.crop_size, cv2)
        # frame -> crop, per track, so a context face can be looked up at
        # the target's frame indices.
        crop_at: Dict[str, Dict[int, np.ndarray]] = {}
        for tr in tracks.tracks:
            crops = crops_by_track.get(tr.track_id)
            if crops is None or crops.shape[0] == 0:
                crop_at[tr.track_id] = {}
                continue
            n = min(len(crops), len(tr.frames))
            crop_at[tr.track_id] = {
                int(tr.frames[i]): crops[i] for i in range(n)}

        fps = float(tracks.fps or self.target_fps)
        if self.window_frames > 0:
            log.info("loconet: scoring %d track(s), S=%d, windows of %d frames",
                     len(tracks.tracks), self.num_speakers, self.window_frames)

        per_track: Dict[str, ASDScore] = {}
        for tr in tracks.tracks:
            context = self._context_tracks(tr, tracks.tracks)
            per_track[tr.track_id] = self._score_track(
                tr, context, crop_at, audio, fps, torch)

        return ASDResult(
            video_path=video_path,
            model_name=self.name,
            fps=tracks.fps,
            per_track=per_track,
        )

    def _context_tracks(
        self, target: FaceTrack, all_tracks: List[FaceTrack],
    ) -> List[FaceTrack]:
        """Up to NUM_SPEAKERS-1 co-occurring faces, most-overlapping first.

        Upstream picks context by frame overlap and pads by repeating when
        too few co-occur, filling every slot with the target for an isolated
        speaker -- the input distribution the AVA weights were evaluated on.
        """
        tgt_frames = set(int(f) for f in target.frames)
        scored: List[Tuple[int, FaceTrack]] = []
        for tr in all_tracks:
            if tr.track_id == target.track_id:
                continue
            overlap = sum(1 for f in tr.frames if int(f) in tgt_frames)
            if overlap > 0:
                scored.append((overlap, tr))
        scored.sort(key=lambda t: t[0], reverse=True)
        context = [tr for _, tr in scored[:self.num_speakers - 1]]
        # Pad to NUM_SPEAKERS-1 slots by repeating what we have, or the
        # target when nothing co-occurs.
        while len(context) < self.num_speakers - 1:
            context.append(context[-1] if context else target)
        return context

    def _score_track(
        self,
        target: FaceTrack,
        context: List[FaceTrack],
        crop_at: Dict[str, Dict[int, np.ndarray]],
        audio: np.ndarray,
        fps: float,
        torch,
    ) -> ASDScore:
        z = self.crop_size
        if target.num_frames == 0 or not crop_at.get(target.track_id):
            return ASDScore(target.track_id,
                            frames=target.frames.copy().astype(np.int32),
                            scores=np.zeros((target.num_frames,),
                                            dtype=np.float32))

        frames_abs = [int(f) for f in target.frames]
        n_frames = len(frames_abs)
        samples_per_frame = int(round(16000 / fps))
        slots = [target] + context                      # slot 0 = target
        bounds = window_bounds(n_frames, self.window_frames)

        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for f0, f1 in bounds:
                w = f1 - f0
                win_frames = frames_abs[f0:f1]

                # (S, w, z, z): each slot's crop at the target's frame
                # indices. A slot absent on a frame contributes zeros, as the
                # val loader does for a context speaker outside its span.
                visual = np.zeros((self.num_speakers, w, z, z), dtype=np.float32)
                for s, tr in enumerate(slots):
                    lut = crop_at.get(tr.track_id, {})
                    for j, af in enumerate(win_frames):
                        c = lut.get(af)
                        if c is not None:
                            visual[s, j] = c

                # Shared audio for the window: 64-band log-mel at 4x the
                # frame rate, on the int16 scale.
                a0 = win_frames[0] * samples_per_frame
                a1 = min(len(audio), (win_frames[-1] + 1) * samples_per_frame)
                clip_audio = audio[a0:a1]
                if clip_audio.size == 0:
                    chunks.append(np.zeros((w,), dtype=np.float32))
                    continue
                log_mel = self._vggish_input.waveform_to_examples(
                    clip_audio, 16000, w, fps, return_tensor=False)

                sc = self._forward(visual, log_mel, w, torch)
                chunks.append(sc[:w])

        scores = (np.concatenate(chunks) if chunks
                  else np.zeros((0,), dtype=np.float32))
        n = min(len(scores), target.num_frames)
        return ASDScore(
            track_id=target.track_id,
            frames=target.frames[:n].astype(np.int32),
            scores=scores[:n].astype(np.float32),
        )

    def _forward(self, visual: np.ndarray, log_mel: np.ndarray,
                 w: int, torch) -> np.ndarray:
        """One joint forward over the window; returns slot-0 (target) scores.

        Replicates `loconet.evaluate_network`'s inner block for b=1: audio
        frontend (shared, then repeated per speaker) -> visual frontend ->
        cross-attention -> AV backend -> `lossAV.FC` + softmax.
        """
        S = self.num_speakers
        vis = torch.FloatTensor(visual)                  # (S, w, z, z)
        # (1, 1, T_audio, 64): VGGish's conv2d wants a single input channel.
        aud = torch.FloatTensor(np.asarray(log_mel))[None, None, :, :]
        if self.device.startswith("cuda"):
            vis = vis.to(self.device)
            aud = aud.to(self.device)

        audio_embed = self._enc.forward_audio_frontend(aud)      # (1, T, 128)
        visual_embed = self._enc.forward_visual_frontend(vis)    # (S, T, 128)
        audio_embed = audio_embed.repeat(S, 1, 1)                # (S, T, 128)
        audio_embed, visual_embed = self._enc.forward_cross_attention(
            audio_embed, visual_embed)
        out = self._enc.forward_audio_visual_backend(
            audio_embed, visual_embed, 1, S)                     # (S*T, 256)
        logits = self._lossAV.FC(out)                            # (S*T, 2)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        # Speaker-major reshape: row 0 is the target at slot 0.
        probs = probs.view(S, -1)[0]
        return probs.detach().cpu().numpy().astype(np.float32)
