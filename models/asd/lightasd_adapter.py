"""The Liao et al. ASD family: Light-ASD (CVPR 2023) and LR-ASD (IJCV 2025).

Two registered models, one implementation. Both clones expose the same
`class ASD(nn.Module)` with an all-defaulted `__init__`, the same three
`forward_*` methods, the same `loadParameters` stripping `module.`, and the
same `weight/` folder; everything that changed between the papers is inside
`ASD_Model` behind identical names. So `LRASD` subclasses `LightASD` and
overrides five constants.

Both take the same two inputs as TalkNet -- 112x112 grayscale crops and
13-d MFCC -- so everything before the first `forward_*` call comes from
`_mfcc_av`.

Neither helps with the multi-face case: both score each candidate face in
isolation, and Light-ASD measures worse than TalkNet when faces compete
(16.73 vs 13.84 competing-speaker false alarms on MSDWild many.val). What
they buy is size and provenance:

  - 0.84M params / 0.51G FLOPs (LR-ASD) or 1.0M / 0.6G (Light-ASD) against
    TalkNet's 15.7M / 1.5G. Stage 6 scores on the GPU in the same process
    that later asks `free_devices` whether a card is free, and that check
    measures FOREIGN memory, so everything this stage holds counts against
    the lip-sync stage that follows on ~100 MiB of headroom.
  - 94.45 / 94.06 AVA val mAP against TalkNet's 92.3.
  - The weights ship inside the clone (`weight/`), so there is no gdrive
    fetch to fail the way TalkNet's does.

Prefer `lr_asd`: 20% fewer parameters and FLOPs, +0.39 AVA val mAP, and
+5.0 Columbia F1 on the AVA weights (86.1 vs 81.1).

The default checkpoint is the TalkSet fine-tune, not upstream's AVA one.
AVA includes dubbed footage, and an AVA-trained detector fires on ~43% of
frames in dubbed video where the answer is zero (Kim et al., Interspeech
2021, vs 24% for an ASW-trained model) -- the condition this pipeline
manufactures. TalkSet also scores better on Columbia (96.4 vs 86.1 for
LR-ASD). Set `<PREFIX>_CHECKPOINT=weight/pretrain_AVA.model` to compare.

If a clone exposes different symbols, override via
`<PREFIX>_MODEL_MODULE` / `<PREFIX>_MODEL_CLASS` rather than editing this
file.
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
    load_audio_int16,
    mfcc_features,
    window_bounds,
)
from .base import ASDResult, ASDScore, BaseASD

log = logging.getLogger(__name__)


class LightASD(BaseASD):
    """Light-ASD (Liao et al., CVPR 2023). Also the base for `LRASD`.

    The five class attributes below are everything that differs between the
    two upstreams, so a subclass overriding them inherits the whole scoring
    path.
    """

    name = "light_asd"
    #: Directory under `models/asd/vendor/`.
    vendor_name = "lightasd"
    #: Prefix for `<PREFIX>_{REPO,CHECKPOINT,WINDOW_FRAMES,MODEL_*}`.
    env_prefix = "LIGHTASD"
    #: Pre-vendor-move location, for a checkout where the move never ran.
    legacy_path = "third_party/Light-ASD"
    clone_url = "https://github.com/Junhua-Liao/Light-ASD.git"
    #: Relative to the clone. See the module docstring on TalkSet vs AVA.
    default_checkpoint = "weight/finetuning_TalkSet.model"

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
        # Same resolution rule as TalkNet and the lip-sync upstreams:
        # vendored path first, `third_party/` as a fallback, and an env var
        # pointing at a missing path treated as a stale pin rather than an
        # instruction. `legacy` is passed for symmetry; neither clone has
        # lived in `third_party/` on this host.
        from .._vendor import resolve_vendor_repo
        self.repo_path = resolve_vendor_repo(
            self.vendor_name, f"{self.env_prefix}_REPO",
            str(repo_path) if repo_path else None,
            legacy=self.legacy_path,
            vendor_root="models/asd/vendor")

        env_ckpt = os.getenv(f"{self.env_prefix}_CHECKPOINT")
        if checkpoint or env_ckpt:
            ck = Path(checkpoint or env_ckpt).expanduser()
            # A relative pin is relative to the CLONE, where upstream
            # keeps its weights, so `weight/finetuning_TalkSet.model` in
            # `.env` keeps meaning that after a vendor move.
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
        # Same bound and blank-means-default rule as
        # `TALKNET_WINDOW_FRAMES`. These encoders are cheaper per frame than
        # TalkNet's, but the activation still scales linearly with track
        # length, which is what produced the stage-6 OOM.
        self.window_frames = int(os.getenv(f"{self.env_prefix}_WINDOW_FRAMES")
                                 or window_frames)
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.repo_path is None or not self.repo_path.exists():
            raise RuntimeError(
                f"{self.name} repo not found at {self.repo_path}. "
                f"Clone it with:\n"
                f"  git clone {self.clone_url} "
                f"models/asd/vendor/{self.vendor_name}\n"
                f"or set {self.env_prefix}_REPO to an existing clone."
            )
        if self.checkpoint is None or not self.checkpoint.exists():
            raise RuntimeError(
                f"{self.name} checkpoint not found at {self.checkpoint}. "
                "Upstream ships weights inside the clone's `weight/` "
                "directory, so a missing one usually means the clone is "
                "shallow or incomplete rather than that a download is owed."
            )
        # Upstream imports its own submodules as `model.Model`, `loss`, so
        # the repo root must be importable as a top-level path.
        #
        # Both clones expose a top-level `ASD.py` and `loss.py`, so with both
        # on sys.path the first inserted wins for every later import. Hence
        # the guard: one Liao-family model per process.
        self._refuse_sibling_on_path()
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))

        model_module = os.getenv(f"{self.env_prefix}_MODEL_MODULE", "ASD")
        model_class = os.getenv(f"{self.env_prefix}_MODEL_CLASS", "ASD")
        import importlib
        import torch
        mod = importlib.import_module(model_module)
        ASD = getattr(mod, model_class)
        # `__init__(self, lr=0.001, lrDecay=0.95, **kwargs)` in both
        # repos -- all defaulted, so the bare call matches Columbia_test.py.
        model = ASD()

        # Upstream's own `loadParameters` strips a `module.` prefix (not
        # the `model.` TalkNet strips) and skips absent or mis-shaped keys.
        # Using it keeps a checkpoint layout change upstream's problem.
        loader = getattr(model, "loadParameters", None)
        if callable(loader):
            loader(str(self.checkpoint))
        else:
            state = torch.load(str(self.checkpoint), map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            clean = {k.replace("module.", ""): v for k, v in state.items()}
            missing = model.load_state_dict(clean, strict=False)
            # strict=False is a fallback, so say what it let through: a
            # wholly unmatched state dict otherwise loads "successfully"
            # and scores noise.
            if getattr(missing, "missing_keys", None):
                log.warning("%s: %d key(s) missing from the checkpoint; "
                            "scores may be meaningless",
                            self.name, len(missing.missing_keys))

        model.eval()
        if self.device.startswith("cuda"):
            model = model.cuda()
        self._model = model
        self._loaded = True
        log.info("%s: loaded from %s", self.name, self.checkpoint)

    def _refuse_sibling_on_path(self) -> None:
        """Refuse if a DIFFERENT clone's `ASD` module is already imported.

        Both repos ship a top-level `ASD.py`, `loss.py` and `model/` under
        identical names and Python caches by module name, so the second
        model loaded in one process silently gets the first one's modules.

        The check is on the loaded module's FILE, not on `sys.path`: a path
        entry outlives the load that added it, and it cannot tell re-loading
        the same model (legal) from loading the other one (not).
        """
        mod = sys.modules.get("ASD")
        origin = getattr(mod, "__file__", None)
        if not origin:
            # Absent, or a namespace/stub with no file -- nothing to
            # conflict with.
            return
        loaded_from = Path(origin).resolve().parent
        if loaded_from == self.repo_path.resolve():
            return
        raise RuntimeError(
            f"Cannot load {self.name}: a sibling clone has already imported "
            f"the top-level `ASD` module from {loaded_from}, and "
            f"{self.repo_path} ships one under the same name -- so this "
            "model would score with the other's weights at no error. Run "
            "one Liao-family ASD model per process."
        )

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
        """`reference_audio` is ignored -- neither model takes enrolment."""
        self._ensure_loaded()

        import cv2
        import torch

        audio = load_audio_int16(audio_path, model_name=self.name)
        crops_by_track = crop_all_tracks(
            tracks.tracks, video_path, self.crop_size, cv2,
        )

        if self.window_frames > 0:
            log.info("%s: scoring %d track(s) in windows of %d frames",
                     self.name, len(tracks.tracks), self.window_frames)

        per_track: Dict[str, ASDScore] = {}
        for tr in tracks.tracks:
            crops = crops_by_track.get(
                tr.track_id,
                np.zeros((0, self.crop_size, self.crop_size), dtype=np.uint8))
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
                            scores=np.zeros((track.num_frames,),
                                            dtype=np.float32))

        n_frames = len(crops)
        eff_fps = float(fps or self.target_fps)
        samples_per_frame = int(round(16000 / eff_fps))
        start_sample = max(0, int(track.frames[0]) * samples_per_frame)
        end_sample = min(len(audio),
                         start_sample + n_frames * samples_per_frame)
        clip_audio = audio[start_sample:end_sample]
        if clip_audio.size == 0:
            return ASDScore(track.track_id,
                            frames=track.frames.copy(),
                            scores=np.zeros((track.num_frames,),
                                            dtype=np.float32))

        # fps IS passed here, unlike the TalkNet adapter: this family's
        # dataLoader scales the MFCC window by 25/fps.
        feats = mfcc_features(clip_audio, fps=eff_fps)

        bounds = window_bounds(n_frames, self.window_frames)
        # The MFCC grid is a fixed 100 Hz whatever the window scaling
        # above: scaling changes what each row COVERS, not the row rate.
        mfcc_per_frame = MFCC_HZ / eff_fps

        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for f0, f1 in bounds:
                want = f1 - f0
                a0 = int(round(f0 * mfcc_per_frame))
                a1 = min(len(feats), int(round(f1 * mfcc_per_frame)))
                if a1 - a0 < 4:
                    # The audio encoder downsamples by 4; with fewer rows
                    # it emits nothing.
                    chunks.append(np.zeros((want,), dtype=np.float32))
                    continue
                sc = self._score_window(np.asarray(feats[a0:a1]),
                                        crops[f0:f1], torch)
                if len(sc) < want:
                    # Hold the last value, or a short window shifts every
                    # LATER frame's score off its own frame.
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

        The caller holds the `torch.no_grad()`; it spans every window.

        Two differences from the TalkNet path:

        1. No cross-attention -- frontends go straight to
           `forward_audio_visual_backend(embedA, embedV)`.
        2. That backend is `x = x1 + x2`, so the `align_pair` length trim is
           load-bearing: a mismatch raises on the add rather than degrading.

        Scoring goes through `lossAV.FC` + softmax, not
        `lossAV.forward(out, labels=None)`, whose `labels=None` branch
        returns unnormalised logits. `ASDResult` promises [0, 1] so the
        Hungarian solver has one metric to compare on.
        """
        audio_feat = torch.FloatTensor(np.asarray(feats))
        video_feat = torch.FloatTensor(np.asarray(crops))

        if self.device.startswith("cuda"):
            audio_feat = audio_feat.cuda()
            video_feat = video_feat.cuda()

        audio_embed = self._model.model.forward_audio_frontend(
            audio_feat.unsqueeze(0))
        video_embed = self._model.model.forward_visual_frontend(
            video_feat.unsqueeze(0))
        audio_embed, video_embed = align_pair(
            audio_embed, video_embed, who=self.name)
        out = self._model.model.forward_audio_visual_backend(
            audio_embed, video_embed)
        logits = self._model.lossAV.FC(out)
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().astype(
            np.float32)


class LRASD(LightASD):
    """LR-ASD (Liao et al., IJCV 2025) -- the extended Light-ASD.

    Identical upstream API, so this inherits the whole scoring path and
    overrides only where the clone and its weights live.

    Against Light-ASD: 0.84M params / 0.51G FLOPs (~20% lower), 94.45% AVA
    val mAP vs 94.06%, 86.1% avg Columbia F1 on the AVA weights vs 81.1%.
    """

    name = "lr_asd"
    vendor_name = "lrasd"
    env_prefix = "LRASD"
    legacy_path = "third_party/LR-ASD"
    clone_url = "https://github.com/Junhua-Liao/LR-ASD.git"
    # `default_checkpoint` inherited: upstream ships the same two filenames
    # in the same `weight/` directory.
