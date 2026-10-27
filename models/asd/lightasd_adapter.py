"""The Liao et al. ASD family: Light-ASD (CVPR 2023) and LR-ASD (IJCV 2025).

Two registered models, one implementation. `Light-ASD/ASD.py` and
`LR-ASD/ASD.py` are interchangeable from an adapter's side -- same
`class ASD(nn.Module)` with an all-defaulted `__init__`, same
`from loss import lossAV, lossV`, same three `forward_*` methods, same
`loadParameters` stripping `module.`, same `weight/` folder shipping an
AVA checkpoint beside `finetuning_TalkSet.model`. Everything that changed
between the two papers -- better encoders, a channel-attention fusion
module, reworked temporal modelling -- is INSIDE `ASD_Model`, behind
identical names. So `LRASD` subclasses `LightASD` and overrides five
constants, and there is no second scoring path to keep in step.

Both take the same two inputs as TalkNet -- 112x112 grayscale crops and
13-d MFCC -- so everything before the first `forward_*` call comes from
`_mfcc_av`.

**Why have either when TalkNet already works.** Not for the multi-face
case: both score each candidate face in isolation (single-candidate input
is contribution (a) of the CVPR paper), so neither can compare two faces
any more than TalkNet can, and Light-ASD measures WORSE when they compete
-- 16.73 competing-speaker false alarms against TalkNet's 13.84 on
MSDWild many.val. What they buy here is size and provenance:

  - 0.84M params / 0.51G FLOPs (LR-ASD) or 1.0M / 0.6G (Light-ASD),
    against TalkNet's 15.7M / 1.5G. Stage 6 scores on the GPU in the same
    process that later asks `free_devices` whether a card is free, and
    that check measures FOREIGN memory -- so everything this stage holds
    counts as our own context against the lip-sync stage that follows on
    ~100 MiB of headroom.
  - 94.45 / 94.06 AVA val mAP against TalkNet's 92.3.
  - The weights ship INSIDE the clone (`weight/`), so there is no gdrive
    fetch to fail the way TalkNet's does.

**Prefer `lr_asd` over `light_asd`.** It is the same authors' extended
version and dominates on every published axis: 20% fewer parameters and
FLOPs, +0.39 AVA val mAP, and +5.0 points of Columbia F1 on the AVA
weights (86.1 vs 81.1) -- that last one being the generalisation axis,
which is the one that matters on unseen footage. `light_asd` stays
registered so the two can be compared on real material rather than on a
paper table.

**The default checkpoint is the TalkSet fine-tune, not the AVA one**, for
both, and that is a deliberate departure from upstream's default. AVA is
built from movies that include dubbed footage, and an AVA-trained
detector fires on ~43% of frames in dubbed video where the correct answer
is zero (Kim et al., Interspeech 2021, measured against 24% for an
ASW-trained model). This pipeline manufactures exactly that condition.
The TalkSet weights also score better on Columbia either way (96.4 vs
86.1 for LR-ASD). Set `<PREFIX>_CHECKPOINT=weight/pretrain_AVA.model` to
compare.

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

    The five class attributes below are everything that differs between
    the two upstreams. A subclass overriding them gets the whole scoring
    path for free -- see the module docstring for why that is safe.
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
        # Same resolution rule as TalkNet and the lip-sync upstreams: the
        # vendored path first, `third_party/` as a fallback for a checkout
        # where the move has not been run, and an env var pointing at a
        # path that no longer exists treated as a stale pin rather than an
        # instruction. Neither of these two has ever lived in
        # `third_party/` on this host -- both are cloned straight to the
        # vendor root -- so `legacy` is passed for symmetry with TalkNet
        # and for anyone who put one there by hand before reading this.
        from .._vendor import resolve_vendor_repo
        self.repo_path = resolve_vendor_repo(
            self.vendor_name, f"{self.env_prefix}_REPO",
            str(repo_path) if repo_path else None,
            legacy=self.legacy_path,
            vendor_root="models/asd/vendor")

        env_ckpt = os.getenv(f"{self.env_prefix}_CHECKPOINT")
        if checkpoint or env_ckpt:
            ck = Path(checkpoint or env_ckpt).expanduser()
            # A relative pin is relative to the CLONE, because that is where
            # upstream keeps its weights -- `weight/finetuning_TalkSet.model`
            # in `.env` should keep meaning that after a vendor move.
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
        # Same bound, same reason, same blank-means-default rule as
        # `TALKNET_WINDOW_FRAMES`. These encoders are cheaper per frame
        # than TalkNet's, so 150 frames is conservative here rather than
        # tight -- but the activation still scales linearly with track
        # length, which is the property that produced the stage-6 OOM.
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
        # the repo root has to be importable as a top-level path.
        #
        # Both clones expose a top-level `ASD.py` and a `loss.py`, so with
        # BOTH on sys.path the first one inserted wins for every later
        # import -- running light_asd and lr_asd in one process would
        # score the second with the first one's modules, silently. Hence
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
        # `__init__(self, lr=0.001, lrDecay=0.95, **kwargs)` upstream, in
        # both repos -- every argument defaulted, so the bare call is what
        # their own Columbia_test.py does.
        model = ASD()

        # Upstream's own `loadParameters` strips a `module.` prefix (not the
        # `model.` TalkNet strips) and skips keys that are absent or
        # mis-shaped. Using it rather than reimplementing means a checkpoint
        # layout change upstream is their problem, not a silent
        # `strict=False` load onto a randomly-initialised network here.
        loader = getattr(model, "loadParameters", None)
        if callable(loader):
            loader(str(self.checkpoint))
        else:
            state = torch.load(str(self.checkpoint), map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            clean = {k.replace("module.", ""): v for k, v in state.items()}
            missing = model.load_state_dict(clean, strict=False)
            # strict=False is a fallback path, so say what it let through --
            # a wholly unmatched state dict otherwise loads "successfully"
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

        Both repos ship a top-level `ASD.py`, `loss.py` and `model/`
        package under identical names, and Python caches by module NAME --
        so the second model loaded in one process silently gets the
        first one's modules and scores with the wrong network. Stage 6
        runs one model per process, so this guards a future caller that
        sweeps both rather than anything today.

        **The check is on the loaded module's FILE, not on `sys.path`.**
        A first attempt scanned the path and was wrong twice over: a path
        entry outlives the load that added it, so it refused a model
        nothing had actually imported, and it had no way to tell
        re-loading the SAME model (legal) from loading the other one
        (not). `sys.modules["ASD"].__file__` answers the real question
        directly.
        """
        mod = sys.modules.get("ASD")
        origin = getattr(mod, "__file__", None)
        if not origin:
            # Absent, or a namespace/stub with no file -- nothing to
            # conflict with, and guessing would refuse a legitimate load.
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

        # fps IS passed here, unlike the TalkNet adapter: this family's own
        # dataLoader scales the MFCC window by 25/fps so the audio grid
        # lines up with the visual stream at any source rate.
        feats = mfcc_features(clip_audio, fps=eff_fps)

        bounds = window_bounds(n_frames, self.window_frames)
        # The MFCC grid is a fixed 100 Hz regardless of the window scaling
        # above -- scaling changes how much audio each row COVERS, not how
        # many rows per second come out.
        mfcc_per_frame = MFCC_HZ / eff_fps

        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for f0, f1 in bounds:
                want = f1 - f0
                a0 = int(round(f0 * mfcc_per_frame))
                a1 = min(len(feats), int(round(f1 * mfcc_per_frame)))
                if a1 - a0 < 4:
                    # The audio encoder downsamples by 4; with fewer rows
                    # than that it emits nothing, so the window could not
                    # contribute scores however it was run.
                    chunks.append(np.zeros((want,), dtype=np.float32))
                    continue
                sc = self._score_window(np.asarray(feats[a0:a1]),
                                        crops[f0:f1], torch)
                if len(sc) < want:
                    # Hold the last value rather than let a short window
                    # shift every LATER frame's score off its own frame.
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

        The caller holds the `torch.no_grad()` -- it spans every window.

        Two differences from the TalkNet path, both from reading upstream
        rather than assuming the architectures match:

        1. **There is no cross-attention.** This family goes frontends ->
           `forward_audio_visual_backend(embedA, embedV)` directly.
        2. **That backend is `x = x1 + x2`.** Element-wise addition, so the
           length trim is load-bearing: a mismatch raises on the add rather
           than degrading. `align_pair` does it for every adapter here.

        Scoring goes through `lossAV.FC` + softmax rather than
        `lossAV.forward(out, labels=None)`. Upstream's own eval path passes
        labels and unpacks a 4-tuple, while the `labels=None` branch returns
        a bare numpy vector of UNNORMALISED logits (`x[:,1]`, no softmax) --
        which would put this adapter's output on a different scale from
        TalkNet's, and `ASDResult` promises [0, 1] so the Hungarian solver
        has one metric to compare on.
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

    Identical upstream API, so this inherits the entire scoring path and
    overrides only where the clone and its weights live. See the module
    docstring for the evidence that the API really is identical.

    Measured against Light-ASD by its own authors: 0.84M params and 0.51G
    FLOPs (both ~20% lower), 94.45% AVA val mAP against 94.06%, and 86.1%
    avg Columbia F1 on the AVA weights against 81.1%. The paper also
    reports cross-dataset SOTA on Talkies, Columbia and RealVAD with no
    fine-tuning, which is the property worth having here -- this corpus
    looks nothing like AVA.
    """

    name = "lr_asd"
    vendor_name = "lrasd"
    env_prefix = "LRASD"
    legacy_path = "third_party/LR-ASD"
    clone_url = "https://github.com/Junhua-Liao/LR-ASD.git"
    # `default_checkpoint` inherited: upstream ships the same two filenames
    # in the same `weight/` directory.
