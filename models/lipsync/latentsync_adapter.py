"""LatentSync adapter (bytedance/LatentSync) 1.5.

Audio-conditioned latent diffusion with SyncNet supervision and TREPA
temporal consistency. Slower than MuseTalk, higher fidelity. Apache-2.0.

**1.6 was removed on 2026-10-28.** It needed ~18 GB against 16311 MiB cards,
so it had no margin and an OOM surfaced as "produced no mp4" -- the one
registered model that could not actually run on this host.

A version is a (checkpoint, resolution) PAIR and `VERSIONS` keeps the two
together. The architecture is identical across versions, so a 1.5 checkpoint
loads happily under a 512 config and generates badly -- no error, a finished
mp4 with a soft mouth. `_ensure_loaded` reads `data.resolution` out of the
config and refuses a mismatched pair. That guard stays with 1.6 gone,
because `LATENTSYNC_UNET_CONFIG` can still point at a hand-edited 512 yaml.

Shells out to the repo's `scripts/inference.py` rather than importing it --
LatentSync pins specific diffusers / accelerate versions.

Env:
  LATENTSYNC_REPO              (default: models/lipsync/vendor/latentsync,
                                falling back to third_party/LatentSync)
  LATENTSYNC_VERSION           (1.5; the only version left)
  LATENTSYNC_UNET_CONFIG       (overrides the version's config)
  LATENTSYNC_UNET_CKPT         (overrides the version's checkpoint)
  LATENTSYNC_PYTHON            (default: current python)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from backend.media_io import probe_video as _probe_video
from .base import (BaseLipSync, LipSyncRequest, LipSyncResult,
                   resolve_env_python, resolve_vendor_repo, run_streaming)

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/LatentSync"

#: version -> (unet config, checkpoint, the resolution that config MUST have)
#:
#: The third element lets `_ensure_loaded` catch a config pointed elsewhere
#: (`LATENTSYNC_UNET_CONFIG`, a hand-edited yaml) while the checkpoint stayed
#: put. 1.5's and 1.6's checkpoints ship from different HF repos under the
#: same basename, so they are stored under distinct names here.
VERSIONS = {
    "1.5": ("configs/unet/stage2.yaml",
            "checkpoints/latentsync_unet_1.5.pt", 256),
}
DEFAULT_VERSION = "1.5"

#: Weight sources, for the error message. It names the file AND the repo
#: because 1.5 and 1.6 both called theirs `latentsync_unet.pt`, and 1.6's
#: may still be sitting in the vendor checkout under that name.
_HF_REPO = {"1.5": "ByteDance/LatentSync-1.5"}


def _config_resolution(cfg_path: Path) -> Optional[int]:
    """`data.resolution` out of a UNet yaml, or None if unreadable.

    Regex, not a yaml parser: avoids a dependency for one integer, and an
    unexpected syntax degrades to "cannot tell" instead of raising.
    """
    import re
    try:
        text = cfg_path.read_text()
    except OSError:
        return None
    m = re.search(r"^\s*resolution:\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None


class LatentSyncSync(BaseLipSync):
    #: Registered as `latentsync15`, not `latentsync`. The id names the
    #: (checkpoint, resolution) pair rather than the family, which is what
    #: kept 1.5 and 1.6 from being confused for each other while both
    #: existed -- and 1.6's id must not come back as a silent alias for
    #: this one, which renders at half the resolution.
    name = "latentsync15"
    licence = "apache-2.0"
    version = DEFAULT_VERSION

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        version: Optional[str] = None,
        unet_config_path: Optional[str] = None,
        unet_ckpt_path: Optional[str] = None,
        python_bin: Optional[str] = None,
        inference_steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int = 1247,
        enable_deepcache: bool = False,
        memory_efficient: bool = True,
        targets_npz: Optional[str] = None,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = resolve_vendor_repo(
            "latentsync", "LATENTSYNC_REPO", repo_path, legacy=DEFAULT_REPO)
        # Explicit argument, then env, then the subclass's own version --
        # asking for a version outright beats the subclass default.
        ver = str(version or os.getenv("LATENTSYNC_VERSION")
                  or self.version).strip()
        if ver not in VERSIONS:
            raise ValueError(
                f"Unknown LatentSync version {ver!r}. "
                f"Available: {', '.join(sorted(VERSIONS))}")
        self.version = ver
        cfg_default, ckpt_default, self.expected_resolution = VERSIONS[ver]
        self.unet_config_path = unet_config_path or os.getenv("LATENTSYNC_UNET_CONFIG") \
            or cfg_default
        self.unet_ckpt_path = unet_ckpt_path or os.getenv("LATENTSYNC_UNET_CKPT") \
            or ckpt_default
        self.python_bin = resolve_env_python("latentsync", "latentsync", python_bin)
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        # DeepCache: a full UNet pass every `cache_interval` (3) steps, deep
        # blocks reused between. It attacks the same per-step cost as
        # `inference_steps`, so the two multiply -- at 10 steps only 4 full
        # passes remain and the mouth softens. Pair it with 20 steps.
        self.enable_deepcache = bool(enable_deepcache)
        # Upstream holds three copies of the clip in RAM (~14 GB on a
        # 750-frame 1080p shard); this composites straight into the encoder
        # and keeps only the source (~4.7 GB). The pixels are identical -- it
        # changes when frames are freed, not how they are drawn.
        self.memory_efficient = bool(memory_efficient)
        # Per-frame face targets: 106 landmarks per frame, the entire
        # contribution the stock InsightFace pass makes, so supplying them
        # reproduces the stock warp while choosing the speaking face.
        #
        # Indexed against the clip THIS adapter is handed, not the whole
        # video -- `backend.lipsync` renumbers frames from zero per shard and
        # segment, so the caller slices first. A length mismatch is refused.
        self.targets_npz = str(targets_npz) if targets_npz else None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"LatentSync repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/bytedance/LatentSync "
                f"{self.repo_path}"
            )
        inference = self.repo_path / "scripts" / "inference.py"
        if not inference.exists():
            raise RuntimeError(f"LatentSync scripts/inference.py missing at {inference}")
        cfg = (self.repo_path / self.unet_config_path) \
              if not Path(self.unet_config_path).is_absolute() \
              else Path(self.unet_config_path)
        ckpt = (self.repo_path / self.unet_ckpt_path) \
               if not Path(self.unet_ckpt_path).is_absolute() \
               else Path(self.unet_ckpt_path)
        if not cfg.exists():
            raise RuntimeError(f"LatentSync UNet config missing: {cfg}")
        if not ckpt.exists():
            raise RuntimeError(
                f"LatentSync {self.version} UNet checkpoint missing: {ckpt}. "
                f"Download it from https://huggingface.co/"
                f"{_HF_REPO.get(self.version, 'ByteDance/LatentSync-1.6')} "
                f"and save it under that exact name -- both versions ship "
                f"the file as `latentsync_unet.pt`, so they are kept apart "
                f"here by filename."
            )

        # The check this adapter exists for: the versions share an
        # architecture, so a mismatched (checkpoint, resolution) pair loads
        # without complaint and generates badly after ~80x clip length of GPU
        # time. Both must move together.
        got = _config_resolution(cfg)
        if got is not None and got != self.expected_resolution:
            raise RuntimeError(
                f"LatentSync {self.version} expects a {self.expected_resolution}"
                f"x{self.expected_resolution} UNet config, but {cfg} declares "
                f"resolution: {got}. The checkpoint and the resolution are a "
                f"pair -- a mismatch renders without erroring and only looks "
                f"wrong. Use {VERSIONS[self.version][0]}, or set "
                f"LATENTSYNC_VERSION to the version this config belongs to."
            )
        if got is None:
            log.warning("latentsync: could not read `resolution` from %s; "
                        "proceeding without the version/config check", cfg)
        self._loaded = True

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        self._ensure_loaded()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin, "-m", "scripts.inference",
            "--unet_config_path", self.unet_config_path,
            "--inference_ckpt_path", self.unet_ckpt_path,
            "--video_path", str(Path(request.video).resolve()),
            "--audio_path", str(Path(request.audio).resolve()),
            "--video_out_path", str(out.resolve()),
            "--inference_steps", str(self.inference_steps),
            "--guidance_scale", str(self.guidance_scale),
            "--seed", str(self.seed),
            "--temp_dir", str((out.parent / f"_temp_gpu{self.cuda_device}").resolve()),
        ]
        if self.enable_deepcache:
            cmd.append("--enable_deepcache")
        if self.memory_efficient:
            cmd.append("--memory_efficient")
        # `request.extra["targets_npz"]` wins over the constructor kwarg: the
        # sharded and segmented paths slice per piece, so the per-request
        # value is the one indexed against the frames this call sees.
        targets = (request.extra or {}).get("targets_npz") or self.targets_npz
        if targets:
            cmd += ["--targets_npz", str(Path(targets).resolve())]
        log.info("latentsync[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(self.repo_path),
                              env=self._subprocess_env(),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return LipSyncResult(
                model_name=f"latentsync-{self.version}", out_video=out,
                duration_s=0.0, fps=0.0,
                error=f"latentsync rc={proc.returncode}: {proc.stderr.strip()[-3000:]}",
            )
        fps, dur = _probe_video(out)
        return LipSyncResult(
            model_name=f"latentsync-{self.version}", out_video=out,
            duration_s=dur, fps=fps,
            extra={"licence": self.licence,
                   "version": self.version,
                   "resolution": self.expected_resolution,
                   "unet_config": self.unet_config_path,
                   "inference_steps": self.inference_steps,
                   "guidance_scale": self.guidance_scale,
                   "deepcache": self.enable_deepcache,
                   "memory_efficient": self.memory_efficient},
        )

    def sync_many(self, requests):
        """Render many pieces in ONE process, loading the diffusion stack once.

        The vendored `scripts/inference.py` builds its pipeline per
        (video, audio) call, so more than one piece shells out to
        `_latentsync_worker.py` instead, which sets up once and loops the
        pieces through the same `pipeline(...)` call. A single piece keeps
        the per-call path.
        """
        if not requests:
            return []
        if len(requests) == 1:
            return [self.sync(requests[0])]
        self._ensure_loaded()

        import json
        import tempfile

        worker = Path(__file__).resolve().parent / "_latentsync_worker.py"
        manifest = []
        for request in requests:
            out = Path(request.out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Per-request targets win: the caller slices per piece, so that
            # value is the one indexed against this piece's frames.
            targets = (request.extra or {}).get("targets_npz") or self.targets_npz
            manifest.append({
                "video": str(Path(request.video).resolve()),
                "audio": str(Path(request.audio).resolve()),
                "out": str(out.resolve()),
                # Own temp dir per piece, so pieces cannot stomp each
                # other's intermediates.
                "temp_dir": str((out.parent /
                                 f"_temp_gpu{self.cuda_device}_{out.stem}").resolve()),
                "targets_npz": str(Path(targets).resolve()) if targets else None,
            })

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as mf:
            json.dump(manifest, mf)
            mf_path = mf.name

        cmd = [
            self.python_bin, str(worker),
            "--repo", str(self.repo_path.resolve()),
            "--manifest", mf_path,
            "--unet_config_path", self.unet_config_path,
            "--inference_ckpt_path", self.unet_ckpt_path,
            "--inference_steps", str(self.inference_steps),
            "--guidance_scale", str(self.guidance_scale),
            "--seed", str(self.seed),
        ]
        if self.enable_deepcache:
            cmd.append("--enable_deepcache")
        if self.memory_efficient:
            cmd.append("--memory_efficient")
        log.info("latentsync[gpu=%s]: warm worker, %d piece(s): %s",
                 self.cuda_device, len(requests), " ".join(cmd))
        try:
            # Stream live: the worker prints `[worker] start/ok/fail` per
            # piece, so progress lands in run.log as it runs.
            proc = run_streaming(cmd, cwd=str(self.repo_path),
                                 env=self._subprocess_env(), log=log,
                                 tag=f"latentsync[gpu={self.cuda_device}]")
        finally:
            Path(mf_path).unlink(missing_ok=True)

        # Per-piece failure reasons the worker printed, keyed by output path.
        reasons = {}
        for line in (proc.stdout or "").splitlines():
            if line.startswith("[worker] fail "):
                rest = line[len("[worker] fail "):]
                path, _, why = rest.partition(" :: ")
                reasons[path.strip()] = why.strip() or "worker reported failure"

        # A non-zero exit with no per-piece reason means the batch died at
        # load (bad env, missing checkpoint): blame every piece.
        batch_err = None
        if proc.returncode != 0 and not reasons:
            batch_err = (f"latentsync worker rc={proc.returncode}: "
                         f"{proc.stdout.strip()[-3000:]}")

        results = []
        for request in requests:
            out = Path(request.out_path)
            key = str(out.resolve())
            # The file is the arbiter -- upstream can exit 0 having written
            # nothing -- so a present, non-empty file is success and a missing
            # one is failure, whatever the return code said.
            ok = out.is_file() and out.stat().st_size > 0
            if not ok:
                err = (reasons.get(key) or batch_err
                       or f"latentsync produced no output for {out.name}")
                results.append(LipSyncResult(
                    model_name=f"latentsync-{self.version}", out_video=out,
                    duration_s=0.0, fps=0.0, error=err))
                continue
            fps, dur = _probe_video(out)
            results.append(LipSyncResult(
                model_name=f"latentsync-{self.version}", out_video=out,
                duration_s=dur, fps=fps,
                extra={"licence": self.licence, "version": self.version,
                       "resolution": self.expected_resolution,
                       "unet_config": self.unet_config_path,
                       "inference_steps": self.inference_steps,
                       "guidance_scale": self.guidance_scale,
                       "deepcache": self.enable_deepcache,
                       "memory_efficient": self.memory_efficient,
                       "warm_batch": len(requests)}))
        return results



