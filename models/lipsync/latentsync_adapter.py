"""LatentSync 1.6 adapter (bytedance/LatentSync).

Audio-conditioned latent diffusion at 512x512 with SyncNet supervision
and TREPA temporal consistency. Slower than MuseTalk, higher fidelity —
plan §8 marks it as the final-render / results-table model.

Weights are Apache-2.0 — commercial-OK, which is why the registry
licence string says `apache-2.0`.

The upstream repo exposes `scripts/inference.py` taking --unet_config_path,
--inference_ckpt_path, --video_path, --audio_path, --video_out_path, etc.
We shell out rather than import — LatentSync pins specific diffusers /
accelerate versions.

Env:
  LATENTSYNC_REPO              (default: third_party/LatentSync)
  LATENTSYNC_UNET_CONFIG       (default: configs/unet/stage2_512.yaml)
  LATENTSYNC_UNET_CKPT         (default: checkpoints/latentsync_unet.pt)
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

from .base import BaseLipSync, LipSyncRequest, LipSyncResult

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/LatentSync"


class LatentSyncSync(BaseLipSync):
    name = "latentsync"
    licence = "apache-2.0"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        unet_config_path: Optional[str] = None,
        unet_ckpt_path: Optional[str] = None,
        python_bin: Optional[str] = None,
        inference_steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int = 1247,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = Path(repo_path or os.getenv("LATENTSYNC_REPO") or DEFAULT_REPO)
        self.unet_config_path = unet_config_path or os.getenv("LATENTSYNC_UNET_CONFIG") \
            or "configs/unet/stage2_512.yaml"
        self.unet_ckpt_path = unet_ckpt_path or os.getenv("LATENTSYNC_UNET_CKPT") \
            or "checkpoints/latentsync_unet.pt"
        self.python_bin = python_bin or os.getenv("LATENTSYNC_PYTHON") or sys.executable
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

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
                f"LatentSync UNet checkpoint missing: {ckpt}. "
                f"Download `latentsync_unet.pt` from the LatentSync HF repo."
            )
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
        ]
        log.info("latentsync[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(self.repo_path),
                              env=self._subprocess_env(),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return LipSyncResult(
                model_name="latentsync-1.6", out_video=out,
                duration_s=0.0, fps=0.0,
                error=f"latentsync rc={proc.returncode}: {proc.stderr.strip()[-3000:]}",
            )
        fps, dur = _probe_video(out)
        return LipSyncResult(
            model_name="latentsync-1.6", out_video=out,
            duration_s=dur, fps=fps,
            extra={"licence": self.licence,
                   "inference_steps": self.inference_steps,
                   "guidance_scale": self.guidance_scale},
        )


def _probe_video(path: Path) -> tuple[float, float]:
    # media_io._ffprobe_bin() rather than shutil.which: there is no
    # ffprobe on this host's PATH, and which() returning None here
    # silently degrades to "unknown" instead of failing (CLAUDE.md §1).
    from backend.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe or not Path(path).exists():
        return (0.0, 0.0)
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        fps = 0.0; dur = 0.0
        if lines:
            rate = lines[0]
            if "/" in rate:
                n, d = rate.split("/", 1)
                fps = float(n) / float(d) if float(d) else 0.0
            else:
                fps = float(rate)
        if len(lines) > 1:
            dur = float(lines[1])
        return (fps, dur)
    except Exception:
        return (0.0, 0.0)
