"""LipGAN adapter (Rudrabha/LipGAN, ACM MM 2019).

The predecessor to Wav2Lip from the same group. Kept as a lightweight
baseline — the architecture is simpler (no SyncNet discriminator loss)
and the weights are older, so quality is below Wav2Lip. Non-commercial
weights, same as Wav2Lip.

Upstream ships TF 1.x / Keras 2.2 code that cannot run on Blackwell
GPUs (sm_120 needs CUDA 12.8+, which no TF 1.x build supports). We run
a PyTorch port instead: `inference_torch.py` reads the original Keras
`.h5` checkpoint, rebuilds the layer DAG as a torch nn.Module, and
reuses Wav2Lip's S3FD face detector (also PyTorch). Audio features are
12-dim MFCC matching the training-time MATLAB preprocessor.

The port writes directly to `--outfile`, so no results_dir shim.

Env:
  LIPGAN_REPO       (default: third_party/LipGAN)
  LIPGAN_CHECKPOINT (default: third_party/LipGAN/logs/lipgan_residual_mel.h5)
  LIPGAN_PYTHON     (default: current python; the port runs in-env and
                     needs torch, torchaudio, librosa, h5py, opencv)

Install:
  git clone https://github.com/Rudrabha/LipGAN third_party/LipGAN
  # download lipgan_residual_mel.h5 into third_party/LipGAN/logs/
  # face detector weights come from third_party/Wav2Lip/face_detection
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


DEFAULT_REPO = "third_party/LipGAN"
DEFAULT_CKPT = "third_party/LipGAN/logs/lipgan_residual_mel.h5"


class LipGANSync(BaseLipSync):
    name = "lipgan"
    licence = "non-commercial"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        python_bin: Optional[str] = None,
        model_arch: str = "residual",
        pads: str = "0,0,0,0",
        fps: float = 25.0,
        lipgan_batch_size: int = 128,
        face_det_batch_size: int = 64,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = Path(repo_path or os.getenv("LIPGAN_REPO") or DEFAULT_REPO)
        self.checkpoint = Path(checkpoint
                                or os.getenv("LIPGAN_CHECKPOINT")
                                or DEFAULT_CKPT)
        self.python_bin = python_bin or os.getenv("LIPGAN_PYTHON") or sys.executable
        self.model_arch = model_arch
        self.pads = pads
        self.fps = fps
        self.lipgan_batch_size = lipgan_batch_size
        self.face_det_batch_size = face_det_batch_size

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"LipGAN repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/Rudrabha/LipGAN "
                f"{self.repo_path}"
            )
        inference = self.repo_path / "inference_torch.py"
        if not inference.exists():
            raise RuntimeError(f"LipGAN inference_torch.py missing at {inference}")
        if not self.checkpoint.exists():
            raise RuntimeError(
                f"LipGAN checkpoint missing at {self.checkpoint}. "
                f"Download lipgan_residual_mel.h5 from the LipGAN README."
            )
        self._loaded = True

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        self._ensure_loaded()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin, "inference_torch.py",
            "--checkpoint_path", str(self.checkpoint.resolve()),
            "--face", str(Path(request.video).resolve()),
            "--audio", str(Path(request.audio).resolve()),
            "--outfile", str(out.resolve()),
            "--fps", str(self.fps),
            "--pads", *self.pads.split(","),
            "--lipgan_batch_size", str(self.lipgan_batch_size),
            "--face_det_batch_size", str(self.face_det_batch_size),
        ]
        log.info("lipgan[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(self.repo_path),
                              env=self._subprocess_env(),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return LipSyncResult(
                model_name="lipgan", out_video=out, duration_s=0.0, fps=0.0,
                error=f"lipgan rc={proc.returncode}: {proc.stderr.strip()[-3000:]}",
            )

        fps, dur = _probe_video(out)
        return LipSyncResult(
            model_name="lipgan", out_video=out,
            duration_s=dur, fps=fps,
            extra={"licence": self.licence,
                   "checkpoint": str(self.checkpoint),
                   "model_arch": self.model_arch},
        )


def _probe_video(path: Path) -> tuple[float, float]:
    """Return (fps, duration_s) using ffprobe when available, else zeros."""
    # media_io._ffprobe_bin() rather than shutil.which: there is no
    # ffprobe on this host's PATH, and which() returning None here
    # silently degrades to "unknown" instead of failing (CLAUDE.md §1).
    from backend.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe or not Path(path).exists():
        return (0.0, 0.0)
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        fps = 0.0
        dur = 0.0
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
