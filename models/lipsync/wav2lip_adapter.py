"""Wav2Lip adapter (Rudrabha/Wav2Lip, ECCV 2020).

Lightweight baseline. Keep as a comparison point to the newer
MuseTalk / LatentSync models.

The upstream repo exposes `inference.py` as its entry point. We shell
out to it rather than importing — Wav2Lip pins old versions of
librosa / numpy / torch and importing it in-process clashes with the
rest of the pipeline.

Env:
  WAV2LIP_REPO       (default: third_party/Wav2Lip)
  WAV2LIP_CHECKPOINT (default: third_party/Wav2Lip/checkpoints/wav2lip_gan.pth)
  WAV2LIP_PYTHON     (default: current python; set if Wav2Lip lives in a
                      separate venv to avoid dependency conflicts)

Install:
  git clone https://github.com/Rudrabha/Wav2Lip third_party/Wav2Lip
  # download wav2lip_gan.pth or wav2lip.pth into third_party/Wav2Lip/checkpoints/
  # also need face_detection/detection/sfd/s3fd.pth
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


DEFAULT_REPO = "third_party/Wav2Lip"
DEFAULT_CKPT = "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"


class Wav2LipSync(BaseLipSync):
    name = "wav2lip"
    licence = "non-commercial"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        python_bin: Optional[str] = None,
        pads: str = "0,10,0,0",
        resize_factor: int = 1,
        wav2lip_batch_size: int = 16,
        face_det_batch_size: int = 4,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = Path(repo_path or os.getenv("WAV2LIP_REPO") or DEFAULT_REPO)
        self.checkpoint = Path(checkpoint
                                or os.getenv("WAV2LIP_CHECKPOINT")
                                or DEFAULT_CKPT)
        self.python_bin = python_bin or os.getenv("WAV2LIP_PYTHON") or sys.executable
        self.pads = pads
        self.resize_factor = resize_factor
        self.wav2lip_batch_size = wav2lip_batch_size
        self.face_det_batch_size = face_det_batch_size

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"Wav2Lip repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/Rudrabha/Wav2Lip "
                f"{self.repo_path}"
            )
        inference = self.repo_path / "inference.py"
        if not inference.exists():
            raise RuntimeError(f"Wav2Lip inference.py missing at {inference}")
        if not self.checkpoint.exists():
            raise RuntimeError(
                f"Wav2Lip checkpoint missing at {self.checkpoint}. "
                f"Download wav2lip_gan.pth from the Wav2Lip README."
            )
        self._loaded = True

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        self._ensure_loaded()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin, "inference.py",
            "--checkpoint_path", str(self.checkpoint.resolve()),
            "--face", str(Path(request.video).resolve()),
            "--audio", str(Path(request.audio).resolve()),
            "--outfile", str(out.resolve()),
            "--pads", *self.pads.split(","),
            "--resize_factor", str(self.resize_factor),
            "--wav2lip_batch_size", str(self.wav2lip_batch_size),
            "--face_det_batch_size", str(self.face_det_batch_size),
            "--nosmooth",
        ]
        log.info("wav2lip[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(self.repo_path),
                              env=self._subprocess_env(),
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return LipSyncResult(
                model_name="wav2lip", out_video=out, duration_s=0.0, fps=0.0,
                error=f"wav2lip rc={proc.returncode}: {proc.stderr.strip()[-3000:]}",
            )
        # rc=0 does not mean it wrote anything. `inference.py:289` muxes
        # temp/result.avi onto --outfile with subprocess.call() and throws the
        # return code away, so a failing ffmpeg loses the render *after* all
        # the GPU work and still exits 0. Check the file, not the code.
        if not out.exists():
            leftover = self.repo_path / "temp" / "result.avi"
            hint = (f" The un-muxed render is at {leftover} — upstream's final "
                    f"ffmpeg step failed silently."
                    if leftover.exists() else "")
            return LipSyncResult(
                model_name="wav2lip", out_video=out, duration_s=0.0, fps=0.0,
                error=f"wav2lip exited 0 but wrote no {out}.{hint} "
                      f"stderr tail: {proc.stderr.strip()[-1500:]}",
            )
        fps, dur = _probe_video(out)
        return LipSyncResult(
            model_name="wav2lip", out_video=out,
            duration_s=dur, fps=fps,
            extra={"licence": self.licence,
                   "checkpoint": str(self.checkpoint)},
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
