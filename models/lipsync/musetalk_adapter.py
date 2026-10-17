"""MuseTalk 1.5 adapter (TMElyralab/MuseTalk).

MuseTalk is latent-space inpainting (not diffusion): frozen VAE +
Whisper-tiny audio encoder + SD-style UNet with cross-attention. 256x256
face region, 30fps+ on a V100 — the fast option for development runs.
Weights are MIT-licensed and marked commercial-OK by the authors.

The upstream repo exposes `scripts/inference.py` which takes a tiny YAML
listing (video, audio) pairs. The adapter writes a one-entry YAML on the
fly, shells out to the script, and copies the result to the requested
output path.

Env:
  MUSETALK_REPO       (default: third_party/MuseTalk)
  MUSETALK_CHECKPOINT (default: third_party/MuseTalk/models — the repo's
                        pretrained-weights folder; adapter also accepts
                        a path to a specific unet.pth)
  MUSETALK_VERSION    (default: v15 — v15 or v1)
  MUSETALK_PYTHON     (default: current python)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseLipSync, LipSyncRequest, LipSyncResult

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/MuseTalk"


class MuseTalkSync(BaseLipSync):
    name = "musetalk"
    licence = "MIT (commercial OK)"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        python_bin: Optional[str] = None,
        version: Optional[str] = None,
        batch_size: int = 20,
        fps: int = 25,
        bbox_shift: int = 0,
        use_float16: bool = False,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = Path(repo_path or os.getenv("MUSETALK_REPO") or DEFAULT_REPO)
        self.checkpoint = checkpoint or os.getenv("MUSETALK_CHECKPOINT")
        self.python_bin = python_bin or os.getenv("MUSETALK_PYTHON") or sys.executable
        self.version = (version or os.getenv("MUSETALK_VERSION") or "v15").lower()
        self.batch_size = batch_size
        self.fps = fps
        self.bbox_shift = bbox_shift
        # Upstream halves pe / vae.vae / unet.model, then derives
        # `weight_dtype = unet.model.dtype` and casts whisper and every latent
        # batch to match (scripts/inference.py:52-56, 76, 201) -- so the half
        # path is internally consistent rather than a partial cast.
        self.use_float16 = use_float16

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.repo_path.exists():
            raise RuntimeError(
                f"MuseTalk repo not found at {self.repo_path}. "
                f"Clone: git clone https://github.com/TMElyralab/MuseTalk "
                f"{self.repo_path}"
            )
        inference = self.repo_path / "scripts" / "inference.py"
        if not inference.exists():
            raise RuntimeError(f"MuseTalk scripts/inference.py missing at {inference}")
        self._loaded = True

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        self._ensure_loaded()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        video_abs = Path(request.video).resolve()
        audio_abs = Path(request.audio).resolve()

        with tempfile.TemporaryDirectory(prefix="musetalk_") as td:
            tdp = Path(td)
            yaml_path = tdp / "task.yaml"
            # MuseTalk's inference.py expects {task_name: {video_path, audio_path, bbox_shift}}
            task_name = Path(request.out_path).stem
            yaml_path.write_text(
                f"{task_name}:\n"
                f"  video_path: \"{video_abs}\"\n"
                f"  audio_path: \"{audio_abs}\"\n"
                f"  bbox_shift: {self.bbox_shift}\n"
            )
            result_dir = tdp / "results"
            result_dir.mkdir(exist_ok=True)
            cmd = [
                self.python_bin, "-m", "scripts.inference",
                "--inference_config", str(yaml_path),
                "--result_dir", str(result_dir),
                "--batch_size", str(self.batch_size),
                "--fps", str(self.fps),
                "--version", self.version,
            ]
            if self.use_float16:
                cmd.append("--use_float16")
            log.info("musetalk[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
            proc = subprocess.run(cmd, cwd=str(self.repo_path),
                                  env=self._subprocess_env(),
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                return LipSyncResult(
                    model_name=f"musetalk-{self.version}", out_video=out,
                    duration_s=0.0, fps=0.0,
                    error=f"musetalk rc={proc.returncode}: {proc.stderr.strip()[-3000:]}",
                )
            produced = list(result_dir.rglob("*.mp4"))
            if not produced:
                return LipSyncResult(
                    model_name=f"musetalk-{self.version}", out_video=out,
                    duration_s=0.0, fps=0.0,
                    error=f"musetalk produced no mp4 in {result_dir}",
                )
            # Prefer the file whose name matches the task name.
            match = [p for p in produced if task_name in p.name]
            src = match[0] if match else produced[0]
            shutil.copy2(src, out)

        fps, dur = _probe_video(out)
        return LipSyncResult(
            model_name=f"musetalk-{self.version}", out_video=out,
            duration_s=dur, fps=fps,
            extra={"licence": self.licence, "version": self.version,
                   "use_float16": self.use_float16,
                   "batch_size": self.batch_size},
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
