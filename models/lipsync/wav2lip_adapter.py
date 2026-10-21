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
import pickle
import tempfile
import sys
from pathlib import Path
from typing import Optional

from . import coords
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
        # 2, not 1. At 1 the vendored S3FD refuses a large frame outright --
        # `inference.py:81` raises "Image too big to run face detection on
        # GPU. Please use the --resize_factor argument" -- and it does so
        # AFTER the pipeline has paid for ASR, MT, TTS and the mix. Measured
        # 2026-10-18 on a 4K source: shard 1 died exactly there while shard 2
        # of the same run succeeded, so it is frame size alone, not the model
        # or the environment.
        #
        # The cost is real and worth stating: resize_factor halves the whole
        # OUTPUT, so 1080p renders at 960x540. Wav2Lip repaints a 96x96 mouth
        # crop either way, so the mouth itself loses nothing -- but the rest
        # of the frame is downscaled, which reads as a quality drop next to
        # MuseTalk, which composites at full resolution. Pass
        # `resize_factor=1` for a small-frame source where full resolution
        # matters and S3FD will not baulk. VRAM also tracks this: ~1.2 GiB at
        # 2 versus ~4.0 GiB at 1 on 720p.
        resize_factor: int = 2,
        wav2lip_batch_size: int = 16,
        face_det_batch_size: int = 4,
        # Reuse per-frame face boxes across runs of the same clip. Wav2Lip
        # has no upstream caching, so `--use_saved_coord`/`--saved_coord`
        # were added to the vendored inference.py with the same names
        # MuseTalk uses, and both adapters share one cache keyed on the
        # file plus the parameters that move a box.
        use_coord_cache: bool = True,
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
        self.use_coord_cache = use_coord_cache

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

        # The boxes depend on pads and resize_factor (both change the box),
        # and on nosmooth (we always pass it, so it is constant -- recorded
        # anyway so the key stays correct if that ever changes).
        coord_params = {"pads": self.pads,
                        "resize_factor": self.resize_factor,
                        "nosmooth": True}
        video_abs = Path(request.video).resolve()
        coord_file = None
        if self.use_coord_cache:
            # Beside the output, not in the scratch dir below: that one is
            # created after this point and destroyed with the run, so the
            # fresh boxes could never be copied back out of it.
            coord_file = out.parent / f"{out.stem}.coords.pkl"
            cached = coords.load(video_abs, "wav2lip", coord_params)
            if cached is not None:
                coord_file.write_bytes(pickle.dumps(cached))
                cmd += ["--use_saved_coord"]
            cmd += ["--saved_coord", "--coord_path", str(coord_file)]
        log.info("wav2lip[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
        # Each invocation gets its OWN working directory, because upstream
        # writes two intermediates at paths relative to the cwd:
        # `temp/temp.wav` (inference.py:232) and `temp/result.avi` (:269),
        # both hardcoded. With `cwd=self.repo_path` every concurrent call
        # shared them — and `backend/lipsync.py` shards one render across
        # every visible GPU and runs the shards AT THE SAME TIME, so two
        # wav2lip processes raced on one `temp/result.avi`. The loser's
        # final mux read a half-written file and died with "Invalid data
        # found when processing input", then exited 0 with no output
        # (CLAUDE.md section 1). Observed on 2026-10-17: a 2-GPU job lost
        # shard 0 and still reported done at 100%.
        #
        # `inference.py` is passed by absolute path, so Python still puts
        # the REPO on sys.path (it uses the script's directory, not the
        # cwd) and the local `models` / `face_detection` imports resolve.
        # s3fd's weights resolve through `__file__`, also unaffected.
        with tempfile.TemporaryDirectory(prefix="wav2lip-") as scratch:
            (Path(scratch) / "temp").mkdir()
            cmd[1] = str((self.repo_path / "inference.py").resolve())
            proc = subprocess.run(cmd, cwd=scratch,
                                  env=self._subprocess_env(),
                                  capture_output=True, text=True)
            leftover_avi = Path(scratch) / "temp" / "result.avi"
            leftover_exists = leftover_avi.exists()
        if proc.returncode != 0:
            return LipSyncResult(
                model_name="wav2lip", out_video=out, duration_s=0.0, fps=0.0,
                error=f"wav2lip rc={proc.returncode}: {proc.stderr.strip()[-3000:]}",
            )
        # rc=0 does not mean it wrote anything. `inference.py:289` muxes
        # temp/result.avi onto --outfile with subprocess.call() and throws the
        # return code away, so a failing ffmpeg loses the render *after* all
        # the GPU work and still exits 0. Check the file, not the code.
        if self.use_coord_cache and coord_file and coord_file.is_file():
            try:
                blob = pickle.loads(coord_file.read_bytes())
                coords.save(video_abs, "wav2lip", coord_params, blob,
                            frames=len(blob.get("boxes", [])))
            except Exception as e:                       # noqa: BLE001
                log.warning("wav2lip: could not cache coords (%s)", e)
            finally:
                coord_file.unlink(missing_ok=True)

        if not out.exists():
            hint = (" Upstream's final ffmpeg mux failed silently; its "
                    "un-muxed render was in this run's private scratch dir "
                    "and has been cleaned up."
                    if leftover_exists else "")
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
