"""Wav2Lip adapter (Rudrabha/Wav2Lip, ECCV 2020).

Lightweight baseline. Keep as a comparison point to the newer
MuseTalk / LatentSync models.

The upstream repo exposes `inference.py` as its entry point. We shell
out to it rather than importing — Wav2Lip pins old versions of
librosa / numpy / torch and importing it in-process clashes with the
rest of the pipeline.

Env:
  WAV2LIP_REPO       (default: models/lipsync/vendor/wav2lip, falling
                      back to third_party/Wav2Lip)
  WAV2LIP_CHECKPOINT (default: <repo>/checkpoints/wav2lip_gan.pth, derived
                      from whichever repo path resolved)
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

from backend.media_io import probe_video as _probe_video
from . import coords
from .base import (BaseLipSync, LipSyncRequest, LipSyncResult,
                   resolve_vendor_repo)

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/Wav2Lip"
DEFAULT_CKPT = "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"

#: The default output ceiling, in scanlines. A source at or below this renders
#: at NATIVE resolution; anything above it is halved. Overridable per process
#: with `WAV2LIP_MAX_HEIGHT`.
#:
#: This replaced a flat `resize_factor = 2` on 2026-10-22. That default cost
#: every run half its resolution to protect against one case: a 4K source,
#: where `inference.py:118` raises "Image too big to run face detection on
#: GPU" (measured 2026-10-18 — shard 1 of a 2-GPU split died there while
#: shard 2 succeeded). But that error is **an OOM fallback, not a pixel
#: limit** — the loop above it halves `face_det_batch_size` on every
#: RuntimeError and only gives up at a batch of 1. So the constraint is VRAM
#: against frame size, and a flat 2 was answering it by downscaling 1080p
#: sources that never needed it.
#:
#: 1080 is the ceiling because 2 is the only factor applied: 2160/2 is
#: exactly 1080, so a 4K source lands on the ceiling rather than under it.
#:
#: Wav2Lip repaints a 96x96 mouth crop at ANY factor, so the mouth itself is
#: unaffected either way — what the old default cost was the other 99% of the
#: frame. VRAM tracks the change and is the thing to watch: ~1.2 GiB at
#: factor 2 versus ~4.0 GiB at 1 on 720p, and 1080p is 2.25x the pixels of
#: 720p. The OOM fallback absorbs the overshoot by halving the detection
#: batch, so the failure mode is a slower render, not a dead one.
MAX_OUTPUT_HEIGHT = int(os.getenv("WAV2LIP_MAX_HEIGHT") or 1080)


def cap_dims(dims: tuple[int, int],
             max_height: int = MAX_OUTPUT_HEIGHT) -> tuple[int, int]:
    """Scale `(w, h)` down to `max_height`, keeping aspect, both sides even.

    A CEILING, never an upscale: a 720p source stays 720p. Inventing pixels
    to reach a nominal 1080p would make every small source render slower and
    look no better.
    """
    w, h = int(dims[0]), int(dims[1])
    if h <= max_height or h <= 0:
        return (w, h)
    scale = max_height / float(h)
    # Even on both axes -- libx264 with yuv420p refuses odd dimensions, and
    # this is the one place a derived size can land on one.
    return (max(2, int(round(w * scale)) // 2 * 2),
            max(2, int(round(h * scale)) // 2 * 2))


def resolve_resize_factor(
    video: Path,
    explicit: Optional[int] = None,
    *,
    max_height: int = MAX_OUTPUT_HEIGHT,
) -> tuple[int, str]:
    """`(factor, reason)` for one source video.

    `explicit` wins untouched. Otherwise: at or below the ceiling render
    native (1), above it halve (2).

    **A failed probe returns 2, not 1.** `_video_dims` answers None when
    ffprobe is missing or the file is unreadable, and None is a silent answer
    in every caller here (CLAUDE.md §1). Guessing "native" on an unknown
    source risks the OOM at stage 9h, after ASR, MT, TTS and the mix have all
    been paid for; guessing "halve" costs resolution on a file we could not
    read. The cheap mistake is the right default.
    """
    if explicit is not None:
        return int(explicit), f"explicit resize_factor={int(explicit)}"
    from backend.lipsync import _video_dims
    dims = _video_dims(Path(video))
    if not dims:
        return 2, (f"could not probe {Path(video).name}; kept the safe "
                   f"factor 2 rather than risk an out-of-memory at render")
    w, h = dims
    if h <= max_height:
        return 1, f"source {w}x{h} is within the {max_height}p ceiling"
    out_h = h // 2
    if out_h > max_height:
        # 8K and up. The rule applies only factor 2, so this lands above the
        # ceiling. Said out loud rather than silently escalating to 4, which
        # would quarter a source nobody asked to quarter.
        return 2, (f"source {w}x{h} is above {max_height}p; factor 2 gives "
                   f"{w // 2}x{out_h}, still above the ceiling — face "
                   f"detection may fall back to a smaller batch")
    return 2, f"source {w}x{h} is above {max_height}p; halving to {w // 2}x{out_h}"


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
        # None means AUTO: cap the output at 1080p. See
        # `resolve_resize_factor` for the rule and why it replaced a flat 2.
        # An explicit int still wins and is passed through untouched, so the
        # adapter-kwarg path (`--adapter-kwargs`, `experiment.py`) is
        # unchanged in meaning.
        resize_factor: Optional[int] = None,
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
        self.repo_path = resolve_vendor_repo(
            "wav2lip", "WAV2LIP_REPO", repo_path, legacy=DEFAULT_REPO)
        # DERIVED from the resolved repo, not a second hardcoded path.
        # `DEFAULT_CKPT` named `third_party/Wav2Lip/...` outright, so after
        # the vendor move the repo would resolve to its new home while the
        # weights still pointed at the old one -- and the failure is the
        # expensive kind: `_ensure_loaded` raises at stage 9h, after ASR,
        # MT, TTS and the mix have been paid for.
        self.checkpoint = Path(
            checkpoint or os.getenv("WAV2LIP_CHECKPOINT")
            or (self.repo_path / "checkpoints" / "wav2lip_gan.pth"))
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

        # Resolved PER VIDEO now, not read off the instance. `backend/
        # lipsync.py` shards a render across the GPUs by time, so every shard
        # is a cut of the same source and resolves the same factor -- but the
        # value has to be computed here, where the video is known, rather
        # than at construction, where it is not.
        resize_factor, why = resolve_resize_factor(
            Path(request.video), self.resize_factor)
        log.info("wav2lip[gpu=%s]: resize_factor=%d (%s)",
                 self.cuda_device, resize_factor, why)

        cmd = [
            self.python_bin, "inference.py",
            "--checkpoint_path", str(self.checkpoint.resolve()),
            "--face", str(Path(request.video).resolve()),
            "--audio", str(Path(request.audio).resolve()),
            "--outfile", str(out.resolve()),
            "--pads", *self.pads.split(","),
            "--resize_factor", str(resize_factor),
            "--wav2lip_batch_size", str(self.wav2lip_batch_size),
            "--face_det_batch_size", str(self.face_det_batch_size),
            "--nosmooth",
        ]

        # The boxes depend on pads and resize_factor (both change the box),
        # and on nosmooth (we always pass it, so it is constant -- recorded
        # anyway so the key stays correct if that ever changes).
        #
        # The RESOLVED factor, not `self.resize_factor`. Keying on the
        # instance attribute would write every auto-resolved run under the
        # same `None`, so a 4K render's half-size boxes would be loaded for a
        # 720p clip and painted at the wrong scale -- at exit code 0, since
        # upstream only checks the box COUNT against the frame count.
        coord_params = {"pads": self.pads,
                        "resize_factor": resize_factor,
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

