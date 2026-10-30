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
                   resolve_vendor_repo, run_streaming)

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/Wav2Lip"
DEFAULT_CKPT = "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"

#: Output ceiling in scanlines. At or below it a source renders NATIVE;
#: above it is halved. Override per process with `WAV2LIP_MAX_HEIGHT`.
#:
#: 1080 because 2 is the only factor applied, so 2160/2 lands exactly on the
#: ceiling. `inference.py:118` ("Image too big to run face detection on GPU")
#: is an OOM fallback, not a pixel limit -- the loop above it halves
#: `face_det_batch_size` on every RuntimeError down to 1 -- so the real
#: constraint is VRAM against frame size: ~1.2 GiB at factor 2 vs ~4.0 GiB at
#: 1 on 720p. Wav2Lip repaints a 96x96 mouth crop at any factor, so a lower
#: factor costs only the rest of the frame.
MAX_OUTPUT_HEIGHT = int(os.getenv("WAV2LIP_MAX_HEIGHT") or 1080)


def cap_dims(dims: tuple[int, int],
             max_height: int = MAX_OUTPUT_HEIGHT) -> tuple[int, int]:
    """Scale `(w, h)` down to `max_height`, keeping aspect, both sides even.

    A ceiling, never an upscale: a 720p source stays 720p.
    """
    w, h = int(dims[0]), int(dims[1])
    if h <= max_height or h <= 0:
        return (w, h)
    scale = max_height / float(h)
    # Even on both axes: libx264 with yuv420p refuses odd dimensions.
    return (max(2, int(round(w * scale)) // 2 * 2),
            max(2, int(round(h * scale)) // 2 * 2))


def resolve_resize_factor(
    video: Path,
    explicit: Optional[int] = None,
    *,
    max_height: int = MAX_OUTPUT_HEIGHT,
) -> tuple[int, str]:
    """`(factor, reason)` for one source video.

    `explicit` wins untouched. Otherwise: native (1) at or below the ceiling,
    halve (2) above it.

    A FAILED PROBE RETURNS 2, not 1. Guessing native on an unknown source
    risks an OOM at stage 9h, after ASR, MT, TTS and the mix are already
    paid for; guessing halve only costs resolution.
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
        # 8K and up. Only factor 2 is applied, so this lands above the
        # ceiling; it says so rather than silently escalating to 4.
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
        # None means AUTO: cap the output at 1080p, per
        # `resolve_resize_factor`. An explicit int wins and is passed through
        # untouched.
        resize_factor: Optional[int] = None,
        # None = scale from VRAM via backend.scaling (baselines 16 / 4 on
        # 16 GiB). An explicit int wins. See backend/scaling.py.
        wav2lip_batch_size: Optional[int] = None,
        face_det_batch_size: Optional[int] = None,
        # Reuse per-frame face boxes across runs. Wav2Lip has no upstream
        # caching, so `--use_saved_coord`/`--saved_coord` were added to the
        # vendored inference.py under MuseTalk's names; both adapters share
        # one cache keyed on the file plus whatever moves a box.
        use_coord_cache: bool = True,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = resolve_vendor_repo(
            "wav2lip", "WAV2LIP_REPO", repo_path, legacy=DEFAULT_REPO)
        # DERIVED from the resolved repo, not a second hardcoded path, or a
        # vendor move leaves the weights pointing at the old home and
        # `_ensure_loaded` raises at stage 9h with everything else paid for.
        self.checkpoint = Path(
            checkpoint or os.getenv("WAV2LIP_CHECKPOINT")
            or (self.repo_path / "checkpoints" / "wav2lip_gan.pth"))
        self.python_bin = python_bin or os.getenv("WAV2LIP_PYTHON") or sys.executable
        self.pads = pads
        self.resize_factor = resize_factor
        if wav2lip_batch_size is None or face_det_batch_size is None:
            from backend.scaling import (wav2lip_batch_default,
                                         wav2lip_face_det_default)
            if wav2lip_batch_size is None:
                wav2lip_batch_size = wav2lip_batch_default()
            if face_det_batch_size is None:
                face_det_batch_size = wav2lip_face_det_default()
        self.wav2lip_batch_size = int(wav2lip_batch_size)
        self.face_det_batch_size = int(face_det_batch_size)
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

        # Resolved PER VIDEO, not at construction: the video is only known
        # here. Shards are cuts of one source, so they all resolve alike.
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

        # Boxes depend on pads, resize_factor and nosmooth, so all three are
        # in the key. Use the RESOLVED factor, never `self.resize_factor`:
        # keying on the instance attribute files every auto run under `None`,
        # so one clip's boxes get painted on another at the wrong scale --
        # at exit code 0, since upstream only checks the box COUNT.
        coord_params = {"pads": self.pads,
                        "resize_factor": resize_factor,
                        "nosmooth": True}
        video_abs = Path(request.video).resolve()
        coord_file = None
        if self.use_coord_cache:
            # Beside the output, not in the scratch dir below: that one is
            # destroyed with the run, so fresh boxes could never be copied
            # back out.
            coord_file = out.parent / f"{out.stem}.coords.pkl"
            cached = coords.load(video_abs, "wav2lip", coord_params)
            if cached is not None:
                coord_file.write_bytes(pickle.dumps(cached))
                cmd += ["--use_saved_coord"]
            cmd += ["--saved_coord", "--coord_path", str(coord_file)]
        log.info("wav2lip[gpu=%s]: %s", self.cuda_device, " ".join(cmd))
        # Each invocation gets its OWN working directory. Upstream hardcodes
        # two cwd-relative intermediates, `temp/temp.wav` (inference.py:232)
        # and `temp/result.avi` (:269), and shards run concurrently, so a
        # shared cwd makes two processes race on one `temp/result.avi`. The
        # loser muxes a half-written file and exits 0 with no output.
        #
        # `inference.py` is passed by absolute path, so Python puts the REPO
        # on sys.path (the script's directory, not the cwd) and the local
        # `models` / `face_detection` imports still resolve. s3fd's weights
        # resolve through `__file__`.
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
        # rc=0 does not mean it wrote anything: `inference.py:289` muxes
        # onto --outfile with subprocess.call() and discards the return code,
        # so a failing ffmpeg loses the render and still exits 0. Check the
        # file, not the code.
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

    def sync_many(self, requests):
        """Render many pieces in ONE process, loading the net once.

        The vendored `inference.py` loads the net inside its render loop, so
        more than one piece shells out to `_wav2lip_worker.py`, which loads
        once and loops; a single piece keeps the per-call path. The coord
        cache works as in `sync()`, per piece: stage a hit before the run,
        copy fresh boxes out after.
        """
        if not requests:
            return []
        if len(requests) == 1:
            return [self.sync(requests[0])]
        self._ensure_loaded()

        import json

        worker = Path(__file__).resolve().parent / "_wav2lip_worker.py"
        manifest = []
        staged = []            # (out, video_abs, coord_params, coord_file|None)
        for request in requests:
            out = Path(request.out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            video_abs = Path(request.video).resolve()
            # Per video, as in sync(): the resolved factor moves the box.
            resize_factor, why = resolve_resize_factor(
                video_abs, self.resize_factor)
            log.info("wav2lip[gpu=%s]: piece %s resize_factor=%d (%s)",
                     self.cuda_device, out.name, resize_factor, why)
            coord_params = {"pads": self.pads,
                            "resize_factor": resize_factor,
                            "nosmooth": True}
            coord_file = None
            use_saved = False
            if self.use_coord_cache:
                coord_file = out.parent / f"{out.stem}.coords.pkl"
                cached = coords.load(video_abs, "wav2lip", coord_params)
                if cached is not None:
                    coord_file.write_bytes(pickle.dumps(cached))
                    use_saved = True
            manifest.append({
                "face": str(video_abs),
                "audio": str(Path(request.audio).resolve()),
                "out": str(out.resolve()),
                "pads": [int(x) for x in self.pads.split(",")],
                "resize_factor": resize_factor,
                "wav2lip_batch_size": self.wav2lip_batch_size,
                "face_det_batch_size": self.face_det_batch_size,
                "use_saved_coord": use_saved,
                "saved_coord": bool(self.use_coord_cache),
                "coord_path": str(coord_file.resolve()) if coord_file else None,
            })
            staged.append((out, video_abs, coord_params, coord_file))

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as mf:
            json.dump(manifest, mf)
            mf_path = mf.name

        cmd = [self.python_bin, str(worker),
               "--repo", str(self.repo_path.resolve()),
               "--manifest", mf_path,
               "--checkpoint_path", str(self.checkpoint.resolve())]
        log.info("wav2lip[gpu=%s]: warm worker, %d piece(s): %s",
                 self.cuda_device, len(requests), " ".join(cmd))
        try:
            # Stream live: the worker prints `[worker] start/ok/fail` per piece.
            proc = run_streaming(cmd, cwd=str(self.repo_path),
                                 env=self._subprocess_env(), log=log,
                                 tag=f"wav2lip[gpu={self.cuda_device}]")
        finally:
            Path(mf_path).unlink(missing_ok=True)

        reasons = {}
        for line in (proc.stdout or "").splitlines():
            if line.startswith("[worker] fail "):
                rest = line[len("[worker] fail "):]
                path, _, w = rest.partition(" :: ")
                reasons[path.strip()] = w.strip() or "worker reported failure"
        batch_err = None
        if proc.returncode != 0 and not reasons:
            batch_err = (f"wav2lip worker rc={proc.returncode}: "
                         f"{proc.stdout.strip()[-3000:]}")

        results = []
        for out, video_abs, coord_params, coord_file in staged:
            # Copy fresh boxes into the cache whatever the outcome -- a
            # cache write must never fail a good render.
            if self.use_coord_cache and coord_file and coord_file.is_file():
                try:
                    blob = pickle.loads(coord_file.read_bytes())
                    coords.save(video_abs, "wav2lip", coord_params, blob,
                                frames=len(blob.get("boxes", [])))
                except Exception as e:                       # noqa: BLE001
                    log.warning("wav2lip: could not cache coords (%s)", e)
                finally:
                    coord_file.unlink(missing_ok=True)

            ok = out.is_file() and out.stat().st_size > 0
            if not ok:
                err = (reasons.get(str(out.resolve())) or batch_err
                       or f"wav2lip produced no output for {out.name}")
                results.append(LipSyncResult(
                    model_name="wav2lip", out_video=out, duration_s=0.0,
                    fps=0.0, error=err))
                continue
            fps, dur = _probe_video(out)
            results.append(LipSyncResult(
                model_name="wav2lip", out_video=out, duration_s=dur, fps=fps,
                extra={"licence": self.licence,
                       "checkpoint": str(self.checkpoint),
                       "warm_batch": len(requests)}))
        return results

