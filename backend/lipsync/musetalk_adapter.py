"""MuseTalk 1.5 adapter (TMElyralab/MuseTalk).

Latent-space inpainting, not diffusion: frozen VAE + Whisper-tiny audio
encoder + SD-style UNet, rebuilding a 256x256 face region. MIT weights.

Upstream's `scripts/inference.py` takes a YAML listing (video, audio) pairs,
so the adapter writes a one-entry YAML, shells out, and copies the result to
the requested output path.

Env:
  MUSETALK_REPO       (default: models/lipsync/vendor/musetalk,
                       falling back to third_party/MuseTalk)
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
import pickle
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from backend.common.media_io import probe_video as _probe_video
from . import coords
from .base import (MAX_RENDER_HEIGHT, BaseLipSync, LipSyncRequest,
                   LipSyncResult, cap_resolution, resolve_env_python,
                   resolve_vendor_repo, run_streaming)

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/MuseTalk"

#: The cap itself is `base.MAX_RENDER_HEIGHT`, imported above and shared with
#: LatentSync since 2026-10-29. MuseTalk's own three reasons for it, all
#: measured on this host:
#:
#: 1. RAM. `read_imgs()` holds every extracted frame in a Python list, so RSS
#:    is linear in pixels -- ~6.1 MiB/frame at 1080p, 4x that at 4K. A 60 s
#:    4K clip measured 36.3 GiB on a 62 GiB host.
#: 2. S3FD returns `[None]` at 3840x2160, and zero detections make
#:    `preprocessing.py:135` divide by zero inside a print, after the model
#:    has loaded.
#: 3. No visible cost: the mouth is rebuilt at 256x256 whatever the frame
#:    size, so the extra pixels are carried through and then discarded.
_CAP_WHY = ("MuseTalk holds every frame in RAM and rebuilds the mouth at "
            "256x256 regardless, so the extra pixels only cost memory")


def _rescale_boxes(boxes, scale):
    """Move a supplied `coord_list` into the RENDERED frame's coordinates.

    `build_coord_list` reads the SOURCE video, but `_cap_resolution` may hand
    MuseTalk a smaller re-encode. An oversized box slices past the frame
    edge, numpy returns a 0-row array and `cv2.resize` raises -- after the
    model has loaded. The adapter owns the conversion because it is the layer
    that changed the frame size.

    `PLACEHOLDER` is matched before scaling and passed through bit-identical:
    it is MuseTalk's "write this frame unchanged" sentinel, not a box at the
    origin.
    """
    sx, sy = scale
    if (sx, sy) == (1.0, 1.0):
        return list(boxes)
    out = []
    for b in boxes:
        t = tuple(b)
        if t == (0.0, 0.0, 0.0, 0.0):
            out.append(t)
            continue
        x1, y1, x2, y2 = t
        out.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
    return out


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
        # None = scale from VRAM via backend.scaling.musetalk_batch_default()
        # (baseline 8 on 16 GiB -- upstream's 20 peaks at 15058 MiB and leaves
        # too little for a second tenant; an OOM here surfaces only as
        # "produced no mp4"). An explicit int wins and is passed through.
        # Batch size is pure throughput/VRAM and costs no quality -- `datagen`
        # is a chunker and each frame pairs with its own latent by index.
        batch_size: Optional[int] = None,
        fps: int = 25,
        bbox_shift: int = 0,
        use_float16: bool = False,
        # Copy MuseTalk's scratch dir out before the TemporaryDirectory is
        # destroyed. It holds `<task>.pkl`, the per-frame coord list where
        # `(0,0,0,0)` marks a frame with no detected face -- the only record
        # of WHICH frames were missed rather than just how many.
        keep_work_dir: bool = False,
        # Reuse per-frame face boxes across runs. MuseTalk's own
        # `--use_saved_coord` writes its pickle to
        # `<result_dir>/../<basename>.pkl`, inside our TemporaryDirectory, so
        # it never survives. Stage a cached pickle in, copy the fresh one out.
        use_coord_cache: bool = True,
        # Downscale anything taller than this before rendering; 0 disables
        # the cap. See MAX_RENDER_HEIGHT.
        max_height: int = MAX_RENDER_HEIGHT,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = resolve_vendor_repo(
            "musetalk", "MUSETALK_REPO", repo_path, legacy=DEFAULT_REPO)
        self.checkpoint = checkpoint or os.getenv("MUSETALK_CHECKPOINT")
        self.python_bin = resolve_env_python("musetalk", "musetalk", python_bin)
        self.version = (version or os.getenv("MUSETALK_VERSION") or "v15").lower()
        if batch_size is None:
            from backend.media.scaling import musetalk_batch_default
            batch_size = musetalk_batch_default()
        self.batch_size = int(batch_size)
        self.fps = fps
        self.bbox_shift = bbox_shift
        self.max_height = int(max_height or 0)
        # Upstream halves pe / vae.vae / unet.model and casts whisper and
        # every latent batch to match, so the half path is consistent rather
        # than a partial cast. `MUSETALK_FP16=1` is the only way in from
        # `backend/cli.py`, which has no --use-float16 flag. Measured here:
        # -13% wall, -25% peak VRAM, face crop unchanged (34.45 vs 34.46 dB).
        if not use_float16:
            use_float16 = (os.getenv("MUSETALK_FP16") or "").strip().lower() \
                in ("1", "true", "yes", "on")
        self.use_float16 = use_float16
        self.keep_work_dir = keep_work_dir
        self.use_coord_cache = use_coord_cache

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

    def _cap_resolution(self, video: Path, work: Path
                        ) -> Tuple[Path, Tuple[float, float]]:
        """Downscale to `self.max_height` if taller. See `base.cap_resolution`.

        Kept as a method because the returned `(sx, sy)` is what
        `_rescale_boxes` consumes a few lines later, and that pairing is the
        contract worth keeping visible in this file.
        """
        return cap_resolution(video, work, self.max_height,
                              tag="musetalk", why=_CAP_WHY, log=log)

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        return self.sync_many([request])[0]

    def sync_many(self, requests):
        """Render many (video, audio) pairs in ONE MuseTalk process.

        `scripts/inference.py` loads the VAE, UNet, Whisper and face parser
        once, then iterates every task in its YAML, releasing each task's
        frames and PNG dir before the next. Writing all pieces into one YAML
        therefore pays the ~9 GiB model load once while keeping peak RAM a
        function of a single piece. `backend.lipsync` hands one GPU's whole
        run of pieces to one call for this.

        Each task carries its own downscale factor, staged coord pickle and
        output. MuseTalk wraps each task in `try/except` and only prints, so
        a silently skipped task is detected by its missing output and
        reported against that request alone.
        """
        if not requests:
            return []
        self._ensure_loaded()

        with tempfile.TemporaryDirectory(prefix="musetalk_") as td:
            tdp = Path(td)
            result_dir = tdp / "results"
            result_dir.mkdir(exist_ok=True)

            # One record per request, built before the YAML so each task's
            # downscale and coord staging is settled up front.
            tasks = []
            any_staged = False           # at least one task has a pickle to load
            want_saved = False           # keep pickles (targets, or cache on)
            yaml_lines: list[str] = []
            for idx, request in enumerate(requests):
                out = Path(request.out_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                video_abs = Path(request.video).resolve()
                audio_abs = Path(request.audio).resolve()
                # Before the coord-cache key is derived: a key on the source
                # file would carry source-sized boxes into a capped render.
                video_abs, frame_scale = self._cap_resolution(video_abs, tdp)

                coord_params = {"bbox_shift": self.bbox_shift,
                                "version": self.version}
                # An ASD-gated target list beats both detection and the
                # cache: MuseTalk otherwise animates whatever S3FD returned
                # first (`d = d[0]`). See `backend.lipsync_targets`.
                targets = (request.extra or {}).get("coord_list")
                # MuseTalk names the pickle `basename.split('.')[0] + ".pkl"`
                # (inference.py:234), truncating at the first dot, into `tdp`.
                # Piece basenames (part00, part01, ...) keep them distinct.
                staged = tdp / (video_abs.name.split(".")[0] + ".pkl")
                if targets:
                    targets = _rescale_boxes(targets, frame_scale)
                    staged.write_bytes(pickle.dumps(list(targets)))
                    any_staged = True
                    want_saved = True
                    log.info("musetalk: task %d uses %d supplied target boxes "
                             "(%d frames left untouched%s)", idx, len(targets),
                             sum(1 for b in targets if tuple(b) == (0.0, 0.0, 0.0, 0.0)),
                             "" if frame_scale == (1.0, 1.0) else
                             f", rescaled x{frame_scale[0]:.4f}/{frame_scale[1]:.4f} "
                             f"for the downscaled frame")
                elif self.use_coord_cache:
                    want_saved = True
                    cached = coords.load(video_abs, "musetalk", coord_params,
                                         expect_frames=_nb_frames(video_abs))
                    if cached is not None:
                        staged.write_bytes(pickle.dumps(cached))
                        any_staged = True

                # Unique YAML key and result_name per task, so two tasks
                # cannot write the same file in the shared `temp_dir`.
                key = f"t{idx:03d}_{out.stem}"
                result_name = f"{key}.mp4"
                yaml_lines.append(
                    f"{key}:\n"
                    f"  video_path: \"{video_abs}\"\n"
                    f"  audio_path: \"{audio_abs}\"\n"
                    f"  bbox_shift: {self.bbox_shift}\n"
                    f"  result_name: \"{result_name}\"\n"
                )
                tasks.append({"request": request, "out": out,
                              "video_abs": video_abs, "targets": bool(targets),
                              "staged": staged, "coord_params": coord_params,
                              "result_name": result_name})

            yaml_path = tdp / "tasks.yaml"
            yaml_path.write_text("".join(yaml_lines))

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
            if any_staged:
                cmd.append("--use_saved_coord")
            if want_saved:
                # Keep every task's pickle for the cache below; inference.py
                # deletes them on the way out otherwise.
                cmd.append("--saved_coord")

            log.info("musetalk[gpu=%s]: %d task(s): %s",
                     self.cuda_device, len(tasks), " ".join(cmd))
            env = dict(self._subprocess_env())
            # Lazy decode saves ~17 GiB RSS on a 2-min 1080p clip at the same
            # wall time and bit-identical output, but only on the
            # --use_saved_coord branch (inference.py:286) -- the cold branch
            # calls get_landmark_and_bbox and loads every frame anyway. An
            # operator-set value wins.
            if "--use_saved_coord" in cmd and "MUSETALK_LAZY_FRAMES" not in env:
                env["MUSETALK_LAZY_FRAMES"] = "1"
            # Stream live: MuseTalk prints `Results saved to <file>` per
            # task, so forwarding gives per-piece progress in run.log.
            proc = run_streaming(
                cmd, cwd=str(self.repo_path), env=env, log=log,
                tag=f"musetalk[gpu={self.cuda_device}]",
                progress_prefixes=("Results saved to", "[memguard]",
                                   "Error occurred during processing"))

            # A non-zero exit means the whole batch failed -- the model load
            # is shared -- so report the same error against every request.
            if proc.returncode != 0:
                err = f"musetalk rc={proc.returncode}: {proc.stdout.strip()[-3000:]}"
                return [LipSyncResult(model_name=f"musetalk-{self.version}",
                                      out_video=t["out"], duration_s=0.0,
                                      fps=0.0, error=err) for t in tasks]

            # MuseTalk writes each task to `<result_dir>/<version>/<result_name>`.
            temp_dir = result_dir / self.version
            tail = (proc.stdout or "").strip().splitlines()[-40:]
            swallowed = [l for l in tail
                         if "Error occurred during processing" in l]

            results = []
            for t in tasks:
                out = t["out"]
                produced = temp_dir / t["result_name"]
                if not produced.is_file():
                    # rc == 0 with no file for THIS task: MuseTalk catches
                    # per-task exceptions and only prints them, so a crash
                    # still exits 0. Attribute the printed message to the
                    # task that lost its output.
                    detail = (swallowed[0] if swallowed
                              else " | ".join(tail[-5:]) or "no output captured")
                    results.append(LipSyncResult(
                        model_name=f"musetalk-{self.version}", out_video=out,
                        duration_s=0.0, fps=0.0,
                        error=(f"musetalk exited 0 but produced no mp4 for "
                               f"{out.name} in {temp_dir} — {detail}")))
                    continue
                shutil.copy2(produced, out)

                if not t["targets"] and self.use_coord_cache \
                        and t["staged"].is_file():
                    try:
                        fresh = pickle.loads(t["staged"].read_bytes())
                        coords.save(t["video_abs"], "musetalk",
                                    t["coord_params"], fresh, frames=len(fresh))
                    except Exception as e:               # noqa: BLE001
                        log.warning("musetalk: could not cache coords (%s)", e)

                fps, dur = _probe_video(out)
                results.append(LipSyncResult(
                    model_name=f"musetalk-{self.version}", out_video=out,
                    duration_s=dur, fps=fps,
                    extra={"licence": self.licence, "version": self.version,
                           "use_float16": self.use_float16,
                           "batch_size": self.batch_size,
                           "keep_work_dir": self.keep_work_dir}))

            if self.keep_work_dir:
                kept = tasks[0]["out"].parent / f"{tasks[0]['out'].stem}.musetalk_work"
                try:
                    if kept.exists():
                        shutil.rmtree(kept)
                    shutil.copytree(tdp, kept,
                                    ignore=shutil.ignore_patterns("*.mp4"))
                    log.info("musetalk: work dir kept at %s", kept)
                except Exception as e:                   # noqa: BLE001
                    log.warning("musetalk: could not keep work dir (%s)", e)

        return results


def _nb_frames(path: Path) -> Optional[int]:
    """Frame count from container metadata. None when it is not recorded.

    Not `-count_frames`: a full decode of these inputs would cost more than
    the cache it validates saves. ffmpeg-re-encoded cuts carry `nb_frames`;
    when a file does not, validation is skipped rather than guessed.
    """
    from backend.common.media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=60)
        first = (r.stdout or "").strip().splitlines()[0].split(",")[0]
        return int(first) if first.isdigit() else None
    except Exception:                                    # noqa: BLE001
        return None

