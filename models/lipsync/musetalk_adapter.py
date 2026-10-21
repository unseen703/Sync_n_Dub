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
import pickle
import tempfile
from pathlib import Path
from typing import Optional

from . import coords
from .base import BaseLipSync, LipSyncRequest, LipSyncResult, resolve_env_python

log = logging.getLogger(__name__)


DEFAULT_REPO = "third_party/MuseTalk"

#: Tallest frame handed to MuseTalk. Above this the input is downscaled
#: first, keeping aspect ratio. Default 1080 since 2026-10-20, by request.
#:
#: Three separate reasons, all measured on this host rather than assumed:
#:
#: 1. **RAM.** `scripts/inference.py` extracts every frame to PNG and then
#:    `read_imgs()` loads ALL of them back into a Python list, so resident
#:    memory is frames x width x height x 3. Measured slope at 1080p is
#:    6.14 MiB/frame against a 5.93 MiB theoretical frame -- i.e. one full
#:    uint8 frame retained per source frame, nothing else. Cost is linear
#:    in PIXELS, so 4K is 4x: a 60 s 4K clip measured 36.29 GiB RSS, and
#:    this host has 62 GiB total.
#: 2. **S3FD gives up on a large frame.** At 3840x2160 it returned `[None]`
#:    -- no detection at all -- and with zero detections
#:    `preprocessing.py:135` divides by zero inside a *print statement*,
#:    after the model has loaded. `ui/talking_head.py` already caps stills
#:    at `MAX_EDGE = 1280` for exactly this.
#: 3. **It costs nothing visible.** MuseTalk rebuilds the mouth at 256x256
#:    whatever the frame size, so above ~1080p the extra pixels are carried
#:    through the whole pipeline and then thrown at a 256x256 crop.
#:
#: The output is then 1080p where the source was larger, which is a real
#: change and is logged rather than left to be discovered.
MAX_RENDER_HEIGHT = 1080


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
        # 8, not upstream's 20. Lowered from 10 on 2026-10-20 by request.
        # Measured on this host: batch 20 peaked 15058 MiB of 16311 (1230
        # spare) and batch 15 peaked 14654 (1657 spare) -- both close enough
        # to the ceiling that a second tenant OOMs the render, which MuseTalk
        # swallows and reports only as "produced no mp4". This box IS shared:
        # another account held 4.2 GiB on gpu0 for hours on 2026-10-20 and
        # refused 30 of 32 segment renders. A smaller batch is the margin.
        #
        # Batch size is a throughput/VRAM knob and costs NO quality:
        # `datagen` is a pure chunker, each frame pairs with its own latent
        # by frame index, and the loop is `@torch.no_grad()` with the models
        # in `.eval()`. It also does not change the floor -- the ~9 GiB of
        # weights and the per-frame latent store dominate -- so this buys
        # headroom, not a different memory curve.
        batch_size: int = 8,
        fps: int = 25,
        bbox_shift: int = 0,
        use_float16: bool = False,
        # Copy MuseTalk's scratch dir out before the TemporaryDirectory is
        # destroyed. It holds `<task>.pkl` (the per-frame coord list, where
        # `(0.0,0.0,0.0,0.0)` marks a frame with no detected face) and, when
        # upstream has not yet deleted them, the composited PNGs. Without
        # this there is no way after the fact to say WHICH frames had no
        # face -- only that the count was wrong.
        keep_work_dir: bool = False,
        # Reuse per-frame face boxes across runs. MuseTalk supports this
        # natively (`--use_saved_coord` / `--saved_coord`) but writes the
        # pickle to `<result_dir>/../<basename>.pkl`, which for us is inside
        # a TemporaryDirectory and dies with it -- so the feature has never
        # actually done anything here. We stage a cached pickle in before
        # the run and copy the fresh one out after.
        use_coord_cache: bool = True,
        # Downscale anything taller than this before rendering. See
        # MAX_RENDER_HEIGHT. 0 disables the cap for a caller who really
        # wants the source resolution and has the RAM for it.
        max_height: int = MAX_RENDER_HEIGHT,
        cuda_device: Optional[int] = None,
    ):
        super().__init__(device=device, cuda_device=cuda_device)
        self.repo_path = Path(repo_path or os.getenv("MUSETALK_REPO") or DEFAULT_REPO)
        self.checkpoint = checkpoint or os.getenv("MUSETALK_CHECKPOINT")
        self.python_bin = resolve_env_python("musetalk", "musetalk", python_bin)
        self.version = (version or os.getenv("MUSETALK_VERSION") or "v15").lower()
        self.batch_size = batch_size
        self.fps = fps
        self.bbox_shift = bbox_shift
        self.max_height = int(max_height or 0)
        # Upstream halves pe / vae.vae / unet.model, then derives
        # `weight_dtype = unet.model.dtype` and casts whisper and every latent
        # batch to match (scripts/inference.py:52-56, 76, 201) -- so the half
        # path is internally consistent rather than a partial cast.
        # `MUSETALK_FP16=1` is the only way to reach this from
        # `backend/cli.py`, which has no --use-float16 flag — the driver
        # script and the launcher do, the pipeline does not. Same shape as
        # MUSETALK_REPO / MUSETALK_VERSION above. Worth reaching: measured
        # on this host, fp16 is -13% wall and -25% peak VRAM with the face
        # crop unchanged (34.45 vs 34.46 dB), per the `clip-run` skill.
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

    def _cap_resolution(self, video: Path, work: Path) -> Path:
        """Downscale to `self.max_height` if taller. Returns what to render.

        Returns the ORIGINAL path on any failure. A render at the source
        resolution is the old behaviour and still works for anything that
        fits; losing the whole job because a probe or a rescale failed
        would be the worse trade.

        Width is forced even (`-2`): H.264 4:2:0 cannot encode an odd
        dimension, and 2160 -> 1080 on a 1920-wide source is fine but an
        unusual aspect ratio lands on an odd width and ffmpeg refuses.
        """
        if self.max_height <= 0:
            return video
        from backend.media_io import _ffmpeg_bin, _ffprobe_bin
        ffprobe, ffmpeg = _ffprobe_bin(), _ffmpeg_bin()
        if not ffprobe or not ffmpeg:
            return video
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 str(video)], capture_output=True, text=True, timeout=60)
            w, h = (int(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])
        except Exception as e:                           # noqa: BLE001
            log.warning("musetalk: could not probe %s (%s); rendering at "
                        "source resolution", video.name, e)
            return video
        if h <= self.max_height:
            return video

        scaled = work / f"{video.stem}.{self.max_height}p.mp4"
        log.info("musetalk: downscaling %dx%d -> %dp before render "
                 "(MuseTalk holds every frame in RAM and rebuilds the mouth "
                 "at 256x256 regardless, so the extra pixels only cost "
                 "memory) -- OUTPUT WILL BE %dp, not %dp",
                 w, h, self.max_height, self.max_height, h)
        try:
            p = subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(video),
                 "-vf", f"scale=-2:{self.max_height}:flags=lanczos",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-c:a", "copy", str(scaled)],
                capture_output=True, text=True)
            if p.returncode != 0 or not scaled.is_file() or not scaled.stat().st_size:
                log.warning("musetalk: downscale failed (%s); rendering at "
                            "source resolution",
                            (p.stderr or "").strip()[-200:])
                return video
        except Exception as e:                           # noqa: BLE001
            log.warning("musetalk: downscale failed (%s); rendering at "
                        "source resolution", e)
            return video
        return scaled

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        self._ensure_loaded()
        out = Path(request.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        video_abs = Path(request.video).resolve()
        audio_abs = Path(request.audio).resolve()

        with tempfile.TemporaryDirectory(prefix="musetalk_") as td:
            tdp = Path(td)
            # BEFORE the yaml is written and before any coord-cache key is
            # derived: a cache entry keyed on the 4K file would carry boxes
            # in 4K coordinates into a 1080p render.
            video_abs = self._cap_resolution(video_abs, tdp)
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

            # MuseTalk derives the pickle path as `<result_dir>/../<stem>.pkl`,
            # i.e. `tdp/<video stem>.pkl`. Stage a hit there and it loads it
            # instead of running DWPose + S3FD over every frame.
            coord_params = {"bbox_shift": self.bbox_shift,
                            "version": self.version}
            # An ASD-gated target list beats both detection and the cache.
            # `backend.lipsync_targets` builds it from the tracks TalkNet
            # confirmed as speaking, so MuseTalk animates that face and
            # writes every other frame through untouched. Without it
            # MuseTalk picks whatever S3FD returned first (`d = d[0]`),
            # which on a multi-face shot is an arbitrary face.
            targets = (request.extra or {}).get("coord_list")
            # MuseTalk names the pickle with `basename.split('.')[0]`
            # (inference.py:99), which truncates at the FIRST dot -- our
            # inputs are `<stem>.dubbed.mp4`, so `Path.stem` gives
            # "x.dubbed" where MuseTalk writes "x". Using the wrong name
            # made the cache a silent no-op in both directions: nothing was
            # ever stored, and a staged hit was never read.
            staged = tdp / (video_abs.name.split(".")[0] + ".pkl")
            if targets:
                staged.write_bytes(pickle.dumps(list(targets)))
                cmd.append("--use_saved_coord")
                # Keep it, or inference.py deletes the file we just wrote.
                cmd.append("--saved_coord")
                log.info("musetalk: using %d supplied target boxes "
                         "(%d frames left untouched)", len(targets),
                         sum(1 for b in targets if tuple(b) == (0.0, 0.0, 0.0, 0.0)))
            elif self.use_coord_cache:
                cached = coords.load(video_abs, "musetalk", coord_params,
                                     expect_frames=_nb_frames(video_abs))
                if cached is not None:
                    staged.write_bytes(pickle.dumps(cached))
                    cmd.append("--use_saved_coord")
                # `--saved_coord` keeps the pickle so we can copy it out;
                # without it inference.py deletes it on the way out.
                cmd.append("--saved_coord")
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
                # rc == 0 and no file is NOT ambiguous here: MuseTalk wraps
                # the whole job in `try/except Exception` and merely PRINTS
                # ("Error occurred during processing:", scripts/inference.py
                # :288), so a real crash still exits 0. Without the captured
                # output the reason is lost entirely -- which cost a whole
                # 15-minute render on 2026-10-18, reported only as "produced
                # no mp4". The tail of stdout carries that printed line.
                tail = (proc.stdout or "").strip().splitlines()[-25:]
                swallowed = [l for l in tail
                             if "Error occurred during processing" in l]
                detail = swallowed[0] if swallowed else " | ".join(tail[-5:])
                return LipSyncResult(
                    model_name=f"musetalk-{self.version}", out_video=out,
                    duration_s=0.0, fps=0.0,
                    error=(f"musetalk exited 0 but produced no mp4 in "
                           f"{result_dir} — {detail or 'no output captured'}"),
                )
            # Prefer the file whose name matches the task name.
            match = [p for p in produced if task_name in p.name]
            src = match[0] if match else produced[0]
            shutil.copy2(src, out)

            if not targets and self.use_coord_cache and staged.is_file():
                try:
                    fresh = pickle.loads(staged.read_bytes())
                    coords.save(video_abs, "musetalk", coord_params, fresh,
                                frames=len(fresh))
                except Exception as e:                   # noqa: BLE001
                    log.warning("musetalk: could not cache coords (%s)", e)

            if self.keep_work_dir:
                kept = out.parent / f"{out.stem}.musetalk_work"
                try:
                    if kept.exists():
                        shutil.rmtree(kept)
                    # ignore the mp4s: they are already the output.
                    shutil.copytree(tdp, kept,
                                    ignore=shutil.ignore_patterns("*.mp4"))
                    log.info("musetalk: work dir kept at %s", kept)
                except Exception as e:                     # noqa: BLE001
                    # Debug aid only -- never fail a good render over it.
                    log.warning("musetalk: could not keep work dir (%s)", e)

        fps, dur = _probe_video(out)
        return LipSyncResult(
            model_name=f"musetalk-{self.version}", out_video=out,
            duration_s=dur, fps=fps,
            extra={"licence": self.licence, "version": self.version,
                   "use_float16": self.use_float16,
                   "batch_size": self.batch_size,
                   "keep_work_dir": self.keep_work_dir},
        )


def _nb_frames(path: Path) -> Optional[int]:
    """Frame count from container metadata. None when it is not recorded.

    Deliberately NOT `-count_frames`: that decodes the whole file, and these
    inputs are hundreds of megabytes -- paying a full decode to validate a
    cache would undo the saving the cache exists for. Our cuts are re-encoded
    by ffmpeg and do carry `nb_frames`; when a file does not, validation is
    skipped rather than guessed.
    """
    from backend.media_io import _ffprobe_bin
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
