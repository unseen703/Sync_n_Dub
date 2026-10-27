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

from backend.media_io import probe_video as _probe_video
from . import coords
from .base import (BaseLipSync, LipSyncRequest, LipSyncResult,
                   resolve_env_python, resolve_vendor_repo, run_streaming)

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


def _rescale_boxes(boxes, scale):
    """Move a supplied `coord_list` into the RENDERED frame's coordinates.

    `backend.lipsync_targets.build_coord_list` reads the SOURCE video, so its
    boxes are clamped to the source frame -- 3840x2160 on the IIT lecture.
    `_cap_resolution` then hands MuseTalk a 1920x1080 re-encode, and
    `scripts/inference.py` slices `frame[y1:y2, x1:x2]` with whatever it was
    given. A 4K box on a 1080p frame slices past the bottom edge, numpy
    returns a 0-row array, and `cv2.resize` raises
    `(-215:Assertion failed) !ssize.empty()` -- *after* the model has loaded
    and the frames have been extracted. Measured: 87 s to fail.

    So the adapter owns this conversion, because the adapter is the layer
    that changed the frame size. The alternative -- having the caller build
    the list against the downscaled file -- would mean every caller
    re-deriving a cap that lives here.

    `PLACEHOLDER` is matched BEFORE scaling and passed through untouched:
    it is MuseTalk's `coord_placeholder` sentinel meaning "write this frame
    unchanged", not a box at the origin, and it has to stay bit-identical to
    be recognised.
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
        self.repo_path = resolve_vendor_repo(
            "musetalk", "MUSETALK_REPO", repo_path, legacy=DEFAULT_REPO)
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

    def _cap_resolution(self, video: Path, work: Path
                        ) -> Tuple[Path, Tuple[float, float]]:
        """Downscale to `self.max_height` if taller.

        Returns `(what to render, (sx, sy))`, where the factors map SOURCE
        pixel coordinates onto the rendered frame. They are `(1.0, 1.0)`
        whenever nothing was rescaled, which includes every failure path.
        Callers that hold coordinates in source space -- the supplied
        `coord_list` -- must apply them; see `_rescale_boxes`.

        Returns the ORIGINAL path on any failure. A render at the source
        resolution is the old behaviour and still works for anything that
        fits; losing the whole job because a probe or a rescale failed
        would be the worse trade.

        Width is forced even (`-2`): H.264 4:2:0 cannot encode an odd
        dimension, and 2160 -> 1080 on a 1920-wide source is fine but an
        unusual aspect ratio lands on an odd width and ffmpeg refuses.
        """
        unscaled = (1.0, 1.0)
        if self.max_height <= 0:
            return video, unscaled
        from backend.media_io import _ffmpeg_bin, _ffprobe_bin
        ffprobe, ffmpeg = _ffprobe_bin(), _ffmpeg_bin()
        if not ffprobe or not ffmpeg:
            return video, unscaled
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 str(video)], capture_output=True, text=True, timeout=60)
            w, h = (int(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])
        except Exception as e:                           # noqa: BLE001
            log.warning("musetalk: could not probe %s (%s); rendering at "
                        "source resolution", video.name, e)
            return video, unscaled
        if h <= self.max_height:
            return video, unscaled

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
                return video, unscaled
        except Exception as e:                           # noqa: BLE001
            log.warning("musetalk: downscale failed (%s); rendering at "
                        "source resolution", e)
            return video, unscaled

        # The factors are measured off the ENCODED file rather than computed
        # from `max_height`, because `-2` rounds the width to an even number:
        # a 1998-wide source lands on 998, not 999, and a factor that assumed
        # otherwise would drift the boxes sideways.
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 str(scaled)], capture_output=True, text=True, timeout=60)
            nw, nh = (int(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])
        except Exception as e:                           # noqa: BLE001
            # The file rendered, so keep it -- but a supplied coord list
            # cannot be converted without the factor, and guessing it is how
            # the 4K-boxes-on-a-1080p-frame crash happened. Refuse instead.
            raise RuntimeError(
                f"musetalk: downscaled {video.name} but could not probe the "
                f"result to convert target boxes ({e})") from e
        return scaled, (nw / w, nh / h)

    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        return self.sync_many([request])[0]

    def sync_many(self, requests):
        """Render many (video, audio) pairs in ONE MuseTalk process.

        MuseTalk's `scripts/inference.py` already loads the VAE, UNet, Whisper
        and face parser ONCE (before the task loop) and then iterates every
        task in its YAML -- its per-task `frame_list` is a local that is
        released on the next iteration's reassignment, and it `rmtree`s the
        per-task PNG dirs between tasks. So the only thing standing between us
        and "load once, cycle frames" was that the adapter wrote a ONE-entry
        YAML and spawned a fresh process per piece. Writing all the pieces
        into one YAML keeps peak RAM a function of a single piece (the loop is
        still sequential) while paying the ~9 GiB model load exactly once for
        the whole batch. `backend.lipsync` hands one GPU's whole run of pieces
        to one call for this reason.

        Each task carries its own downscale factor, its own staged coord
        pickle, and its own output; a task that MuseTalk silently skips (it
        wraps each in `try/except` and only prints) is detected by its missing
        output and reported in that request's own result, not allowed to abort
        the siblings.
        """
        if not requests:
            return []
        self._ensure_loaded()

        with tempfile.TemporaryDirectory(prefix="musetalk_") as td:
            tdp = Path(td)
            result_dir = tdp / "results"
            result_dir.mkdir(exist_ok=True)

            # One record per request, built before the YAML so every task's
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
                # BEFORE the coord-cache key is derived: a key on the 4K file
                # would carry boxes in 4K coordinates into a 1080p render.
                video_abs, frame_scale = self._cap_resolution(video_abs, tdp)

                coord_params = {"bbox_shift": self.bbox_shift,
                                "version": self.version}
                # An ASD-gated target list beats both detection and the cache.
                # Built by `backend.lipsync_targets` from the tracks TalkNet
                # confirmed speaking, so MuseTalk animates that face and writes
                # every other frame through untouched; without it MuseTalk
                # takes whatever S3FD returned first (`d = d[0]`).
                targets = (request.extra or {}).get("coord_list")
                # MuseTalk names the pickle `basename.split('.')[0] + ".pkl"`
                # (inference.py:234), truncating at the FIRST dot, and writes it
                # to `<result_dir>/../<name>.pkl` i.e. here in `tdp`. Each
                # piece's video has a distinct basename (part00, part01, ...),
                # so the pickles do not collide across tasks.
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

                # A unique YAML key AND a unique result_name per task, so the
                # shared `temp_dir` cannot have two tasks write the same file.
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
                # Keeps every task's pickle so a freshly-detected one can be
                # copied into the cache below; without it inference.py deletes
                # them on the way out.
                cmd.append("--saved_coord")

            log.info("musetalk[gpu=%s]: %d task(s): %s",
                     self.cuda_device, len(tasks), " ".join(cmd))
            env = dict(self._subprocess_env())
            # Lazy decode saves ~17 GiB RSS on a 2-min 1080p clip at identical
            # wall time with bit-identical output -- but ONLY on the
            # --use_saved_coord branch (vendor/musetalk/scripts/inference.py:286).
            # The cold branch calls get_landmark_and_bbox, which loads every
            # frame anyway. An operator-set value wins; see
            # scripts/ab_lazy_frames.py.
            if "--use_saved_coord" in cmd and "MUSETALK_LAZY_FRAMES" not in env:
                env["MUSETALK_LAZY_FRAMES"] = "1"
            # Stream live: MuseTalk prints `Results saved to <file>` as each
            # task finishes, so forwarding those lines puts per-piece progress
            # in run.log instead of one silent block until the batch is done.
            proc = run_streaming(
                cmd, cwd=str(self.repo_path), env=env, log=log,
                tag=f"musetalk[gpu={self.cuda_device}]",
                progress_prefixes=("Results saved to", "[memguard]",
                                   "Error occurred during processing"))

            # A non-zero exit is the WHOLE batch failing (an import error, a
            # dead interpreter) -- the model load is shared, so it takes every
            # task with it. Report the same error against each request.
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
                    # rc == 0 and no file for THIS task: MuseTalk wraps each
                    # task in `try/except Exception` and only PRINTS
                    # ("Error occurred during processing:"), so a crash on one
                    # task still exits 0 and the siblings carry on. Attribute
                    # the swallowed message to the task that lost its output.
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

