"""Shared types and abstract base for lip-sync adapters.

An adapter takes a video and a driving audio track (the dubbed audio from
the mix stage) and re-renders the on-screen mouths to match it. Only the
lower-face region changes.

Each adapter wraps an external upstream clone (see `backend/lipsync/vendor/`).
None installs from PyPI, so the adapter clones the repo and loads its
checkpoints via env vars.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


def resolve_env_python(model: str, env_name: str,
                       explicit: Optional[str] = None,
                       *, legacy_env_names: Sequence[str] = ()) -> str:
    """Interpreter for an adapter that cannot run in the `dubbing` env.

    MuseTalk and LatentSync have upstream pins incompatible with `dubbing`,
    so they have their own conda envs.

    Order: explicit argument -> `<MODEL>_PYTHON` -> sibling conda env ->
    any `legacy_env_names` -> `sys.executable` (warned). The sibling env is
    tried before `sys.executable` because falling back to `dubbing` runs
    MuseTalk where `import diffusers` fails -- after ASR, MT, TTS and the
    mix are paid for, with the adapter still logging "done".

    `legacy_env_names` covers an env being renamed on disk: the new name is
    preferred, the old one still works, and a run between the code change
    and the `conda rename` does not land on the wrong interpreter. The
    fallback LOGS, so a checkout still on the old name says so rather than
    going quiet.

    The sibling path is derived from `sys.executable`
    (`<prefix>/envs/<env_name>/bin/python`), not hard-coded.
    """
    import logging
    import os
    import sys

    log = logging.getLogger(__name__)
    var = f"{model.upper()}_PYTHON"

    for value, source in ((explicit, "argument"), (os.getenv(var), var)):
        if value and str(value).strip():
            return str(value).strip()

    try:
        envs_dir = Path(sys.executable).resolve().parents[2]
        sibling = envs_dir / env_name / "bin" / "python"
        if sibling.is_file():
            log.info("%s: using sibling env interpreter %s (set %s to override)",
                     model, sibling, var)
            return str(sibling)
        for legacy in legacy_env_names:
            old = envs_dir / legacy / "bin" / "python"
            if old.is_file():
                log.warning(
                    "%s: conda env %r not found, falling back to the former "
                    "name %r at %s. Rename it (`conda rename -n %s %s`) or "
                    "set %s.", model, env_name, legacy, old, legacy,
                    env_name, var)
                return str(old)
    except (IndexError, OSError):
        pass

    log.warning(
        "%s: no %s and no sibling '%s' conda env; falling back to %s. This "
        "interpreter is very likely missing %s's dependencies, and the "
        "failure lands at the lip-sync stage after the rest of the run has "
        "already been paid for.",
        model, var, env_name, sys.executable, model,
    )
    return sys.executable


@dataclass
class LipSyncRequest:
    """One (video, audio) pair to run through a lip-sync model.

    `video` is the source; `audio` is the driving track, which in the
    pipeline is the mixed dubbed audio from `backend.mix`.
    """
    video: Path
    audio: Path
    out_path: Path
    face_bbox: Optional[tuple] = None    # optional (x1,y1,x2,y2) pre-detected
    extra: dict = field(default_factory=dict)


@dataclass
class LipSyncResult:
    model_name: str
    out_video: Path
    duration_s: float
    fps: float
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "model_name": self.model_name,
            "out_video": str(self.out_video),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "error": self.error,
            "extra": self.extra,
        }


class BaseLipSync(ABC):
    name: str = "base"
    licence: str = "unknown"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from backend.diarize.base import _resolve_device
        self.device = device or _resolve_device()
        # Physical CUDA id to pin the adapter's subprocess to via
        # CUDA_VISIBLE_DEVICES. None = inherit parent env (default GPU).
        self.cuda_device: Optional[int] = kwargs.get("cuda_device")
        self._loaded = False

    def _subprocess_env(self) -> dict:
        """Env for the adapter's shell-out, pinned to self.cuda_device if set.

        Also puts our ffmpeg on PATH. Every vendored upstream shells out to
        a bare `ffmpeg` somewhere and there is none on this host's PATH;
        Wav2Lip's `inference.py` discards the return code, so a missing
        ffmpeg loses the render after the GPU work and still exits 0.
        """
        import os
        from pathlib import Path

        env = os.environ.copy()
        if self.cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
        try:
            from backend.common.media_io import _ffmpeg_bin
            ffdir = str(Path(_ffmpeg_bin()).parent)
            if ffdir not in env.get("PATH", "").split(os.pathsep):
                env["PATH"] = ffdir + os.pathsep + env.get("PATH", "")
        except Exception:                 # no ffmpeg at all — the adapter reports it
            pass
        return env

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def sync(self, request: LipSyncRequest) -> LipSyncResult:
        """Render a lip-synced video. Returns metadata about the output."""

    def sync_many(self, requests: "list[LipSyncRequest]") -> "list[LipSyncResult]":
        """Render several (video, audio) pairs reusing one loaded model.

        `backend.lipsync` splits a long clip into pieces to bound host RAM;
        handing a GPU's whole run of pieces to one call lets an adapter load
        the model once and cycle only the per-piece frame buffers.

        Default: one `sync()` per request, so an adapter with no warm path
        still works. The returned list matches `requests` in order and
        length; a failed request carries its error in its own
        `LipSyncResult.error` rather than aborting the rest.
        """
        return [self.sync(r) for r in requests]


#: Tallest frame handed to a lip-sync model; taller input is downscaled
#: first, keeping aspect ratio, and that is logged. Every model here rebuilds
#: a small fixed region -- Wav2Lip 96x96, MuseTalk and LatentSync 256x256 --
#: so the extra pixels are carried through the whole render and then
#: discarded. What they are not is free: both upstreams hold frames in RAM,
#: so cost is linear in pixels and a 4K frame is 4x a 1080p one.
MAX_RENDER_HEIGHT = 1080


def cap_resolution(video, work, max_height: int = MAX_RENDER_HEIGHT, *,
                   tag: str = "lipsync", why: str = "", log=None):
    """Downscale `video` to `max_height` if taller, into `work`.

    Returns `(what to render, (sx, sy))`, the factors mapping SOURCE pixel
    coordinates onto the rendered frame. `(1.0, 1.0)` whenever nothing was
    rescaled, including every failure path -- a caller holding source-space
    coordinates must apply them, or its boxes slice past the frame edge and
    `cv2.resize` raises after the model has loaded.

    Returns the original path on any failure, so a probe or rescale error
    degrades to a source-resolution render rather than losing the job. Width
    is forced even (`-2`) because H.264 4:2:0 cannot encode an odd dimension.

    Shared rather than per-adapter: MuseTalk has capped since the 4K S3FD
    divide-by-zero, LatentSync was capped on 2026-10-29 for the same RAM
    reason, and a second copy of this would be a second place for the
    coordinate-space contract to drift.
    """
    import logging
    import subprocess
    from pathlib import Path as _Path

    log = log or logging.getLogger(__name__)
    video = _Path(video)
    unscaled = (1.0, 1.0)
    if int(max_height or 0) <= 0:
        return video, unscaled
    max_height = int(max_height)

    from backend.common.media_io import _ffmpeg_bin, _ffprobe_bin
    ffprobe, ffmpeg = _ffprobe_bin(), _ffmpeg_bin()
    if not ffprobe or not ffmpeg:
        return video, unscaled

    def _dims(path):
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=60)
        return (int(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])

    try:
        w, h = _dims(video)
    except Exception as e:                                   # noqa: BLE001
        log.warning("%s: could not probe %s (%s); rendering at source "
                    "resolution", tag, video.name, e)
        return video, unscaled
    if h <= max_height:
        return video, unscaled

    scaled = _Path(work) / f"{video.stem}.{max_height}p.mp4"
    log.info("%s: downscaling %dx%d -> %dp before render%s -- OUTPUT WILL BE "
             "%dp, not %dp", tag, w, h, max_height,
             f" ({why})" if why else "", max_height, h)
    try:
        p = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(video),
             "-vf", f"scale=-2:{max_height}:flags=lanczos",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p",
             # FRAME PARITY. Without this the re-encode is free to add or
             # drop frames to make a CFR stream: measured here, a 33-frame
             # 4K clip whose audio runs 38 ms past its video came back as 34.
             # One frame in, one frame out is what the whole segment/join
             # path downstream assumes. `-vsync 0` rather than `-fps_mode
             # passthrough` because `media_io._ffmpeg_bin()` can resolve
             # ffmpeg 4.4, which predates `-fps_mode`.
             "-vsync", "0",
             "-c:a", "copy", str(scaled)],
            capture_output=True, text=True)
        if p.returncode != 0 or not scaled.is_file() or not scaled.stat().st_size:
            log.warning("%s: downscale failed (%s); rendering at source "
                        "resolution", tag, (p.stderr or "").strip()[-200:])
            return video, unscaled
    except Exception as e:                                   # noqa: BLE001
        log.warning("%s: downscale failed (%s); rendering at source "
                    "resolution", tag, e)
        return video, unscaled

    # Measure the factors off the ENCODED file, not from `max_height`: `-2`
    # rounds width to even, so a computed factor drifts the boxes sideways on
    # an odd-width source.
    try:
        nw, nh = _dims(scaled)
    except Exception as e:                                   # noqa: BLE001
        # Keep the rendered file, but refuse to guess the factor: a guessed
        # one puts oversized boxes on a smaller frame.
        raise RuntimeError(
            f"{tag}: downscaled {video.name} but could not probe the result "
            f"to convert target boxes ({e})") from e
    return scaled, (nw / w, nh / h)


def run_streaming(cmd, *, cwd=None, env=None, log=None, tag="",
                  progress_prefixes=("[worker]",), tail_lines=500):
    """Run `cmd`, forward its progress lines to `log` LIVE, return a
    CompletedProcess whose `stdout` holds the last `tail_lines` of combined
    output (stderr merged in) for error reporting. `stderr` is "".

    Live streaming is the point. `subprocess.run(capture_output=True)` buffers
    all child output until the process exits, so a warm worker rendering N
    pieces in one call showed NOTHING in the log until every piece was done --
    the per-segment progress that the old one-process-per-segment path printed
    for free disappeared the moment the pieces were batched. Reading the pipe
    line by line and forwarding the lines that start with a `progress_prefixes`
    entry (the drivers' `[worker] ...` lines, MuseTalk's `Results saved to`)
    puts each piece back in run.log as it happens.

    stderr is merged into stdout so a failing child's traceback is captured in
    the returned tail; tqdm bars use `\\r` without `\\n`, so they do not stream
    as spam -- only newline-terminated status lines do.
    """
    import collections
    import subprocess

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, text=True, bufsize=1, errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf: "collections.deque" = collections.deque(maxlen=tail_lines)
    if proc.stdout is not None:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            buf.append(line)
            if log is not None and tag and \
                    any(line.startswith(p) for p in progress_prefixes):
                log.info("%s: %s", tag, line)
        proc.stdout.close()
    rc = proc.wait()
    return subprocess.CompletedProcess(cmd, rc, "\n".join(buf), "")


# `VENDOR_ROOT` / `resolve_vendor_repo` moved to `models/_vendor.py` on
# 2026-10-23: TalkNet (models/asd/) needs the same resolution, and importing
# the lip-sync base from the ASD package to get it would be the wrong
# dependency entirely. Re-exported here so existing imports keep working.
from backend.common.vendor import VENDOR_ROOT, resolve_vendor_repo  # noqa: E402,F401
