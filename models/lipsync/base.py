"""Shared types and abstract base for lip-sync adapters.

An adapter takes a video and a driving audio track (the dubbed audio from
the mix stage) and re-renders the on-screen mouths to match it. Only the
lower-face region changes.

Each adapter wraps an external upstream clone (see `models/lipsync/vendor/`).
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
        from models.diarization.base import _resolve_device
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
            from backend.media_io import _ffmpeg_bin
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
from .._vendor import VENDOR_ROOT, resolve_vendor_repo  # noqa: E402,F401
