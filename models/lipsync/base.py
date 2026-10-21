"""Shared types + abstract base for lip-sync adapters.

A lip-sync adapter takes a video file and a *driving* audio track (the
dubbed audio from the mix stage) and produces a new video where the
on-screen faces are animated to match the driving audio. Content stays
the same; only the mouth / lower-face region is re-rendered.

The three adapters we ship — Wav2Lip, MuseTalk 1.5, LatentSync 1.6 —
each wrap an external third-party repo (see `third_party/`). None of
them have a usable PyPI install; the adapter is responsible for cloning
the repo and loading its checkpoints via env vars.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def resolve_env_python(model: str, env_name: str,
                       explicit: Optional[str] = None) -> str:
    """Interpreter for an adapter that CANNOT run in the `dubbing` env.

    MuseTalk and LatentSync have upstream pins incompatible with `dubbing`,
    which is why they have their own conda envs. The adapters used to resolve
    `<MODEL>_PYTHON or sys.executable`, and an empty value is the normal case:
    neither variable was in `.env.example` (added 2026-10-17), and
    `LIPGAN_PYTHON` / `WAV2LIP_PYTHON` being legitimately blank makes a blank
    look safe. The fallback then ran MuseTalk under `dubbing`, where
    `import diffusers` fails -- AFTER a full run has paid for ASR, MT, TTS and
    the mix, with the adapter still logging "done" and the manifest recording
    an output path that does not exist.

    So the sibling conda env is tried BEFORE `sys.executable`: `dubbing`'s
    interpreter lives at `<prefix>/envs/dubbing/bin/python`, so a sibling is
    `<prefix>/envs/<env_name>/bin/python`. Derived from `sys.executable`
    rather than hard-coded, which is the same trick `media_io._ffmpeg_bin()`
    uses to find the env actually executing. Resolving it here fixes the CLI,
    the UI runner, the edit agent and `experiment.py` at once, instead of
    each launcher remembering to export a variable.

    Order: explicit argument -> `<MODEL>_PYTHON` -> sibling env ->
    `sys.executable` (warned, because that is the shape that failed).
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

    `video` is the source video (we re-render the lower face region on
    top of its original frames). `audio` is the driving track — in the
    pipeline this is the mixed dubbed audio from `backend.mix`.
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

        Also puts our ffmpeg on PATH. Every one of these upstream repos
        shells out to the bare name `ffmpeg` somewhere, and there is none on
        this host's PATH (CLAUDE.md §1) — Wav2Lip's `inference.py:289` muxes
        its `temp/result.avi` onto the output that way and discards the
        return code, so a missing ffmpeg costs the whole render *after* the
        GPU work is finished, and `inference.py` still exits 0 with no
        output file. Prepending the directory here fixes it once for every
        adapter instead of once per runner script.
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
