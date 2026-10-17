"""Shared types + abstract base for face-restoration adapters.

A restoration adapter is a *post-process* on a lip-synced video. The
lip-sync models re-render the lower face at a fixed, small working
resolution — Wav2Lip paints a 96x96 mouth crop back onto the frame, and
at `resize_factor=2` the frame itself is already halved — so the mouth
region comes back blurrier than the face around it. A restoration model
re-detects the face, rebuilds it at 512x512, and pastes it back, which
is what Wav2Lip-GFPGAN and Wav2Lip-CodeFormer both do.

The contract is deliberately narrower than lip-sync: video in, video
out, no driving audio. Audio is carried through (or re-muxed from an
explicitly supplied track) so the stage is drop-in after
`backend.lipsync.run_lipsync`.

Like the lip-sync adapters, each implementation wraps an external repo
clone under `third_party/` and loads its checkpoints via env vars.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RestoreRequest:
    """One video to run through a face-restoration model.

    `video` is the lip-synced render. `audio`, when given, replaces the
    video's own audio track in the output — pass the driving audio so a
    silent or 16 kHz-mono adapter output comes back with the dubbed
    track at a profile players actually accept (see
    `backend.lipsync._needs_remux`).
    """
    video: Path
    out_path: Path
    audio: Optional[Path] = None
    extra: dict = field(default_factory=dict)


@dataclass
class RestoreResult:
    model_name: str
    out_video: Path
    duration_s: float
    fps: float
    frames: int = 0
    # Frames in which the detector found at least one face. A restoration
    # run that reports 0 here did nothing at all, and will otherwise look
    # like a success — see the "verify semantically" guardrail in CLAUDE.md.
    frames_with_faces: int = 0
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "model_name": self.model_name,
            "out_video": str(self.out_video),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "frames": self.frames,
            "frames_with_faces": self.frames_with_faces,
            "error": self.error,
            "extra": self.extra,
        }


class BaseRestore(ABC):
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

        Also preloads the conda libstdc++. The restoration models all run a
        face detector, so they hit the `cv2` / `libtbb` CXXABI_1.3.15 trap
        described in CLAUDE.md §1 — whichever libstdc++ loads first decides
        whether the import works at all.
        """
        import os
        from pathlib import Path

        env = os.environ.copy()
        if self.cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)
        env.setdefault(
            "LD_PRELOAD",
            "/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6",
        )
        # And our ffmpeg on PATH — these repos shell out to the bare name and
        # there is none on this host's PATH. See BaseLipSync._subprocess_env.
        try:
            from backend.media_io import _ffmpeg_bin
            ffdir = str(Path(_ffmpeg_bin()).parent)
            if ffdir not in env.get("PATH", "").split(os.pathsep):
                env["PATH"] = ffdir + os.pathsep + env.get("PATH", "")
        except Exception:
            pass
        return env

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def restore(self, request: RestoreRequest) -> RestoreResult:
        """Re-render the faces in a video at higher fidelity."""
