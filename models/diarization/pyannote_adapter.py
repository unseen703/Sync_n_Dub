"""pyannote community-1 diarization adapter.

Model card: https://hf.co/pyannote/speaker-diarization-community-1
Needs an HF token AND the gate accepted from the model page.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseDiarizer, DiarizationResult, Turn, canonicalize_speaker_labels


# Regex to extract the SONAME an auditwheel-hashed lib satisfies. E.g.
# `libavutil-a5b94d28.so.59.39.100` → `libavutil.so.59`.
_HASHED_LIB_RE = re.compile(r"^(?P<stem>lib[^/]+?)-[0-9a-f]{8}\.so\.(?P<major>\d+)")


def _find_av_libs_dir() -> Optional[Path]:
    for site in sys.path:
        cand = Path(site) / "av.libs"
        if cand.is_dir():
            return cand
    return None


_RELAUNCH_ENV_VAR = "DUBBING_FFMPEG_SHIM_APPLIED"


def _build_ffmpeg_soname_shim() -> Optional[str]:
    """Symlink unhashed sonames (`libavutil.so.59`) to the auditwheel-hashed
    files PyAV ships in `<site-packages>/av.libs/`.

    torchcodec's native lib DT_NEEDEDs the unhashed names, which PyAV's hashed
    filenames break. Returns the shim directory, or None without PyAV.
    """
    av_libs = _find_av_libs_dir()
    if av_libs is None:
        return None
    shim = Path(tempfile.gettempdir()) / "dubbingstudio-ffmpeg-shim" / "lib"
    shim.mkdir(parents=True, exist_ok=True)
    for f in av_libs.iterdir():
        m = _HASHED_LIB_RE.match(f.name)
        if not m:
            continue
        target = shim / f"{m.group('stem')}.so.{m.group('major')}"
        if target.is_symlink() or target.exists():
            continue
        try:
            target.symlink_to(f)
        except OSError:
            pass
    return str(shim)


def ensure_ffmpeg_libs_on_ld_path_or_reexec() -> None:
    """Point LD_LIBRARY_PATH at the ffmpeg soname shim plus PyAV's libs.

    glibc caches LD_LIBRARY_PATH at process startup, so mutating `os.environ`
    does not affect later dlopen() calls — on a box with no system ffmpeg this
    sets the vars and re-execs with the same argv. A sentinel env var stops
    the loop after one hop.
    """
    if os.environ.get(_RELAUNCH_ENV_VAR):
        return
    # No re-exec needed when the system ffmpeg libs are present.
    import ctypes.util
    if ctypes.util.find_library("avutil"):
        os.environ[_RELAUNCH_ENV_VAR] = "1"
        return
    shim = _build_ffmpeg_soname_shim()
    if shim is None:
        # No PyAV to shim with; torchcodec fails loudly, which is fine.
        os.environ[_RELAUNCH_ENV_VAR] = "1"
        return
    av_libs = _find_av_libs_dir()
    prepend = f"{shim}{os.pathsep}{av_libs}"
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    new = f"{prepend}{os.pathsep}{cur}" if cur else prepend
    os.environ["LD_LIBRARY_PATH"] = new
    os.environ[_RELAUNCH_ENV_VAR] = "1"
    # sys.argv drops the `python -m pkg` prefix, so use sys.orig_argv to keep
    # the -m form; running the file directly loses the package's implicit
    # sys.path entry. orig_argv[0] is the interpreter basename, not a
    # resolvable path, hence sys.executable for execv.
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    os.execv(sys.executable, argv)


class PyannoteDiarizer(BaseDiarizer):
    name = "pyannote"
    default_checkpoint = "pyannote/speaker-diarization-community-1"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint = checkpoint or self.default_checkpoint
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.hf_token:
            raise RuntimeError(
                "pyannote community-1 requires HF_TOKEN in the environment.\n"
                "Get one at https://hf.co/settings/tokens and accept the model gate at "
                "https://hf.co/pyannote/speaker-diarization-community-1"
            )
        ensure_ffmpeg_libs_on_ld_path_or_reexec()
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise RuntimeError(
                "pyannote.audio not installed. `pip install pyannote.audio>=3.4`"
            ) from e

        # pyannote.audio 4.x renamed `use_auth_token` to `token`; keep both.
        try:
            pipeline = Pipeline.from_pretrained(self.checkpoint, token=self.hf_token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(self.checkpoint, use_auth_token=self.hf_token)
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained returned None for {self.checkpoint}. "
                "Most common cause: the model gate is not accepted for this HF token."
            )

        try:
            import torch
            if self.device.startswith("cuda") and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            elif self.device == "mps":
                pipeline.to(torch.device("mps"))
        except Exception:
            pass

        self._pipeline = pipeline
        self._loaded = True

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        self._ensure_loaded()
        audio_path = Path(audio_path)

        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        # pyannote 4.x's torchcodec reader emits segments with start > end on
        # long clips (~30+ min). Passing an in-memory waveform bypasses that
        # path entirely, so load once with soundfile.
        import numpy as np
        import soundfile as sf
        import torch
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)
        else:
            waveform = torch.from_numpy(data.T)             # (C, T)

        result = self._pipeline(
            {"waveform": waveform, "sample_rate": sr, "uri": audio_path.stem},
            **kwargs,
        )

        # 4.x wraps the Annotation in a DiarizeOutput; 3.x returned it directly.
        annotation = getattr(result, "speaker_diarization", result)

        turns = [
            Turn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))
        turns = canonicalize_speaker_labels(turns)

        return DiarizationResult(
            turns=turns,
            audio_path=audio_path,
            model_name=self.checkpoint,
            extra={"backend": "pyannote.audio"},
        )
