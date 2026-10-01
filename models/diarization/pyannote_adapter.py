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
    """Materialize unhashed-soname symlinks (`libavutil.so.59` etc.) pointing
    at the auditwheel-hashed files PyAV ships under `<site-packages>/av.libs/`.
    Returns the shim directory as a string, or None if PyAV isn't installed.

    We need this because pyannote.audio 4.x pulls torchcodec, whose native
    `libtorchcodec_core*.so` DT_NEEDEDs unhashed sonames — the auditwheel
    hashes on PyAV's bundled libs break that lookup. Symlinks fix it.
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
    """Ensure LD_LIBRARY_PATH points at our ffmpeg soname shim + PyAV's libs.

    glibc caches LD_LIBRARY_PATH at process startup — mutating `os.environ`
    at runtime does NOT affect subsequent dlopen() calls. So if we detect a
    box without system ffmpeg AND we have PyAV to fall back on, we set the
    env vars and re-exec Python with the same argv. The sentinel env var
    stops the re-exec loop after the first hop.
    """
    if os.environ.get(_RELAUNCH_ENV_VAR):
        return
    # No re-exec needed if system ffmpeg libs are present.
    import ctypes.util
    if ctypes.util.find_library("avutil"):
        os.environ[_RELAUNCH_ENV_VAR] = "1"
        return
    shim = _build_ffmpeg_soname_shim()
    if shim is None:
        # No PyAV to shim with; torchcodec will fail loudly, and that's fine.
        os.environ[_RELAUNCH_ENV_VAR] = "1"
        return
    av_libs = _find_av_libs_dir()
    prepend = f"{shim}{os.pathsep}{av_libs}"
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    new = f"{prepend}{os.pathsep}{cur}" if cur else prepend
    os.environ["LD_LIBRARY_PATH"] = new
    os.environ[_RELAUNCH_ENV_VAR] = "1"
    # sys.argv drops the `python -m pkg` prefix. Use sys.orig_argv (3.10+) so
    # a re-exec of `python -m backend.cli ...` preserves the -m form; without
    # it, `python backend/cli.py` runs the file directly and loses the
    # package's implicit sys.path entry. orig_argv[0] is the interpreter
    # basename as invoked (e.g. "python"), not a resolvable path — pass
    # sys.executable to execv while keeping orig_argv as the argv list.
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

        # pyannote.audio 3.x → 4.x renamed `use_auth_token` to `token`; keep a
        # fallback so we work against either release.
        try:
            pipeline = Pipeline.from_pretrained(self.checkpoint, token=self.hf_token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(self.checkpoint, use_auth_token=self.hf_token)
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained returned None for {self.checkpoint}. "
                "Most common cause: the model gate is not accepted for this HF token."
            )

        # Move to GPU if we can.
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

        # pyannote.audio 4.x's internal torchcodec file reader occasionally
        # emits segments with start > end on longer clips (observed on files
        # ~30+ min). The pipeline accepts `{"waveform": tensor, "sample_rate": sr}`
        # as a drop-in alternative that uses an in-memory tensor and bypasses
        # the torchcodec path entirely — load once with soundfile instead.
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

        # pyannote 4.x returns a DiarizeOutput dataclass wrapping an Annotation
        # on `.speaker_diarization`; 3.x returned the Annotation directly.
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
