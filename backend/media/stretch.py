"""Time-stretch backends, chosen explicitly and measured.

How the synthesised audio is stretched decides whether the dub sounds like
the speaker or a chipmunk, so the backend is a named, inspectable variable
rather than a silent fallback. A missing pyrubberband swallowed by a bare
`except` lands every stretch on an `np.interp` resample -- a pitch shifter,
not a time stretcher -- with nothing logged.

`ffmpeg_with_filter` probes for the FILTER, not the binary: the ffmpeg on
PATH here has `atempo` but no `rubberband`, while the imageio-ffmpeg build
has both.

Backend semantics are uniform: `factor > 1` means speed up (shorter output),
matching `actual_duration / desired_duration`.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

BACKENDS = ("rubberband", "ffmpeg_atempo", "librosa", "linear")
# Ordered best-sounding first: rubberband is true phase-vocoder+transient
# handling, atempo is a cheaper WSOLA-ish filter, librosa is a plain phase
# vocoder, and linear does not preserve pitch at all.
PREFERENCE = ("rubberband", "ffmpeg_atempo", "librosa", "linear")

# atempo refuses anything outside this range in a single instance.
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0

# ffmpeg's `rubberband` filter drops its final output block when the input
# frame count is an exact multiple of 1024 -- measured ~68 ms short on a
# 24 kHz clip at factor 1.10, exact with one frame added or removed. Every
# `bodhan` segment is a whole number of 1024-frame blocks, so it hits all of
# them. Pad the input with silence so the dropped block is padding, then trim
# back to the exact target length.
_RB_TAIL_PAD_FRAMES = 8192


@dataclass
class StretchResult:
    backend: str
    requested_factor: float      # actual_dur / desired_dur; >1 = speed up
    applied_factor: float        # after the cap
    achieved_duration_s: float
    capped: bool
    pitch_preserving: bool
    error: Optional[str] = None


def _require(pkg: str, pip_name: Optional[str] = None):
    """Import `pkg` or raise an ImportError that names the pip package."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise ImportError(
            f"Time stretching needs `{pip_name or pkg}`. "
            f"Install it with `pip install {pip_name or pkg}`."
        ) from e


# ---------------------------------------------------------------------------
# backend discovery
# ---------------------------------------------------------------------------

def _candidate_ffmpeg_bins() -> List[str]:
    """ffmpeg builds worth probing, best-stocked first.

    The bundled imageio-ffmpeg static build leads -- the only one here built
    with librubberband -- and the PATH ffmpeg comes last, since preferring it
    is what hides the missing filter.
    """
    cands: List[str] = []

    def _add(p: Optional[str]) -> None:
        if p and p not in cands and Path(p).exists():
            cands.append(p)

    try:
        import imageio_ffmpeg
        _add(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:
        log.debug("imageio_ffmpeg unavailable: %s", e)
    _add(str(Path(__file__).resolve().parent.parent / ".venv-diarization" / "bin" / "ffmpeg"))
    _add(shutil.which("ffmpeg"))
    try:
        from backend.common.media_io import _ffmpeg_bin
        _add(_ffmpeg_bin())
    except Exception as e:
        log.debug("media_io._ffmpeg_bin unavailable: %s", e)
    return cands


def _binary_has_filter(binary: str, name: str) -> bool:
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        log.debug("filter probe failed for %s: %s", binary, e)
        return False
    # Rows read " ..C rubberband  A->A  Apply ...". Anchor on the name
    # column so a mention in another filter's description cannot match.
    pat = re.compile(rf"^\s*\S+\s+{re.escape(name)}\s", re.MULTILINE)
    return bool(pat.search(proc.stdout or "") or pat.search(proc.stderr or ""))


@lru_cache(maxsize=None)
def ffmpeg_with_filter(name: str) -> Optional[str]:
    """Path to an ffmpeg that actually carries filter `name`, else None."""
    for binary in _candidate_ffmpeg_bins():
        if _binary_has_filter(binary, name):
            log.debug("ffmpeg with %s filter: %s", name, binary)
            return binary
    log.debug("no ffmpeg on this host carries the %s filter", name)
    return None


def _librosa_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("librosa") is not None


def available_backends() -> List[str]:
    """Which of `BACKENDS` can run here, in `PREFERENCE` order."""
    out: List[str] = []
    for b in PREFERENCE:
        if b == "rubberband" and ffmpeg_with_filter("rubberband"):
            out.append(b)
        elif b == "ffmpeg_atempo" and ffmpeg_with_filter("atempo"):
            out.append(b)
        elif b == "librosa" and _librosa_available():
            out.append(b)
        elif b == "linear":
            out.append(b)          # numpy+soundfile are hard deps already
    return out


def resolve_backend(name: str = "auto") -> str:
    """Resolve `name` to a usable backend, or raise.

    A named backend never downgrades silently; only "auto" may choose.
    """
    avail = available_backends()
    if name == "auto":
        if not avail:
            raise RuntimeError("no time-stretch backend available")
        chosen = avail[0]
        log.info("stretch backend: %s (available: %s)", chosen, ", ".join(avail))
        return chosen
    if name not in BACKENDS:
        raise ValueError(f"unknown stretch backend {name!r}; known: {', '.join(BACKENDS)}")
    if name not in avail:
        raise RuntimeError(
            f"stretch backend {name!r} is not available on this host "
            f"(available: {', '.join(avail) or 'none'}). "
            f"Install it or pass backend='auto'."
        )
    log.info("stretch backend: %s (explicit)", name)
    return name


# ---------------------------------------------------------------------------
# ffmpeg filter construction
# ---------------------------------------------------------------------------

def _atempo_chain(factor: float) -> List[float]:
    """Split `factor` into atempo-legal steps whose product is `factor`."""
    steps: List[float] = []
    f = float(factor)
    while f > _ATEMPO_MAX:
        steps.append(_ATEMPO_MAX)
        f /= _ATEMPO_MAX
    while f < _ATEMPO_MIN:
        steps.append(_ATEMPO_MIN)
        f /= _ATEMPO_MIN
    steps.append(f)
    return steps


def ffmpeg_stretch_filter(factor: float, *, backend: str) -> str:
    """The `-filter:a` string that applies `factor` under `backend`."""
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if backend == "rubberband":
        # pitch=1 keeps the speaker's voice; crisp transients stop
        # consonants smearing at the small factors used here.
        return f"rubberband=tempo={factor:.6f}:pitch=1:transients=crisp"
    if backend == "ffmpeg_atempo":
        return ",".join(f"atempo={s:.6f}" for s in _atempo_chain(factor))
    raise ValueError(f"{backend!r} is not an ffmpeg backend")


# ---------------------------------------------------------------------------
# stretching
# ---------------------------------------------------------------------------

def _run_ffmpeg(binary: str, wav_in: Path, wav_out: Path, filt: str,
                sample_rate: Optional[int]) -> None:
    cmd = [binary, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(wav_in), "-filter:a", filt]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-c:a", "pcm_s16le", str(wav_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg stretch failed (rc={proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\nSTDERR: {proc.stderr.strip()}"
        )


def _resample_to(audio, sr_in: int, sr_out: Optional[int]):
    """Actually resample `audio` from `sr_in` to `sr_out`.

    The ffmpeg backends get this from `-ar`; the in-process ones do not.
    `sf.write(..., sample_rate)` alone writes the same samples under a new
    header, which rescales both duration and pitch. `None` or an unchanged
    rate returns the array untouched.
    """
    import numpy as np
    if not sr_out or int(sr_out) == int(sr_in) or len(audio) == 0:
        return audio
    n_target = max(1, int(round(len(audio) * float(sr_out) / float(sr_in))))
    src = np.arange(len(audio))
    idx = np.linspace(0, len(audio) - 1, n_target)
    if audio.ndim == 1:
        return np.interp(idx, src, audio).astype(audio.dtype)
    return np.stack([
        np.interp(idx, src, audio[:, c]).astype(audio.dtype)
        for c in range(audio.shape[1])
    ], axis=1)


def _linear_resample(wav_in: Path, wav_out: Path, factor: float,
                     sample_rate: Optional[int]) -> None:
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(wav_in), always_2d=False)
    if len(audio) == 0:
        raise RuntimeError(f"{wav_in} holds no audio frames; nothing to stretch")
    n_target = max(1, int(round(len(audio) / factor)))
    idx = np.linspace(0, len(audio) - 1, n_target)
    src = np.arange(len(audio))
    if audio.ndim == 1:
        out = np.interp(idx, src, audio).astype(audio.dtype)
    else:
        out = np.stack([
            np.interp(idx, src, audio[:, c]).astype(audio.dtype)
            for c in range(audio.shape[1])
        ], axis=1)
    sf.write(str(wav_out), _resample_to(out, sr, sample_rate), sample_rate or sr)


def _librosa_stretch(wav_in: Path, wav_out: Path, factor: float,
                     sample_rate: Optional[int]) -> None:
    librosa = _require("librosa")
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(wav_in), always_2d=False)
    if len(audio) == 0:
        raise RuntimeError(f"{wav_in} holds no audio frames; nothing to stretch")
    if audio.ndim == 1:
        out = librosa.effects.time_stretch(audio.astype(np.float32), rate=factor)
    else:
        chans = [librosa.effects.time_stretch(audio[:, c].astype(np.float32), rate=factor)
                 for c in range(audio.shape[1])]
        n = min(len(c) for c in chans)
        out = np.stack([c[:n] for c in chans], axis=1)
    if sample_rate and int(sample_rate) != int(sr):
        out = librosa.resample(np.asarray(out).T if out.ndim > 1 else np.asarray(out),
                               orig_sr=sr, target_sr=int(sample_rate))
        out = out.T if out.ndim > 1 else out
    sf.write(str(wav_out), out, sample_rate or sr)


def time_stretch(
    wav_in: Path, wav_out: Path, *,
    factor: float,
    backend: str = "auto",
    sample_rate: Optional[int] = None,
) -> StretchResult:
    """Stretch `wav_in` by `factor` (>1 = faster/shorter) into `wav_out`.

    `sample_rate` is an optional *output resample*: `None` (the default)
    keeps the input's rate and every backend is a pure time stretch.
    """
    import soundfile as sf
    wav_in = Path(wav_in); wav_out = Path(wav_out)
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    chosen = resolve_backend(backend)
    in_info = sf.info(str(wav_in))
    if in_info.frames <= 0:
        raise RuntimeError(f"{wav_in} holds no audio frames; nothing to stretch")
    wav_out.parent.mkdir(parents=True, exist_ok=True)

    if chosen in ("rubberband", "ffmpeg_atempo"):
        filt_name = "rubberband" if chosen == "rubberband" else "atempo"
        binary = ffmpeg_with_filter(filt_name)
        if binary is None:                    # resolve_backend already checked
            raise RuntimeError(f"ffmpeg with {filt_name} vanished between probe and use")
        filt = ffmpeg_stretch_filter(factor, backend=chosen)
        if chosen == "rubberband":
            # See _RB_TAIL_PAD_FRAMES. atrim counts samples at the filter
            # graph's rate, which is the input's; any `-ar` resample happens
            # afterwards at the encoder, so the trimmed duration is right
            # either way.
            n_target = max(1, int(round(in_info.frames / factor)))
            filt = (f"apad=pad_len={_RB_TAIL_PAD_FRAMES},{filt},"
                    f"atrim=end_sample={n_target}")
        _run_ffmpeg(binary, wav_in, wav_out, filt, sample_rate)
    elif chosen == "librosa":
        _librosa_stretch(wav_in, wav_out, factor, sample_rate)
    else:
        _linear_resample(wav_in, wav_out, factor, sample_rate)

    # ffmpeg's filters drop inputs shorter than one analysis window on the
    # floor: rc=0, zero packets written, no complaint. Returning that as a
    # successful 0.000 s stretch is how a dropped segment reaches the mix as
    # silence, so refuse it here rather than three stages downstream.
    out_info = sf.info(str(wav_out))
    if out_info.frames <= 0:
        raise RuntimeError(
            f"{chosen} produced an empty file from {wav_in.name} "
            f"({in_info.duration:.3f}s, factor {factor:.4f}); the input is "
            f"likely shorter than the backend's analysis window"
        )

    return StretchResult(
        backend=chosen,
        requested_factor=float(factor),
        applied_factor=float(factor),
        achieved_duration_s=float(out_info.duration),
        capped=False,
        pitch_preserving=(chosen != "linear"),
    )


def stretch_to_duration(
    wav_in: Path, wav_out: Path, *,
    target_duration_s: float,
    max_stretch: float = 0.10,
    backend: str = "auto",
) -> StretchResult:
    """Fit `wav_in` to `target_duration_s`, clamped to ±`max_stretch`.

    Clamping is reported, never hidden: a capped segment still overruns its
    slot and the mix stage needs to know.
    """
    import soundfile as sf
    wav_in = Path(wav_in); wav_out = Path(wav_out)
    actual = float(sf.info(str(wav_in)).duration)
    desired = max(0.01, float(target_duration_s))
    ratio = actual / desired
    applied = max(1.0 - max_stretch, min(1.0 + max_stretch, ratio))
    capped = applied != ratio

    res = time_stretch(wav_in, wav_out, factor=applied, backend=backend)
    res.requested_factor = ratio
    res.capped = capped
    if capped:
        log.info("stretch capped: wanted %.3fx, applied %.3fx (%s) on %s",
                 ratio, applied, res.backend, wav_in.name)
    return res


def wsola_stretch_cap(
    wav_in: Path, wav_out: Path, *,
    target_duration_s: float,
    max_stretch: float = 0.10,
    backend: str = "auto",
) -> "tuple[float, bool]":
    """Stretch `wav_in` to `target_duration_s`, capped at +-`max_stretch`.

    Returns (applied_factor, budget_violation). A thin shim around
    `stretch_to_duration` so `backend.tts` can unpack the two values it needs
    without carrying the full `StretchResult`.
    """
    res = stretch_to_duration(
        wav_in, wav_out,
        target_duration_s=target_duration_s,
        max_stretch=max_stretch,
        backend=backend,
    )
    return res.applied_factor, res.capped
