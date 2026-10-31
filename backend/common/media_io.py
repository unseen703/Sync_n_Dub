"""Media I/O: demux, audio extraction and probes, wrapping ffmpeg.

Defaults to mono 16 kHz WAV, which diarization and ASR both want; change
`sample_rate` / `mono` for stem separation, which prefers 44.1 kHz stereo.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _interpreter_bin(name: str) -> Optional[str]:
    """`name` from the directory of the running interpreter, if it is there.

    Scripts are launched by naming the conda python outright rather than by
    activating the env, so PATH often has no ffmpeg or ffprobe while
    `envs/dubbing/bin` has both.
    """
    cand = Path(sys.executable).parent / name
    return str(cand) if cand.exists() else None


def _ffmpeg_bin() -> str:
    """Prefer system ffmpeg, then the running env's, then imageio-ffmpeg.

    These builds carry different FILTERS — the conda 7.1 has `atempo` but not
    `rubberband`, the imageio static build has both — so probe per filter via
    `backend.stretch.available_backends()` rather than assuming which binary
    this returned.
    """
    for name in ("ffmpeg",):
        found = shutil.which(name)
        if found:
            return found
    found = _interpreter_bin("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio-ffmpeg not installed. "
            "`sudo apt install ffmpeg` or `pip install imageio-ffmpeg`."
        ) from e


def _ffprobe_bin() -> Optional[str]:
    """System ffprobe, else the running env's, else beside our ffmpeg.

    imageio-ffmpeg publishes no ffprobe entry point, though its static build
    sometimes has one in the same directory. Returns None when there is none
    at all, which degrades every caller silently: probes answer "duration
    unknown" and `backend.lipsync._needs_remux` answers "no remux needed" for
    everything.
    """
    found = shutil.which("ffprobe")
    if found:
        return found
    found = _interpreter_bin("ffprobe")
    if found:
        return found
    try:
        ffmpeg = Path(_ffmpeg_bin())
    except RuntimeError:
        return None
    for name in ("ffprobe", "ffprobe.exe"):
        cand = ffmpeg.parent / name
        if cand.exists():
            return str(cand)
    return None


def extract_audio(
    video: Path,
    out_wav: Optional[Path] = None,
    *,
    sample_rate: int = 16_000,
    mono: bool = True,
    overwrite: bool = True,
) -> Path:
    """Demux and resample `video`'s audio track to a WAV file.

    Returns the output path, written next to the video with a `.wav` suffix
    when `out_wav` is None.
    """
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)

    if out_wav is None:
        out_wav = video.with_suffix(".wav")
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    if out_wav.exists() and not overwrite:
        log.info("extract_audio: %s already exists, skipping", out_wav)
        return out_wav

    cmd = [
        _ffmpeg_bin(),
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video),
        "-vn",                                        # drop video
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    log.info("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR: {proc.stderr.strip()}"
        )

    if not out_wav.exists():
        raise RuntimeError(f"ffmpeg reported success but {out_wav} does not exist")
    log.info("extract_audio: wrote %s (%.1f MB)", out_wav, out_wav.stat().st_size / 1e6)
    return out_wav


_FFMPEG_DURATION = re.compile(
    r"Duration:\s*(\d+):(\d{2}):(\d{2})[.,](\d{1,3})"
)


def _duration_via_ffmpeg(media: Path) -> float:
    """Read the container `Duration:` line ffmpeg prints on stderr.

    ffmpeg exits non-zero with no output file given, so the return code is
    ignored and only the banner matters.
    """
    cmd = [_ffmpeg_bin(), "-hide_banner", "-i", str(media)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = _FFMPEG_DURATION.search(proc.stderr or "")
    if not m:
        raise RuntimeError(
            f"could not read duration of {media} via ffmpeg:\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR: {(proc.stderr or '').strip()[-2000:]}"
        )
    h, mm, s, frac = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000.0


def probe_duration(media: Path) -> float:
    """Duration in seconds via ffprobe, or ffmpeg when none is installed.

    Callers treat a failure as "unknown" and substitute a guess, which poisons
    tempo maths, so exhaust every reachable binary before giving up.
    """
    media = Path(media)
    if not media.exists():
        raise FileNotFoundError(media)
    ffprobe = _ffprobe_bin()
    if ffprobe:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(media),
            ],
            capture_output=True, text=True,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out and out != "N/A":
            try:
                return float(out)
            except ValueError:
                log.warning("probe_duration: ffprobe returned %r for %s", out, media)
        else:
            log.warning("probe_duration: ffprobe failed for %s (rc=%s), "
                        "falling back to ffmpeg", media, proc.returncode)
    return _duration_via_ffmpeg(media)


def probe_video(path: Path) -> tuple[float, float]:
    """Return `(fps, duration_s)` via ffprobe, else zeros.

    The zeros are a deliberate best-effort answer, unlike `probe_duration`
    above: callers are adapters reporting on a render they just finished, and
    a failed probe should not take down a render that worked. `_needs_remux`
    is the one reader that acts on the number and treats 0.0 as "cannot tell".
    """
    ffprobe = _ffprobe_bin()
    if not ffprobe or not Path(path).exists():
        return (0.0, 0.0)
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        fps = 0.0
        dur = 0.0
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
