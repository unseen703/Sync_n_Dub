"""Media I/O: demux + audio extraction, thin wrapper around ffmpeg.

Diarization and Bodhan ASR both want mono 16 kHz WAV, so that is the default.
Change `sample_rate` / `mono` to keep the source characteristics for stem
separation, which prefers 44.1 kHz stereo.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _ffmpeg_bin() -> str:
    """Prefer system ffmpeg; fall back to the imageio-ffmpeg bundled build."""
    for name in ("ffmpeg",):
        found = shutil.which(name)
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


def extract_audio(
    video: Path,
    out_wav: Optional[Path] = None,
    *,
    sample_rate: int = 16_000,
    mono: bool = True,
    overwrite: bool = True,
) -> Path:
    """Demux + resample the audio track of `video` to a WAV file.

    Returns the output WAV path. If `out_wav` is None, writes next to the
    video with `.wav` suffix.
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


def probe_duration(media: Path) -> float:
    """Return duration in seconds via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        # imageio-ffmpeg does not ship ffprobe.
        raise RuntimeError("ffprobe not on PATH. Install with `sudo apt install ffmpeg`.")
    proc = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())
