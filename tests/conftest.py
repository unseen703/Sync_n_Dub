"""Shared fixtures. No models, no network, no real clip -- only tiny generated WAVs.

`backend.timing.raw_clip_duration` reads the raw TTS clip's length off the
file with `soundfile.info`, so "testable with no audio" still needs a file of
the right length to exist. `mini_merged` writes one per segment, exactly as
long as the fixture's `tts_duration_s` says, and rewrites the clip paths onto
them -- in a tmp directory, never next to the committed fixture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def tone(duration_s: float, *, sample_rate: int = 24000, freq: float = 440.0,
         amplitude: float = 0.5) -> np.ndarray:
    """A pure sine, which is the only signal whose pitch a test can assert on."""
    n = max(1, int(round(duration_s * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_tone(path: Path, duration_s: float, *, sample_rate: int = 24000,
               freq: float = 440.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone(duration_s, sample_rate=sample_rate, freq=freq),
             sample_rate)
    return path


_TINY_MP4: Path | None = None


def write_tiny_mp4(path: Path, *, duration_s: float = 1.0) -> Path:
    """A REAL mp4 of `duration_s`, silent, 64x64, black.

    The lip-sync tests used to stand a source video up as `b"\\0" * 1024`,
    which was fine until segmenting added an unconditional
    `probe_duration(face_video)` at `backend/lipsync.py:218`. Ten tests then
    failed with `moov atom not found` — not because the behaviour under test
    broke, but because the stand-in was never a container.

    Stubbing the probe instead would have been shorter and would have
    removed the one call these tests most want left alone: they exist to
    catch lip-sync reporting success over a file that is not there, and
    every probe they skip is a check they stop making.

    Encoded ONCE per session and copied, so the ffmpeg cost is paid once.
    """
    global _TINY_MP4
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _TINY_MP4 is None or not _TINY_MP4.is_file():
        import subprocess
        import tempfile
        from backend.media_io import _ffmpeg_bin

        cache = Path(tempfile.gettempdir()) / "dubbingstudio-tiny.mp4"
        if not cache.is_file():
            subprocess.run(
                [_ffmpeg_bin(), "-y", "-v", "error",
                 "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration_s}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                 str(cache)],
                check=True, capture_output=True)
        _TINY_MP4 = cache

    path.write_bytes(_TINY_MP4.read_bytes())
    return path


def materialise(dest_dir: Path, *, sample_rate: int = 24000) -> Path:
    """Copy `mini.mt.json` into `dest_dir` with real clips behind every segment.

    Returns the path of the copy. The committed fixture is never touched.
    """
    dest_dir = Path(dest_dir)
    raw_dir = dest_dir / "tts" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((FIXTURES / "mini.mt.json").read_text())
    for seg in data["segments"]:
        dur = float(seg["tts_duration_s"])
        wav = write_tone(raw_dir / f"{seg['id']}.wav", dur, sample_rate=sample_rate)
        seg["tts_raw_audio"] = str(wav)
        seg["tts_audio"] = str(wav)
    dest = dest_dir / "mini.mt.json"
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return dest


@pytest.fixture
def mini_merged(tmp_path: Path) -> Path:
    """The three-segment fixture, materialised under `tmp_path`."""
    return materialise(tmp_path / "mini")


@pytest.fixture
def mini_data(mini_merged: Path) -> dict:
    return json.loads(mini_merged.read_text())
