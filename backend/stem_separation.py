"""Stem separation — dialogue (vocals) vs music-and-effects (no_vocals).

Wraps Demucs (HTDemucs by default). Uses the demucs CLI programmatically so
we do not have to babysit the model loading loop ourselves.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _ensure_ffmpeg_on_path() -> None:
    """Demucs shells out to `ffmpeg` for any non-wav input. When the system
    has no ffmpeg installed, fall back to the one bundled by imageio-ffmpeg
    by prepending its directory to PATH for the current process.

    Deliberately not `media_io._ffmpeg_bin()`: demucs calls bare `ffmpeg`
    itself, so what it needs is a PATH entry, not a path we hold. Decoding
    needs no filter the imageio build lacks, so landing on 7.0.2 rather than
    the env's 7.1 is fine here.
    """
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        ff = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return
    ff_dir = str(ff.parent)
    # imageio's binary is named e.g. `ffmpeg-linux-x86_64-v7.0.2`, not `ffmpeg`.
    # Symlink it as `ffmpeg` in a session tmp dir and put that dir on PATH so
    # `shutil.which("ffmpeg")` inside demucs succeeds.
    if ff.name != "ffmpeg":
        import tempfile
        shim_dir = Path(tempfile.gettempdir()) / "dubbingstudio-ffmpeg-shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim = shim_dir / "ffmpeg"
        if not shim.exists():
            try:
                shim.symlink_to(ff)
            except OSError:
                shutil.copy2(ff, shim)
                shim.chmod(0o755)
        ff_dir = str(shim_dir)
    os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")


@dataclass
class StemPaths:
    dialogue: Path       # a.k.a. vocals
    music_effects: Path  # a.k.a. no_vocals — everything that is not vocals
    model: str
    root: Path           # demucs output root, i.e. `root/<model>/<track>/`


def separate_stems(
    audio_or_video: Path,
    out_dir: Path,
    *,
    model: str = "htdemucs",
    device: Optional[str] = None,
    two_stems: str = "vocals",
    overwrite: bool = False,
) -> StemPaths:
    """Run Demucs on the input; return the vocals + no_vocals WAV paths.

    `audio_or_video` can be a video file (demucs will call ffmpeg internally)
    or an audio file. For best separation quality, pass a stereo 44.1 kHz
    source; the 16 kHz mono WAV extracted for diarization is NOT the right
    input here.
    """
    src = Path(audio_or_video)
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # demucs writes to <out_dir>/<model>/<track_stem>/<stem>.wav
    target_dir = out_dir / model / src.stem
    dialogue = target_dir / f"{two_stems}.wav"
    music_effects = target_dir / f"no_{two_stems}.wav"

    if not overwrite and dialogue.exists() and music_effects.exists():
        log.info("separate_stems: %s already present, skipping", target_dir)
        return StemPaths(dialogue=dialogue, music_effects=music_effects, model=model, root=out_dir)

    try:
        import demucs.separate
    except ImportError as e:
        raise RuntimeError("demucs not installed. `pip install demucs>=4.0.1`") from e

    _ensure_ffmpeg_on_path()

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    argv = [
        "-n", model,
        "--two-stems", two_stems,
        "-d", device,
        "-o", str(out_dir),
        str(src),
    ]
    log.info("demucs.separate: %s", " ".join(argv))
    # demucs.separate.main() is the console entry point — it calls sys.exit
    # on argparse errors, model-load failures, and unknown --two-stems values.
    # SystemExit inherits from BaseException, so an unwrapped exit would
    # escape past Gradio's request handler and kill the worker thread. Convert
    # it to a regular exception the pipeline can render as an error card.
    try:
        demucs.separate.main(argv)
    except SystemExit as e:
        raise RuntimeError(
            f"demucs.separate exited with code {e.code}. argv={argv}. "
            f"Common causes: model weights failed to download, invalid "
            f"--two-stems={two_stems!r}, or corrupted ~/.cache/torch/hub entry."
        ) from e

    if not dialogue.exists() or not music_effects.exists():
        raise RuntimeError(
            f"Demucs finished but expected outputs missing under {target_dir}. "
            f"Contents: {[p.name for p in target_dir.glob('*')] if target_dir.exists() else '<no dir>'}"
        )
    log.info(
        "separate_stems: wrote %s + %s",
        dialogue.name, music_effects.name,
    )
    return StemPaths(dialogue=dialogue, music_effects=music_effects, model=model, root=out_dir)
