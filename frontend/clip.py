"""Cut the chosen part of a video out, before the pipeline sees it.

There is no clip-window flag in `backend/`, so the window is a UI concern: the
server cuts into `<job>/videos/` and hands the pipeline a local file.

Two things about the ffmpeg call are load-bearing. `-ss` goes BEFORE `-i`, or
it becomes a decode-and-discard seek that costs minutes on a long source. And
it re-encodes: `-c copy` starts the file at the preceding keyframe and leaves
the first frames undecodable, so the clip would not begin where asked.

There is no system ffmpeg on this host, so the binary comes from
`backend.media_io._ffmpeg_bin()`.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from ._io import env_int as _env_int

log = logging.getLogger("frontend.clip")

#: Long enough that a stuck ffmpeg surfaces, short enough that it never holds
#: a long render hostage.
CUT_TIMEOUT_S = _env_int("DUBBING_UI_CUT_TIMEOUT_S", 2 * 60 * 60)


def cut_window(src: Path, out: Path, start_s: float, end_s: float,
               *, crf: int = 18, timeout_s: Optional[int] = None) -> Path:
    """Re-encode `src[start_s:end_s]` to `out`. Returns `out`."""
    from backend.common.media_io import _ffmpeg_bin

    src, out = Path(src), Path(out)
    if not src.is_file():
        raise FileNotFoundError(src)
    dur = float(end_s) - float(start_s)
    if dur <= 0:
        raise ValueError(f"empty window: {start_s} -> {end_s}")
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{float(start_s):.3f}",          # BEFORE -i: container seek
        "-i", str(src),
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(out),
    ]
    log.info("cut %s [%.2f, %.2f] -> %s", src.name, start_s, end_s, out)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout_s or CUT_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg cut failed (rc={proc.returncode}): "
            f"{(proc.stderr or '').strip()[-800:]}"
        )
    # A zero exit with no file is a recurring ffmpeg failure mode here, so the
    # artifact is the arbiter, not the return code.
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg reported success but wrote nothing to {out}. "
            f"stderr: {(proc.stderr or '').strip()[-800:]}"
        )
    return out


def adopt_whole(src: Path, out: Path) -> Path:
    """Place the whole video at `out` without a second copy on disk.

    A whole-video job has nothing to cut, and re-encoding would cost minutes
    and quality for nothing. Hardlinks when source and job folder share a
    filesystem, copies when they do not.
    """
    src, out = Path(src), Path(out)
    if not src.is_file():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.samefile(src):
            return out
        out.unlink()
    try:
        os.link(src, out)
        log.info("adopted %s -> %s (hardlink)", src.name, out)
    except OSError as e:
        log.info("hardlink refused (%s); copying instead", e)
        import shutil
        shutil.copy2(src, out)
    return out
