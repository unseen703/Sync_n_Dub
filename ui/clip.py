"""Cut the chosen part of a video out, before the pipeline sees it.

There is no clip-window flag anywhere in `backend/` (CLAUDE.md section 2), and
adding one would mean editing the file the CLI, the sweep, `experiment.py` and
`edit_agent.rerun` all share. So the window is a UI-side concern: the server
cuts the clip into `<job>/videos/` and hands the pipeline a local file.
`_adopt_into_job` returns a path unchanged when it is already inside
`<job>/videos/`, so this needs no backend change at all.

Two things about the ffmpeg call are load-bearing:

  * **`-ss` goes before `-i`.** After `-i` it is a decode-and-discard seek:
    correct, and minutes slower on a long source. Before `-i` it seeks the
    container first.
  * **It re-encodes.** `-c copy` would start the file at the keyframe *before*
    the requested time and leave the first frames undecodable, so the clip
    would not start where the operator asked. Modelled on
    `scripts/clip_asd_musetalk_run.py::phase_cut`, which re-encodes for the
    same reason; the audio comes along because every downstream stage wants it.

There is no system ffmpeg on this host, so the binary always comes from
`backend.media_io._ffmpeg_bin()` (CLAUDE.md section 1).
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from ._io import env_int as _env_int

log = logging.getLogger("ui.clip")

#: Long enough that a stuck ffmpeg surfaces, short enough that it is not the
#: thing holding a 20-minute render hostage.
CUT_TIMEOUT_S = _env_int("DUBBING_UI_CUT_TIMEOUT_S", 2 * 60 * 60)


def cut_window(src: Path, out: Path, start_s: float, end_s: float,
               *, crf: int = 18, timeout_s: Optional[int] = None) -> Path:
    """Re-encode `src[start_s:end_s]` to `out`. Returns `out`."""
    from backend.media_io import _ffmpeg_bin

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
    # A zero exit with no file is the repo's recurring ffmpeg failure mode
    # (CLAUDE.md section 1 — Wav2Lip's mux loses the render and exits 0), so
    # the artifact is checked rather than the return code.
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg reported success but wrote nothing to {out}. "
            f"stderr: {(proc.stderr or '').strip()[-800:]}"
        )
    return out


def adopt_whole(src: Path, out: Path) -> Path:
    """Place the whole video at `out` without a second copy on disk.

    A whole-video job has no window to cut, and re-encoding it to say so
    would cost minutes and lose quality for nothing. Hardlink when the source
    and the job folder are on the same filesystem; fall back to a copy when
    they are not.
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
