"""Stage 9i — face restoration over a finished lip-sync render.

Runs after 9h, never as part of a submission: `--restore gfpgan` on the CLI,
`POST /api/jobs/{id}/enhance` in the frontend. The restorer writes video only, so
the render's audio is remuxed here after the output is decode-checked.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

STAGE_MARKER = "=== stage 9i: face restoration (%s) ==="


def list_renders(job_root: Path) -> list:
    """Every lip-sync render in this job, newest first.

    A job can hold several — `lipsync/` from the pipeline, `lipsync_asd/` from
    the gated script, or both during an A/B — so the operator picks which one
    to clean rather than inheriting a guess.
    """
    out = []
    for sub in ("lipsync_asd", "lipsync"):
        d = Path(job_root) / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.mp4"), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if p.is_file() and p.stat().st_size:
                out.append(p)
    return out


def find_render(job_root: Path) -> Optional[Path]:
    """The lip-sync render this job produced, newest first.

    `lipsync_asd/` wins over `lipsync/` when both exist: the gated render is
    the better one to restore, and the gated script writes there so it cannot
    overwrite the ungated render it is compared against.
    """
    for sub in ("lipsync_asd", "lipsync"):
        d = job_root / sub
        if not d.is_dir():
            continue
        hits = sorted((p for p in d.glob("*.mp4") if p.is_file()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None


def find_source(job_root: Path) -> Optional[Path]:
    """The video the lip-sync model was given, for the frame diff.

    This is what `run_lipsync` passes as `pre_muxed_video`. Without it no mask
    can be built and every frame is restored — slower, still correct.
    """
    d = job_root / "mix"
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.dubbed.mp4"))
    return hits[0] if hits else None


def _dims(path: Path) -> Optional[Tuple[int, int]]:
    from .media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True, timeout=60)
        w, h = (int(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2])
        return (w, h)
    except Exception:                                     # noqa: BLE001
        return None


def _decoded_frames(path: Path) -> int:
    from .media_io import _ffprobe_bin
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return -1
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
             str(path)], capture_output=True, text=True)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:                                     # noqa: BLE001
        return -1


def _carry_audio(silent: Path, from_video: Path, out: Path) -> Optional[str]:
    """Mux the render's audio onto the restored video. Error string or None."""
    from .media_io import _ffmpeg_bin
    ffmpeg = _ffmpeg_bin() or "ffmpeg"
    p = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(silent), "-i", str(from_video),
         "-map", "0:v:0", "-map", "1:a:0?",
         # `-c:a copy`, not aac: the audio is already AAC 48k, and re-encoding
         # it plus `-shortest` trims frames.
         "-c:v", "copy", "-c:a", "copy", str(out)],
        capture_output=True, text=True)
    if p.returncode != 0 or not out.is_file():
        return f"remux rc={p.returncode}: {(p.stderr or '').strip()[-500:]}"
    return None


def run_restore(
    job_root: Path,
    *,
    model: str = "gfpgan",
    render: Optional[Path] = None,
    source: Optional[Path] = None,
    gated: bool = True,
    strength: float = 1.0,
    scale: Optional[str] = None,
    only_center_face: bool = False,
    num_gpus: Optional[int] = None,
    workers_per_gpu: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Restore a job's render. Returns a JSON-able summary, never raises.

    `scale` is "same" (default) or "2x"; anything above 1x also turns on
    `--bg-upsample`, since GFPGAN gives the face real detail while the
    background is only LANCZOS-interpolated to meet it.

    `workers_per_gpu` splits each card further; None means one worker per card.
    A single worker leaves the GPU largely idle, and CPU contention is what
    caps how far this can be raised.
    """
    from models.restore import get_restore
    from models.restore.base import RestoreRequest

    t0 = time.time()
    job_root = Path(job_root)
    log.info(STAGE_MARKER, model)

    render = Path(render) if render else find_render(job_root)
    if not render or not render.is_file():
        return {"ok": False, "error": "no lip-sync render to restore",
                "stage": "9i"}
    source = Path(source) if source else find_source(job_root)

    out_dir = Path(out_dir) if out_dir else (job_root / "enhance")
    out_dir.mkdir(parents=True, exist_ok=True)
    silent = out_dir / f"{render.stem}.enhanced.silent.mp4"
    final = out_dir / f"{render.stem}.enhanced.mp4"

    scale_to = None
    bg_upsample = False
    if scale and scale not in ("same", "1x"):
        dims = _dims(render)
        if dims:
            mult = 2 if scale in ("2x", "2") else 1
            scale_to = (dims[0] * mult, dims[1] * mult)
            bg_upsample = mult > 1

    rs = get_restore(model)
    res = rs.restore(RestoreRequest(
        video=render, out_path=silent, source=source, gated=gated,
        strength=strength, scale_to=scale_to,
        only_center_face=only_center_face, bg_upsample=bg_upsample,
        num_gpus=num_gpus, workers_per_gpu=workers_per_gpu))

    if res.error or not res.produced:
        return {"ok": False, "stage": "9i", "model": model,
                "error": res.error or "no output written",
                "wall_s": round(time.time() - t0, 1)}

    want = _decoded_frames(render)
    err = _carry_audio(silent, render, final)
    if err:
        # The silent render is real work; keep it rather than lose the pass
        # over a mux.
        log.warning("restore: %s; keeping the silent render at %s", err, silent)
        final = silent
    else:
        got = _decoded_frames(final)
        if want > 0 and got != want:
            return {"ok": False, "stage": "9i", "model": model,
                    "error": (f"muxed output decodes {got} frame(s), source "
                              f"render has {want}; keeping {silent}"),
                    "wall_s": round(time.time() - t0, 1)}
        silent.unlink(missing_ok=True)

    dt = time.time() - t0
    log.info("restore: stage 9i done in %.0fs → %s", dt, final)
    return {"ok": True, "stage": "9i", "model": model,
            "video": str(final), "source_render": str(render),
            "frames": res.frames, "restored": res.restored,
            "passed_through": res.passed_through,
            "changed_frames": res.extra.get("changed_frames"),
            "gated": bool(res.extra.get("mask")),
            "strength": strength,
            # What actually ran: the script collapses to fewer workers when a
            # shard would fall under MIN_FRAMES_PER_SHARD, and a wall time only
            # compares against the split that produced it.
            "workers": res.extra.get("workers"),
            "workers_per_gpu": workers_per_gpu,
            "scale_to": list(scale_to) if scale_to else None,
            "wall_s": round(dt, 1),
            "s_per_frame": round(dt / max(1, res.frames or 1), 4)}
