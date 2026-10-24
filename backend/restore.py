"""Stage 9i — face restoration over a finished lip-sync render.

Runs AFTER 9h and is never part of a submission, because there is nothing to
restore until a render exists. On the CLI that is `--restore gfpgan`; in the
UI it is `POST /api/jobs/{id}/enhance`, the same shape as `POST .../lipsync`
on a draft run.

Why it is its own stage rather than a lip-sync model's second half:

  * It is orthogonal. It takes a video and returns a video and does not care
    which adapter drew the mouths, so `--lipsync X --restore gfpgan` is 5 + 2
    rather than the 5 x 2 that `wav2lip_hd` implied.
  * It is expensive in its own right. Measured 2026-10-22 on job
    20261022-070010: 1156 s against the MuseTalk render's 447 s -- 2.6x the
    cost of what it cleans. Bundling it meant every restoration tweak paid
    for a re-render it had no reason to.
  * It is iterative by nature. Strength, gating and the output size are
    things an operator looks at and adjusts; that only works against a fixed
    render.

**The audio is carried across here, not by the restorer.** The enhancer
writes video only (`-an` on its encoder), and the 2026-10-22 failure was a
remux fired against a still-being-written file: `-c:v copy` took a stream
with no finalised moov atom, returned 0, and the `&&` deleted the good
intermediate. So the remux is sequenced after a verified render, uses
`-c:a copy` (the render's audio is already AAC 48k -- re-encoding it plus
`-shortest` is what cost two frames), and the result is decode-checked
before anything is removed.
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

    Restoration is orthogonal to lip-sync -- it takes a video and returns a
    video -- so the operator picks which render to clean rather than
    inheriting a guess. A job can genuinely hold several: `lipsync/` from
    the pipeline, `lipsync_asd/` from the ASD-gated script, and in an A/B
    both at once (job 20261022-070010 has exactly that).
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

    `lipsync_asd/` is preferred over `lipsync/` when both exist: the gated
    render is the better one to restore, and `scripts/lipsync_asd_gated.py`
    writes there specifically so it cannot overwrite the ungated render it
    is being compared against.
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
    """The video the lip-sync model was GIVEN, for the frame diff.

    `mix/<stem>.dubbed.mp4` is what `run_lipsync` passes as
    `pre_muxed_video`. Without it the mask cannot be built and every frame
    is restored -- slower, still correct.
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
         # `-c:a copy`, not aac: the render's audio is already AAC 48k, and
         # re-encoding plus `-shortest` trimmed two frames on 2026-10-22.
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
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Restore a job's render. Returns a JSON-able summary, never raises.

    `scale` is "same" (default) or "2x". Anything larger than the input also
    turns on `--bg-upsample`, because at `upscale > 1` GFPGAN gives the face
    real detail while the background is only LANCZOS-interpolated to meet it
    -- invisible at a same-size target, obvious on a 512p source doubled.
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
        num_gpus=num_gpus))

    if res.error or not res.produced:
        return {"ok": False, "stage": "9i", "model": model,
                "error": res.error or "no output written",
                "wall_s": round(time.time() - t0, 1)}

    want = _decoded_frames(render)
    err = _carry_audio(silent, render, final)
    if err:
        # The silent render is real work; keep it and say so rather than
        # losing the pass over a mux.
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
            "scale_to": list(scale_to) if scale_to else None,
            "wall_s": round(dt, 1),
            "s_per_frame": round(dt / max(1, res.frames or 1), 4)}
