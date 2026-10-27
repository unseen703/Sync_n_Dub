"""Stage 9i from the UI — "Enhance resolution" on a finished render.

The same shape as `POST .../lipsync` on a draft job: a thing the pipeline
genuinely has a step for, which is not part of a submission because it
cannot run until a render exists. The progress ribbon draws it as a
`deferred` pill so a job that has not been enhanced reads as "has not had
its turn" rather than "will never have it".

Three things here are deliberate:

1. **A background THREAD plus a poll endpoint**, matching
   `ui/talking_head.py` and `ui/bbox_video.py`. Measured 2026-10-22: the
   ungated pass was 1156 s on 1500 frames. Even gated and split over two
   GPUs that is minutes, which is long enough that an inline request dies to
   a browser timeout while the GPU work carries on.

2. **`readiness()` returns the blockers, not a boolean.** The button's
   disabled state and its tooltip both come from the server, so they cannot
   drift from what the POST actually enforces -- the lesson
   `lipsync_readiness` already encodes.

3. **The FILE is the arbiter in `status()`**, not the stored state. A killed
   worker leaves a stale `running` that would otherwise be polled forever,
   and `done` with no mp4 is the failure CLAUDE.md S4 records twice over.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

STATUS_NAME = "STATUS.json"

#: Measured on this host 2026-10-22, 1080p, GFPGAN face-enhance only:
#: 1156 s / 1500 frames ungated. A gated pass restores roughly 15-30% of
#: frames on dialogue footage and passes the rest through at decode cost.
SEC_PER_RESTORED_FRAME = 0.77
SEC_PER_PASSED_FRAME = 0.02
#: Three networks per worker (RetinaFace resnet50, GFPGAN v1.4, ParseNet).
STARTUP_S = 40.0

_LOCKS: Dict[str, threading.Lock] = {}


def enhance_dir(job_root: Path) -> Path:
    return Path(job_root) / "enhance"


def _status_path(job_root: Path) -> Path:
    return enhance_dir(job_root) / STATUS_NAME


def _write_status(job_root: Path, **kw) -> Dict[str, Any]:
    d = enhance_dir(job_root)
    d.mkdir(parents=True, exist_ok=True)
    doc = {"updated_at": time.time(), **kw}
    tmp = _status_path(job_root).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, _status_path(job_root))
    return doc


def status(job_root: Path) -> Dict[str, Any]:
    """State of this job's enhancement, with the file as the arbiter."""
    job_root = Path(job_root)
    p = _status_path(job_root)
    if not p.is_file():
        return {"state": "idle"}
    try:
        doc = json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return {"state": "idle"}

    vid = doc.get("video")
    has_file = bool(vid and Path(vid).is_file() and Path(vid).stat().st_size)
    if doc.get("state") == "done" and not has_file:
        doc["state"] = "failed"
        doc.setdefault("error", "reported done but no video on disk")
    if doc.get("state") == "running":
        # A worker killed mid-run never writes a terminal state. Nothing here
        # can distinguish that from a slow render except time, so this is a
        # ceiling rather than a liveness check.
        stale_after = float(doc.get("expected_s") or 1800) * 4 + 600
        if time.time() - float(doc.get("started_at") or 0) > stale_after:
            doc["state"] = "failed"
            doc.setdefault("error", "worker stopped reporting")
    return doc


def resolve_render(job_root: Path, render: Optional[str]) -> Optional[Path]:
    """The render to enhance: the one asked for, else the newest.

    An id is matched against the job's own list rather than joined onto a
    path -- `Path / ".."` is not a string operation, and a request naming a
    file outside the job must not read it.

    Ids are DIRECTORY-QUALIFIED (`lipsync_asd/x.mp4`), not bare names. A
    job that has been A/B'd holds the same stem in both `lipsync/` and
    `lipsync_asd/` -- job 20261022-070010 has exactly that -- so a bare
    name picks one of the two arbitrarily, and the arm you get back is the
    one thing an A/B cannot be vague about.
    """
    from backend.restore import find_render, list_renders

    if not render:
        return find_render(Path(job_root))
    want = str(render).strip().lstrip("/")
    for p in list_renders(Path(job_root)):
        if render_id(Path(job_root), p) == want or str(p) == str(render):
            return p
    return None


def render_id(job_root: Path, p: Path) -> str:
    """`lipsync_asd/x.mp4` — stable, job-relative, unambiguous."""
    try:
        return str(Path(p).relative_to(Path(job_root)))
    except ValueError:
        return Path(p).name


def _render_facts(job_root: Path, render: Optional[str] = None) -> Dict[str, Any]:
    from backend.restore import (_decoded_frames, _dims, find_source,
                                 list_renders)

    render_p = resolve_render(Path(job_root), render)
    if not render_p:
        return {"renders": [render_id(Path(job_root), p)
                            for p in list_renders(Path(job_root))]}
    dims = _dims(render_p)
    render = render_p
    return {"renders": [render_id(Path(job_root), p)
                        for p in list_renders(Path(job_root))],
            "render": str(render),
            "render_id": render_id(Path(job_root), render),
            "render_name": render.name,
            "source": str(find_source(Path(job_root)) or "") or None,
            "width": dims[0] if dims else None,
            "height": dims[1] if dims else None,
            "frames": _decoded_frames(render)}


def estimate_seconds(frames: int, *, gated: bool, workers: int = 2,
                     scale: str = "same") -> Optional[list]:
    """`[low, high]` seconds, or None when the frame count is unknown.

    A band, not a number: the gated share depends on how much of the clip
    has a confirmed speaking face, which is 15.7% on one measured job and
    29% on another window of the same job.
    """
    if not frames or frames <= 0:
        return None
    w = max(1, workers)
    if gated:
        shares = (0.10, 0.35)
    else:
        shares = (1.0, 1.0)
    out = []
    for s in shares:
        restored = frames * s
        passed = frames - restored
        secs = (restored * SEC_PER_RESTORED_FRAME
                + passed * SEC_PER_PASSED_FRAME) / w + STARTUP_S
        if scale not in ("same", "1x"):
            # bg_upsample turns on, and Real-ESRGAN over a whole frame
            # measured 10.9 s/frame at 1080p. Order of magnitude, not a
            # promise.
            secs *= 12
        out.append(int(round(secs)))
    return out


def readiness(job_root: Path, render: Optional[str] = None) -> Dict[str, Any]:
    """Everything the Enhance resolution control needs, in one read.

    `renders` lists every lip-sync output in the job so the operator can
    choose. Restoration is model-agnostic -- it never reads which adapter
    drew the mouths -- so which render to clean is a choice, not a lookup.
    """
    job_root = Path(job_root)
    facts = _render_facts(job_root, render)
    blockers: list[str] = []

    if not facts.get("render"):
        blockers.append("Lip-sync has not produced a render yet.")
    elif not facts.get("frames") or facts["frames"] <= 0:
        blockers.append("The lip-sync render does not decode.")

    # Staged edits would mean enhancing audio-and-video from before the
    # review. `routes_edits.run_lipsync_now` refuses for the same reason.
    pending = job_root / "edits" / "pending.json"
    if pending.is_file():
        try:
            doc = json.loads(pending.read_text())
            if doc.get("ops"):
                blockers.append(
                    f"{len(doc['ops'])} edit(s) are staged. Apply or discard "
                    f"them first.")
        except Exception:                                 # noqa: BLE001
            pass

    try:
        from models.restore import get_restore
        blockers.extend(get_restore("gfpgan").blockers())
    except Exception as e:                                # noqa: BLE001
        blockers.append(f"restore model unavailable: {e}")

    st = status(job_root)
    if st.get("state") == "running":
        blockers.append("An enhancement is already running for this job.")

    return {
        "can_start": not blockers,
        "blockers": blockers,
        "already": st.get("state") == "done",
        "status": st,
        "estimate_s": estimate_seconds(facts.get("frames") or 0, gated=True),
        "estimate_s_ungated": estimate_seconds(facts.get("frames") or 0,
                                               gated=False),
        **facts,
    }


def start(job_root: Path, *, strength: float = 1.0, gated: bool = True,
          scale: str = "same", only_center_face: bool = False,
          num_gpus: Optional[int] = None,
          workers_per_gpu: Optional[int] = None,
          render: Optional[str] = None) -> Dict[str, Any]:
    """Kick the pass off in a thread. Idempotent while one is running."""
    job_root = Path(job_root)
    key = str(job_root.resolve())
    lock = _LOCKS.setdefault(key, threading.Lock())
    if not lock.acquire(blocking=False):
        return status(job_root)

    facts = _render_facts(job_root, render)
    frames = facts.get("frames") or 0
    est = estimate_seconds(frames, gated=gated, scale=scale)

    def _run():
        try:
            from backend.restore import run_restore
            res = run_restore(job_root, render=Path(facts["render"]),
                              source=Path(facts["source"]) if facts.get("source")
                              else None,
                              gated=gated, strength=strength, scale=scale,
                              only_center_face=only_center_face,
                              num_gpus=num_gpus,
                              workers_per_gpu=workers_per_gpu)
            if res.get("ok"):
                _write_status(job_root, state="done", started_at=t0,
                              expected_s=(est or [None])[-1], **res)
            else:
                _write_status(job_root, state="failed", started_at=t0,
                              error=res.get("error") or "enhancement failed")
        except Exception as e:                            # noqa: BLE001
            log.exception("enhance: failed for %s", job_root)
            _write_status(job_root, state="failed", started_at=t0,
                          error=f"{type(e).__name__}: {e}")
        finally:
            lock.release()

    t0 = time.time()
    _write_status(job_root, state="running", started_at=t0,
                  expected_s=(est or [None])[-1], frames=frames,
                  gated=gated, strength=strength, scale=scale,
                  source_render=facts.get("render"))
    threading.Thread(target=_run, daemon=True,
                     name=f"enhance-{job_root.name[:16]}").start()
    return status(job_root)
