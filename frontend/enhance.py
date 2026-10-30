"""Stage 9i from the UI — "Enhance resolution" on a finished render.

Runs the restoration pass in a background thread with a poll endpoint, since
it takes minutes. `readiness()` returns the blockers the POST enforces, and
`status()` treats the output file as the arbiter rather than the stored state.
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

#: Per-WORKER cost of one restored frame at 1080p, GFPGAN face-enhance only.
#:
#: Measured 2026-10-28 on job 20261026-120224, two shards of the same video:
#: 0.356 s/frame and ~0.43. The SLOWER one is the number to use, because a
#: run ends when its last shard does — a mean under-estimates every run, and
#: the gap here is content rather than noise (the fast shard had 669 frames
#: with no face detected, and those skip the expensive half).
#:
#: The old 0.77 was a whole-run wall divided by frames, which folded the
#: serial work below into a term that then got divided by the worker count.
SEC_PER_RESTORED_FRAME = 0.43
SEC_PER_PASSED_FRAME = 0.02

#: Per-frame cost of the work that does NOT parallelise: the diff that
#: builds the mask (gated runs only), then the concat, the audio remux and
#: the decode-check. Same job: 2093 s wall against ~1620 s of worker time,
#: so roughly 470 s of serial work over 7497 frames.
SEC_PER_SERIAL_FRAME = 0.063

#: Three networks per worker (RetinaFace resnet50, GFPGAN v1.4, ParseNet).
STARTUP_S = 40.0

#: What a SECOND worker on an already-busy card is worth. One per GPU held
#: both cards at ~84% utilisation, so the next one fills gaps rather than
#: doubling throughput. Not measured directly yet — which is part of why
#: this function returns a band and not a number.
EXTRA_WORKER_EFFICIENCY = 0.6

#: Mirrors `routes_enhance.start_enhance`'s default. Two copies because the
#: UI never sends `workers_per_gpu`, and an estimate computed for a
#: different split than the one that will run is worse than no estimate.
DEFAULT_WORKERS_PER_GPU = 2


def _n_gpus() -> int:
    """GPUs a run will spread over.

    `backend/face_enhance.py` counts them with torch inside the same
    `dubbing` env (basicsr / gfpgan / facexlib / realesrgan were folded in
    on 2026-10-30); `visible_cuda_devices` is the canonical count on this
    side and honours `CUDA_VISIBLE_DEVICES` the same way, so the two agree.
    """
    try:
        from backend.common.gpu import visible_cuda_devices
        return len(visible_cuda_devices()) or 1
    except Exception:                                     # noqa: BLE001
        return 1


def default_workers() -> int:
    """How many workers a UI submission will actually start."""
    return max(1, _n_gpus()) * DEFAULT_WORKERS_PER_GPU


def effective_workers(workers: int) -> float:
    """`workers`, discounted for the ones sharing a card.

    One per GPU scales; each one beyond that is worth
    `EXTRA_WORKER_EFFICIENCY`. A float because it is a divisor, not a count.
    """
    workers = max(1, int(workers))
    gpus = max(1, _n_gpus())
    on_own_card = min(workers, gpus)
    return on_own_card + (workers - on_own_card) * EXTRA_WORKER_EFFICIENCY


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
        # A worker killed mid-run never writes a terminal state, and nothing
        # here can tell that from a slow render except time. A ceiling, not a
        # liveness check.
        stale_after = float(doc.get("expected_s") or 1800) * 4 + 600
        if time.time() - float(doc.get("started_at") or 0) > stale_after:
            doc["state"] = "failed"
            doc.setdefault("error", "worker stopped reporting")
    return doc


def resolve_render(job_root: Path, render: Optional[str]) -> Optional[Path]:
    """The render to enhance: the one asked for, else the newest.

    The id is matched against the job's own list rather than joined onto a
    path, so a request naming a file outside the job cannot read it. Ids are
    directory-qualified (`lipsync_asd/x.mp4`): an A/B'd job holds the same stem
    in both `lipsync/` and `lipsync_asd/`, and a bare name would pick an arm
    arbitrarily.
    """
    from backend.restore.restore import find_render, list_renders

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
    from backend.restore.restore import (_decoded_frames, _dims, find_source,
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


def estimate_seconds(frames: int, *, gated: bool, workers: Optional[int] = None,
                     scale: str = "same") -> Optional[list]:
    """`[low, high]` seconds, or None when the frame count is unknown.

    A band, not a number, because the two biggest terms are both unknown
    before the run:

    * **How much the gate actually skips depends on the lip-sync model.**
      MuseTalk leaves whole frames untouched and gates down to 15-30%.
      LatentSync re-encodes every frame, so the diff finds ~99% changed and
      the gate saves nothing — measured 7422 of 7497 on job 20261026-120224.
      The low end is the MuseTalk case, the high end is LatentSync.
    * **Workers beyond one per card share it.** See `effective_workers`.
    """
    if not frames or frames <= 0:
        return None
    w = effective_workers(workers if workers else default_workers())
    if gated:
        shares = (0.15, 1.0)
    else:
        shares = (1.0, 1.0)
    out = []
    for s in shares:
        restored = frames * s
        passed = frames - restored
        # The serial term is OUTSIDE the division on purpose: the mask diff,
        # the concat and the remux run once however many workers there are,
        # and folding them into the per-frame cost is what made the old
        # estimate scale the wrong way with the worker count.
        gpu_work = (restored * SEC_PER_RESTORED_FRAME
                    + passed * SEC_PER_PASSED_FRAME)
        if scale not in ("same", "1x"):
            # bg_upsample turns on, and Real-ESRGAN over a whole frame runs
            # ~10.9 s/frame at 1080p. Order of magnitude, not a promise.
            # It multiplies the GPU work only — the mask diff and the remux
            # cost the same whatever size the frames come out.
            gpu_work *= 12
        out.append(int(round(gpu_work / w
                             + frames * SEC_PER_SERIAL_FRAME
                             + STARTUP_S)))
    return out


def readiness(job_root: Path, render: Optional[str] = None) -> Dict[str, Any]:
    """Everything the Enhance resolution control needs, in one read.

    `renders` lists every lip-sync output in the job: restoration never reads
    which adapter drew the mouths, so which render to clean is a choice rather
    than a lookup.
    """
    job_root = Path(job_root)
    facts = _render_facts(job_root, render)
    blockers: list[str] = []

    if not facts.get("render"):
        blockers.append("Lip-sync has not produced a render yet.")
    elif not facts.get("frames") or facts["frames"] <= 0:
        blockers.append("The lip-sync render does not decode.")

    # Staged edits would mean enhancing video from before the review;
    # `routes_edits.run_lipsync_now` refuses for the same reason.
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
        from backend.restore.restore import get_restore
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
    # The split this run will actually use, not the default. `expected_s`
    # feeds the stale-worker ceiling in `status()`, so an estimate for a
    # narrower split than the one running would call a healthy pass dead.
    workers = (max(1, _n_gpus()) * workers_per_gpu) if workers_per_gpu else None
    est = estimate_seconds(frames, gated=gated, scale=scale, workers=workers)

    def _run():
        try:
            from backend.restore.restore import run_restore
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
