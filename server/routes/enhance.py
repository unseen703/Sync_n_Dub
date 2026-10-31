"""Routes for stage 9i — "Enhance resolution" on a finished render.

Must be included ABOVE the StaticFiles mount in `server/server.py`: Starlette
matches in registration order, so a later router is shadowed entirely and
every route answers 404/405 from the static handler.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from ..service import enhance as _enhance
from ..utils._io import data_dir as _data_dir

log = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = _data_dir()

#: A bad id is rejected, never sanitised — a cleaned-up id would read a
#: different job's status and show it as this one's.
def _root(job_id: str) -> Path:
    jid = (job_id or "").strip("/")
    if not jid or "/" in jid or ".." in jid:
        raise HTTPException(400, f"Not a job id: {job_id!r}")
    root = DATA_DIR / "jobs" / jid
    if not root.is_dir():
        raise HTTPException(404, f"Job not found: {job_id}")
    return root


@router.get("/api/jobs/{job_id}/enhance/readiness")
def enhance_readiness(job_id: str, render: Optional[str] = None) -> Dict[str, Any]:
    """Can this job be enhanced, and if not, why not.

    `can_start` drives the button's disabled state and `blockers` its tooltip,
    both from the server, so neither can drift from what the POST enforces.
    Same contract as `GET .../lipsync/readiness`.
    """
    return _enhance.readiness(_root(job_id), render)


@router.get("/api/jobs/{job_id}/enhance")
def enhance_status(job_id: str) -> Dict[str, Any]:
    """Poll one job's enhancement. The FILE is the arbiter, not the state."""
    return _enhance.status(_root(job_id))


@router.post("/api/jobs/{job_id}/enhance")
def start_enhance(job_id: str,
                  body: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    """Run GFPGAN over the job's lip-sync render, in the background.

      strength          0..1, default 1.0. A blend between the restored and
                        original frame applied by `backend/face_enhance.py` —
                        NOT GFPGAN's own `weight`, which is inert on
                        arch='clean' and never read.
      gated             default true. Restores only the frames lip-sync
                        actually changed, found by diffing the render against
                        its input. Cuts the pass from ~20 minutes to a few on
                        a 1500-frame clip.
      scale             "same" (default) or "2x". Above 1x also turns on
                        Real-ESRGAN for the background, or the face gains real
                        detail the frame around it only interpolates to meet.
      only_center_face  default false. "Center" is a geometric guess, not the
                        speaking face.
      workers_per_gpu   1..4, default unset (one worker per card). One worker
                        leaves its card idle most of the time, so a second
                        fills the gaps; peak VRAM rises linearly and the
                        workers contend for CPU, so above 2 wants measuring.
    """
    root = _root(job_id)
    body = body or {}

    render = body.get("render")
    ready = _enhance.readiness(root, render)
    if not ready["can_start"]:
        # 409, not 400: the request is well-formed, the job is not ready.
        raise HTTPException(409, "; ".join(ready["blockers"]))

    try:
        strength = float(body.get("strength", 1.0))
    except (TypeError, ValueError):
        raise HTTPException(400, "strength must be a number between 0 and 1")
    if not 0.0 <= strength <= 1.0:
        raise HTTPException(400, "strength must be between 0 and 1")

    scale = str(body.get("scale") or "same")
    if scale not in ("same", "1x", "2x"):
        raise HTTPException(400, "scale must be 'same' or '2x'")

    # Defaulted to 2 (= 4 workers on this 2-GPU host), not left unset. A
    # worker peaked at 1691 MiB of 16311 and left its card idle 66% of the
    # time; measured 2026-10-28, two workers held both cards at ~84%. The
    # UI does not send this field at all, so a default here is the only way
    # it reaches a UI submission.
    workers_per_gpu = body.get("workers_per_gpu", 2)
    if workers_per_gpu is not None:
        try:
            workers_per_gpu = int(workers_per_gpu)
        except (TypeError, ValueError):
            raise HTTPException(400, "workers_per_gpu must be an integer")
        # Capped, not trusted: each worker loads its own RetinaFace, GFPGAN
        # and ParseNet, multiplying VRAM on a shared card.
        if not 1 <= workers_per_gpu <= 4:
            raise HTTPException(400, "workers_per_gpu must be between 1 and 4")

    # At the POST, not at readiness: this is when the VRAM is committed.
    try:
        from server.runner import _preflight_gpu
        _preflight_gpu()
    except HTTPException:
        raise
    except Exception as e:                                # noqa: BLE001
        raise HTTPException(503, f"No free GPU for enhancement: {e}") from None

    return _enhance.start(
        root,
        strength=strength,
        gated=bool(body.get("gated", True)),
        scale=scale,
        only_center_face=bool(body.get("only_center_face", False)),
        num_gpus=body.get("num_gpus"),
        workers_per_gpu=workers_per_gpu,
        # Restoration never reads the model name, so a job holding both
        # `lipsync/` and `lipsync_asd/` outputs is a choice, not a default.
        render=render,
    )
