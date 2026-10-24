"""Routes for stage 9i — "Enhance resolution" on a finished render.

**This module's router MUST be included ABOVE `app.mount("/", StaticFiles(...))`
in `ui/server.py`.** Starlette matches in registration order, so a router
added after the static mount is shadowed entirely: every GET answers a bare
404 and every POST a 405, served by the static handler, which reads exactly
like a routing typo. Introspecting `app.routes` does not reveal it — an
included router shows up as one opaque entry whether or not it works. The
honest check is to call the routes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from .auth import basic_auth
from . import enhance as _enhance
from ._io import data_dir as _data_dir

log = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = _data_dir()

#: Mirrors `routes_edits`' refusal style: a bad id is rejected, never
#: sanitised. `Path / ".."` is not a string operation, and a cleaned-up id
#: would read a different job's status and show it as this one's.
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

    `can_start` drives the button's disabled state and `blockers` its
    tooltip, both from the server — so neither can drift from what the POST
    enforces. That is the same contract `GET .../lipsync/readiness` already
    has, and it exists because a browser re-deriving the rule is a second
    implementation nobody keeps in step.
    """
    return _enhance.readiness(_root(job_id), render)


@router.get("/api/jobs/{job_id}/enhance")
def enhance_status(job_id: str) -> Dict[str, Any]:
    """Poll one job's enhancement. The FILE is the arbiter, not the state."""
    return _enhance.status(_root(job_id))


@router.post("/api/jobs/{job_id}/enhance", dependencies=[Depends(basic_auth)])
def start_enhance(job_id: str,
                  body: Optional[Dict[str, Any]] = Body(None)) -> Dict[str, Any]:
    """Run GFPGAN over the job's lip-sync render, in the background.

    Options, all with defaults that match what was measured rather than what
    sounds generous:

      strength          0..1, default 1.0. A blend between the restored
                        frame and the original, applied by
                        `scripts/face_enhance.py`. NOT GFPGAN's own `weight`
                        argument, which is inert on arch='clean' — it is
                        forwarded into `GFPGANv1Clean.forward(**kwargs)` and
                        never read.
      gated             default true. Restore only the frames the lip-sync
                        model actually changed, found by diffing the render
                        against its input. 1156 s -> minutes on a 1500-frame
                        clip, and it leaves untouched frames alone instead of
                        moving them away from the source for no reason.
      scale             "same" (default) or "2x". Anything above 1x also
                        turns on Real-ESRGAN for the background, because
                        otherwise the face gets real detail and the frame
                        around it is only interpolated to meet it.
      only_center_face  default false. "Center" is a geometric guess, not the
                        speaking face.
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

    # At the POST, not at readiness: this is the moment the VRAM is
    # committed, same as `routes_edits.run_lipsync_now` and the talking-head
    # render.
    try:
        from ui.runner import _preflight_gpu
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
        # Which render to clean. Restoration never reads the model name, so
        # a job holding both `lipsync/` and `lipsync_asd/` outputs is a
        # choice to make rather than a preference to hard-code.
        render=render,
    )
