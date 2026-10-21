"""The ASD review video: who the pipeline thinks is talking, drawn on screen.

The operator corrects the active speaker by WATCHING, so they need to see
what the pipeline saw. This draws one box per tracked face per frame, green
where TalkNet scored it above the speaking threshold and red where it did
not, labelled with the track id, the speaker that track belongs to, and the
score itself.

It is a review artifact, never an input to anything. Nothing downstream reads
it, so a wrong box here misleads a human rather than mis-rendering a dub --
which is why it can afford to show ALL faces, including the ones the lip-sync
stage would ignore. That contrast is the point: a frame with two red boxes
and no green one is exactly the reaction shot where MuseTalk would otherwise
animate an arbitrary mouth.

Rendering is a full decode + composite pass over the clip, which is minutes
at 1080p -- so it runs in a thread and reports through a status file, the
same shape `ui/progress.py` already polls. A blocking GET would die to a
browser timeout while the work continued.
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

REVIEW_SUBDIR = "review"
STATUS_NAME = "bbox.status.json"
VIDEO_NAME = "bbox.mp4"

#: Matches `asd_on_screen_threshold` and `lipsync_targets`, so "speaking"
#: means one thing across the review screen and the render it informs.
SPEAK_THRESHOLD = 0.5

_LOCKS: Dict[str, threading.Lock] = {}


def paths(job_root: Path) -> Dict[str, Path]:
    d = Path(job_root) / REVIEW_SUBDIR
    return {"dir": d, "video": d / VIDEO_NAME, "status": d / STATUS_NAME}


def _write_status(job_root: Path, **kw) -> Dict[str, Any]:
    p = paths(job_root)
    p["dir"].mkdir(parents=True, exist_ok=True)
    doc = {"updated_at": time.time(), **kw}
    tmp = p["status"].with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, p["status"])
    return doc


def status(job_root: Path) -> Dict[str, Any]:
    p = paths(job_root)
    try:
        doc = json.loads(p["status"].read_text())
    except (OSError, ValueError):
        doc = {}
    # The FILE is the arbiter, not the status doc: a killed render leaves a
    # stale "running" behind forever, and CLAUDE.md records what trusting a
    # status field over the artifact costs.
    ready = p["video"].is_file() and p["video"].stat().st_size > 0
    state = doc.get("state") or ("done" if ready else "absent")
    if state == "done" and not ready:
        state = "failed"
        doc.setdefault("error", "render reported done but wrote no video")
    return {**doc, "state": state, "ready": ready}


def render(job_root: Path, *, source: Optional[Path] = None,
           threshold: float = SPEAK_THRESHOLD,
           labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Draw the boxes. Blocking; `start()` is the background wrapper."""
    import cv2
    import numpy as np

    from models.faces import FaceTracks

    job_root = Path(job_root)
    p = paths(job_root)
    labels = labels or {}

    npz = next(iter(sorted(job_root.glob("faces/*.npz"))), None)
    if npz is None:
        return _write_status(job_root, state="failed",
                             error="no face tracks: stage 5 did not run")
    src = source or next(iter(sorted(job_root.glob("videos/*.mp4"))), None)
    if src is None or not Path(src).is_file():
        return _write_status(job_root, state="failed",
                             error="no source clip in videos/")

    tracks = FaceTracks.load_npz(npz)
    # Stage 6 persists PER-SEGMENT assignments, not a per-frame score array
    # (`<stem>.<model>.assignments.json`: seg_id, speaker, face_track,
    # asd_score, on_screen, start_s, end_s). An earlier version of this
    # module looked for a `scores.npz` that is never written, so every box
    # came out grey -- the render ran, looked plausible, and showed nothing.
    #
    # So a track is drawn as SPEAKING over the frames of a segment that was
    # assigned to it and marked on-screen. Outside any segment nothing is
    # known about who is talking, and those frames stay grey rather than
    # being coloured by assumption.
    spk_of: Dict[str, str] = {}
    spans: list = []                      # (f_start, f_end, track_id, score)
    asd_json = next(iter(sorted(job_root.glob("asd/*.assignments.json"))), None)
    if asd_json is not None:
        try:
            recs = json.loads(asd_json.read_text())
        except (OSError, ValueError):
            recs = []
        probe = cv2.VideoCapture(str(src))
        _fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
        probe.release()
        for a in recs:
            tid = a.get("face_track")
            if not tid:
                continue
            tid = str(tid)
            spk_of.setdefault(tid, str(a.get("speaker") or ""))
            spans.append((int(float(a.get("start_s") or 0.0) * _fps),
                          int(float(a.get("end_s") or 0.0) * _fps),
                          tid, float(a.get("asd_score") or 0.0),
                          bool(a.get("on_screen"))))

    def _verdict(tid: str, frame_idx: int):
        """(colour, label-suffix) for this track at this frame."""
        for f0, f1, t, sc, on in spans:
            if f0 <= frame_idx <= f1:
                if t == tid:
                    return ((0, 220, 0) if on else (60, 60, 230),
                            f"{sc:.2f}" + ("" if on else " off"))
                # Another track owns this moment, so this face is not the
                # one the pipeline believes is talking.
                return (60, 60, 230), "-"
        return (190, 190, 190), "?"

    box_at: Dict[int, list] = {}
    for tr in tracks.tracks:
        fr = np.asarray(tr.frames).astype(int)
        bb = np.asarray(tr.bboxes)
        for i, f in enumerate(fr):
            box_at.setdefault(int(f), []).append((tr.track_id, bb[i][:4]))

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    p["dir"].mkdir(parents=True, exist_ok=True)
    tmp = p["dir"] / "bbox.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    _write_status(job_root, state="running", frames_total=total, frames_done=0)

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            thick = max(2, int(round(w / 960)))
            fs = max(0.5, w / 1920 * 0.8)
            for tid, bb in box_at.get(idx, []):
                x1, y1, x2, y2 = [int(v) for v in bb]
                colour, suffix = _verdict(tid, idx)
                tag = f"{tid} {suffix}"
                spk = spk_of.get(tid)
                if spk:
                    tag += f" {labels.get(spk, spk)}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                              fs, thick)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 8)),
                              (x1 + tw + 6, y1), colour, -1)
                cv2.putText(frame, tag, (x1 + 3, max(th, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick)
            vw.write(frame)
            idx += 1
            if total and idx % 50 == 0:
                _write_status(job_root, state="running", frames_total=total,
                              frames_done=idx)
    finally:
        cap.release()
        vw.release()

    if idx == 0:
        tmp.unlink(missing_ok=True)
        return _write_status(job_root, state="failed",
                             error="decoded no frames from the source clip")
    os.replace(tmp, p["video"])
    log.info("bbox review: %d frames -> %s", idx, p["video"])
    return _write_status(job_root, state="done", frames_total=total,
                         frames_done=idx, threshold=threshold)


def start(job_root: Path, **kw) -> Dict[str, Any]:
    """Kick the render off in a thread, unless one is already going."""
    job_root = Path(job_root)
    key = str(job_root)
    lock = _LOCKS.setdefault(key, threading.Lock())
    if not lock.acquire(blocking=False):
        return status(job_root)

    def _run():
        try:
            render(job_root, **kw)
        except Exception as e:                          # noqa: BLE001
            log.exception("bbox review failed for %s", job_root.name)
            _write_status(job_root, state="failed", error=f"{type(e).__name__}: {e}")
        finally:
            lock.release()

    _write_status(job_root, state="running", frames_done=0)
    threading.Thread(target=_run, daemon=True, name="bbox-review").start()
    return status(job_root)
