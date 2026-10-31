"""ASD review video: one labelled box per tracked face per frame.

A review artifact only — nothing downstream reads it, so it shows ALL faces,
including the ones lip-sync would ignore. Rendering is a full decode pass
(minutes at 1080p), so it runs in a thread and reports through a status file.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

REVIEW_SUBDIR = "review"
STATUS_NAME = "bbox.status.json"
VIDEO_NAME = "bbox.mp4"

#: Matches `asd_on_screen_threshold` and `lipsync_targets`.
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
    # The file is the arbiter, not the status doc: a killed render leaves a
    # stale "running" behind forever.
    ready = p["video"].is_file() and p["video"].stat().st_size > 0
    state = doc.get("state") or ("done" if ready else "absent")
    if state == "done" and not ready:
        state = "failed"
        doc.setdefault("error", "render reported done but wrote no video")
    return {**doc, "state": state, "ready": ready}


def source_clip(job_root: Path, source: Optional[Path] = None) -> Optional[Path]:
    """The clip the boxes are drawn over."""
    src = source or next(iter(sorted(Path(job_root).glob("videos/*.mp4"))), None)
    return Path(src) if src and Path(src).is_file() else None


def load_overlay(job_root: Path, src: Path):
    """Return `(box_at, spk_of, spans, fps)` — everything needed to mark a frame.

    Stage 6 persists per-segment assignments
    (`asd/*.assignments.json`), not a per-frame score array: a track is drawn
    as speaking over the frames of a segment assigned to it and marked
    on-screen, and frames outside any segment stay grey.
    """
    import cv2
    import numpy as np

    from backend.faces.faces import FaceTracks

    job_root = Path(job_root)
    npz = next(iter(sorted(job_root.glob("faces/*.npz"))), None)
    if npz is None:
        raise FileNotFoundError("no face tracks: stage 5 did not run")

    probe = cv2.VideoCapture(str(src))
    fps = probe.get(cv2.CAP_PROP_FPS) or 25.0
    probe.release()

    spk_of: Dict[str, str] = {}
    spans: list = []                      # (f_start, f_end, track_id, score, on)
    asd_json = next(iter(sorted(job_root.glob("asd/*.assignments.json"))), None)
    if asd_json is not None:
        try:
            recs = json.loads(asd_json.read_text())
        except (OSError, ValueError):
            recs = []
        for a in recs:
            tid = a.get("face_track")
            if not tid:
                continue
            tid = str(tid)
            spk_of.setdefault(tid, str(a.get("speaker") or ""))
            spans.append((int(float(a.get("start_s") or 0.0) * fps),
                          int(float(a.get("end_s") or 0.0) * fps),
                          tid, float(a.get("asd_score") or 0.0),
                          bool(a.get("on_screen"))))

    tracks = FaceTracks.load_npz(npz)
    box_at: Dict[int, list] = {}
    for tr in tracks.tracks:
        fr = np.asarray(tr.frames).astype(int)
        bb = np.asarray(tr.bboxes)
        for i, f in enumerate(fr):
            box_at.setdefault(int(f), []).append((tr.track_id, bb[i][:4]))
    return box_at, spk_of, spans, fps


def verdict(spans: list, tid: str, frame_idx: int):
    """Return `(colour, label-suffix)` for this track at this frame.

    Green = speaking and on screen, red = someone else owns this moment or
    this track is off screen, grey = no segment covers this frame.
    """
    for f0, f1, t, sc, on in spans:
        if f0 <= frame_idx <= f1:
            if t == tid:
                return ((0, 220, 0) if on else (60, 60, 230),
                        f"{sc:.2f}" + ("" if on else " off"))
            # Another track owns this moment.
            return (60, 60, 230), "-"
    return (190, 190, 190), "?"


def draw_frame(frame, idx: int, *, box_at, spk_of, spans,
               labels: Optional[Dict[str, str]] = None):
    """Mark one decoded frame in place, and return it."""
    import cv2

    labels = labels or {}
    h, w = frame.shape[:2]
    thick = max(2, int(round(w / 960)))
    fs = max(0.5, w / 1920 * 0.8)
    for tid, bb in box_at.get(idx, []):
        x1, y1, x2, y2 = [int(v) for v in bb]
        colour, suffix = verdict(spans, tid, idx)
        tag = f"{tid} {suffix}"
        spk = spk_of.get(tid)
        if spk:
            tag += f" {labels.get(spk, spk)}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)),
                      (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, tag, (x1 + 3, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick)
    return frame


def render(job_root: Path, *, source: Optional[Path] = None,
           threshold: float = SPEAK_THRESHOLD,
           max_edge: Optional[int] = None,
           labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Draw the boxes. Blocking; `start()` is the background wrapper.

    `max_edge` downscales the output only — boxes are drawn at source
    resolution and the marked frame is shrunk after, so labels scale with the
    picture. Left at source size, 4K output runs to hundreds of MB a minute
    through cv2's mp4v encoder.
    """
    import cv2

    job_root = Path(job_root)
    p = paths(job_root)
    labels = labels or {}

    # Check face tracks before the clip: the common case is a job that has its
    # video but never ran stage 5, and the clip error misdirects there.
    if not any(job_root.glob("faces/*.npz")):
        return _write_status(job_root, state="failed",
                             error="no face tracks: stage 5 did not run")
    src = source_clip(job_root, source)
    if src is None:
        return _write_status(job_root, state="failed",
                             error="no source clip in videos/")
    try:
        box_at, spk_of, spans, _fps = load_overlay(job_root, src)
    except FileNotFoundError as e:
        return _write_status(job_root, state="failed", error=str(e))

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Both sides even: the writer accepts odd ones but downstream H.264
    # transcodes do not (4:2:0 subsamples by two).
    scale = min(1.0, max_edge / float(max(w, h))) if max_edge else 1.0
    ow, oh = (w, h) if scale >= 1.0 else (int(w * scale) & ~1, int(h * scale) & ~1)
    p["dir"].mkdir(parents=True, exist_ok=True)
    tmp = p["dir"] / "bbox.tmp.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    _write_status(job_root, state="running", frames_total=total, frames_done=0)

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = draw_frame(frame, idx, box_at=box_at, spk_of=spk_of,
                               spans=spans, labels=labels)
            if (ow, oh) != (w, h):
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
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


# ---------------------------------------------------------------------------
# running it by hand
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Render one job's ASD review video from the command line.

        $PY -m server.utils.bbox_video --job <job_id> --max-edge 1280

    Calls `render` directly and blocks, unlike the UI's threaded route.
    Nothing is recomputed — face tracks and ASD assignments are read off disk
    as the run left them.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Render a job's ASD review video.")
    ap.add_argument("--job", required=True, help="job id under <data>/jobs/")
    ap.add_argument("--data-dir",
                    default=str(Path(__file__).resolve().parent.parent.parent / "data"))
    ap.add_argument("--max-edge", type=int, default=1280,
                    help="downscale the output's longest edge (0 = source "
                         "size). A 4K review video is hundreds of MB and "
                         "shows nothing extra.")
    ap.add_argument("--source", default=None,
                    help="override the clip (default: <job>/videos/*.mp4)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    job_root = Path(args.data_dir) / "jobs" / args.job
    if not job_root.is_dir():
        print(f"no such job: {job_root}", file=sys.stderr)
        return 2

    try:
        from ..service.speakers import read_labels
        labels = read_labels(job_root)
    except Exception:                                   # noqa: BLE001
        labels = {}

    t0 = time.time()
    st = render(job_root, source=Path(args.source) if args.source else None,
                max_edge=args.max_edge or None, labels=labels)
    if st.get("state") != "done":
        print(f"failed: {st.get('error')}", file=sys.stderr)
        return 1

    out = paths(job_root)["video"]
    size = out.stat().st_size if out.is_file() else 0
    print(f"\n{out}")
    print(f"{st.get('frames_done')} frames · {size / 1e6:.1f} MB · "
          f"{time.time() - t0:.1f}s")
    # mp4v, not H.264: `cv2.VideoWriter` has no H.264 encoder in this build,
    # so VLC and mpv play the file but browsers will not.
    print("codec: mp4v — plays in VLC/mpv. For a browser:\n"
          f"  ffmpeg -i {out} -c:v libx264 -pix_fmt yuv420p "
          f"{out.with_name('bbox.h264.mp4')}")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
