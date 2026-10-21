"""Talking Head page: a photo, a language, a line — and a face check first.

The dubbing pipeline re-voices an existing performance. This page does the
other thing: it animates a STILL portrait against a chosen clip. MuseTalk
supports a single image natively (`get_file_type` accepts .jpg/.png and the
frame cycle reuses the one frame for every audio chunk), verified on this
host: one 1280x720 PNG plus 7 s of audio produced a 7.0 s / 25 fps render.

TWO MEASURED CONSTRAINTS ARE ENCODED HERE, NOT GUESSED
------------------------------------------------------
1. **A 4K still fails.** Measured 2026-10-19: a 3840x2160 portrait made
   MuseTalk's S3FD return `[None]` -- no face at that size -- and with zero
   detections `average_range_minus` is empty, so a *print statement* at
   `preprocessing.py:135` divides by zero. The render dies after the model
   has loaded. `MAX_EDGE` downscales before that can happen, and the probe
   reports it rather than letting the user discover it in a failed job.
2. **Only the mouth moves.** MuseTalk inpaints the occluded lower face and
   copies everything else through: measured 0.87% of the frame changing
   between output frames. There is no head motion, no blinking, no gaze.
   `ANIMATION_NOTE` says so on the page, because a user expecting SadTalker's
   head movement will read a correct render as a broken one.

The dialogue catalogue is DATA (`data/talking_head/dialogues.json`), not code,
so lines and their audio links can be added without a deploy. Every entry
ships with `audio_url: null` deliberately -- the clips are supplied later, and
an entry without one is offered-but-marked rather than hidden, so adding the
link is the only step that makes it playable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = REPO_ROOT / "data" / "talking_head" / "dialogues.json"

#: Longest edge we hand to the renderer. Above this MuseTalk's S3FD finds no
#: face at all and the run dies on a divide-by-zero; 1280 is the width that
#: was verified working end to end.
MAX_EDGE = 1280

#: Below this a face is too small for a 256x256 crop to be anything but
#: upscaled guesswork, which MuseTalk will inpaint detail into that was never
#: in the source.
MIN_FACE_PX = 96

ANIMATION_NOTE = (
    "MuseTalk animates the mouth only — the head, eyes and background stay "
    "exactly as they are in the photo. For head movement and blinking you "
    "want a portrait-animation model (SadTalker, Hallo), which is not "
    "installed here."
)


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------

def _load() -> Dict[str, Any]:
    try:
        return json.loads(CATALOGUE.read_text())
    except (OSError, ValueError) as e:
        log.warning("talking head: catalogue unreadable (%s)", e)
        return {"categories": [], "dialogues": {}}


def categories() -> List[Dict[str, str]]:
    return list(_load().get("categories") or [])


def dialogues_for(language: str) -> List[Dict[str, Any]]:
    """Lines offered for one language, newest shape first.

    An entry with no `audio_url` is RETURNED, flagged `playable: False`.
    Hiding it would make a half-populated catalogue look empty and give no
    hint that the only missing piece is a link.
    """
    doc = _load()
    rows = doc.get("dialogues", {}).get((language or "").strip().lower()) or []
    out = []
    for r in rows:
        url = r.get("audio_url")
        out.append({
            "id": r.get("id"),
            "category": r.get("category"),
            "text": r.get("text") or "",
            "audio_url": url,
            "playable": bool(url),
        })
    return out


def languages_with_dialogues() -> List[str]:
    return sorted(k for k, v in (_load().get("dialogues") or {}).items() if v)


# ---------------------------------------------------------------------------
# the face check
# ---------------------------------------------------------------------------

_APP = None


def _detector(device: str = "cpu"):
    global _APP
    if _APP is not None:
        return _APP
    try:
        from backend.gpu import ensure_onnxruntime_cuda
        ensure_onnxruntime_cuda()
    except Exception:                                   # noqa: BLE001
        log.debug("talking head: ort cuda preload failed", exc_info=True)
    # `import torch` FIRST, and it is not decoration. cv2's bundled
    # libtbb.so.12 wants CXXABI_1.3.15, which the system
    # /lib/x86_64-linux-gnu/libstdc++.so.6 does not have and the conda one
    # does; whichever library loads first decides which the process gets.
    # Importing insightface on its own raises
    #   libstdc++.so.6: version `CXXABI_1.3.15' not found
    # and torch pulls the conda copy in first (CLAUDE.md section 1).
    #
    # The LD_PRELOAD that normally covers this is exported by
    # `scripts/ui_run_job.sh`, which launches pipeline JOBS -- the UI SERVER
    # is started by hand and has no such wrapper, so the preload is simply
    # absent from its environment. A lazy import inside a request then turned
    # the whole upload into a bare 500 with the reason only in the server's
    # stderr. Verified on this host: `import torch` first and the import
    # succeeds with no preload at all.
    import torch  # noqa: F401
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=("detection",))
    app.prepare(ctx_id=-1 if device == "cpu" else 0, det_size=(640, 640))
    _APP = app
    return _APP


def probe_image(path: Path, *, device: str = "cpu") -> Dict[str, Any]:
    """Is there a usable human face in this picture?

    Answers the page's gate. `ok` is the only thing the UI branches on; the
    rest explains WHY, because "no face found" and "the face is 40 px wide"
    need different things from the user.
    """
    import cv2

    path = Path(path)
    img = cv2.imread(str(path))
    if img is None:
        return {"ok": False, "reason": "unreadable",
                "message": "That file could not be opened as an image."}

    h, w = img.shape[:2]
    try:
        faces = _detector(device).get(img)
    except Exception as e:                              # noqa: BLE001
        # A detector that cannot load is an environment fault, not a bad
        # photo, and the two need different things from whoever reads it.
        # Answering 500 put the reason in the server's stderr only, where
        # the person uploading never sees it.
        log.exception("talking head: face detector unavailable")
        return {"ok": False, "reason": "detector_unavailable", "width": w,
                "height": h,
                "message": f"The face detector could not start on the "
                           f"server ({type(e).__name__}). The photo was not "
                           f"the problem."}
    if not faces:
        return {"ok": False, "reason": "no_face", "width": w, "height": h,
                "message": "No face found in that photo. A clear, "
                           "front-facing portrait works best."}

    # Biggest face: on a group photo it is the subject, and MuseTalk animates
    # exactly one face per frame anyway.
    def area(f):
        x1, y1, x2, y2 = f.bbox[:4]
        return float(x2 - x1) * float(y2 - y1)

    faces = sorted(faces, key=area, reverse=True)
    x1, y1, x2, y2 = [int(v) for v in faces[0].bbox[:4]]
    fw, fh = x2 - x1, y2 - y1

    out = {
        "ok": True, "reason": "ok", "width": w, "height": h,
        "n_faces": len(faces),
        "face_box": [x1, y1, x2, y2],
        "face_px": min(fw, fh),
        "warnings": [],
        "note": ANIMATION_NOTE,
    }
    if max(w, h) > MAX_EDGE:
        # Not fatal: `prepare_image` downscales. Said out loud because the
        # output resolution is not what the user uploaded.
        out["warnings"].append(
            f"This image is {w}x{h}; it will be scaled to {MAX_EDGE}px on its "
            f"longest edge. Above that the renderer's own face detector finds "
            f"nothing and the job fails.")
    if min(fw, fh) < MIN_FACE_PX:
        out["warnings"].append(
            f"The face is only {min(fw, fh)}px across. It is rebuilt at "
            f"256x256, so anything smaller is upscaled and will look soft.")
    if len(faces) > 1:
        out["warnings"].append(
            f"{len(faces)} faces found; the largest one will be animated.")
    return out


def prepare_image(src: Path, dst: Path, *, max_edge: int = MAX_EDGE) -> Path:
    """Write a copy the renderer can actually handle. See MAX_EDGE."""
    import cv2

    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"not an image: {src}")
    h, w = img.shape[:2]
    scale = min(1.0, float(max_edge) / float(max(w, h)))
    if scale < 1.0:
        w, h = int(round(w * scale)), int(round(h * scale))

    # BOTH SIDES MUST BE EVEN. MuseTalk encodes the frames with
    # `libx264 -vf format=yuv420p`, and 4:2:0 chroma subsamples by two, so an
    # odd dimension makes x264 refuse to open the encoder at all:
    #   Could not open encoder before EOF
    #   Nothing was written into output file
    # Upstream then trips over `save_dir_full` (only assigned on the VIDEO
    # input path) while cleaning up, and its broad `except Exception` reports
    # THAT instead -- so the real cause never reaches the log.
    #
    # Found on a real 454x675 upload. Portraits are arbitrary sizes; the
    # 1280x720 test image happened to be even on both sides and hid this
    # completely. Cropping a row/column beats padding: a black edge would be
    # composited into the output frame.
    w, h = w - (w % 2), h - (h % 2)
    if (w, h) != img.shape[1::-1]:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)
    return dst
