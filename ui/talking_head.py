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
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._io import data_dir as _data_dir

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = REPO_ROOT / "data" / "talking_head" / "dialogues.json"

#: Named in the "wrong interpreter" message below. `DUBBING_PY`, when set
#: (`ui.server.main()` defaults it to its own `sys.executable`), is trusted
#: first; otherwise derived the way `models/lipsync/base.py::resolve_env_python`
#: derives a sibling env — from the conda prefix two levels above whichever
#: python is running — so it stays right if the install moves. None rather
#: than a literal `/data1/...` fallback: on any install where the sibling
#: derivation fails too, that path does not exist on the operator's machine
#: either, and a message pointing at it is worse than one admitting it does
#: not know.
_sibling_dubbing_py = (Path(sys.executable).resolve().parent.parent.parent
                        / "dubbing" / "bin" / "python")
_DUBBING_PY = (
    os.environ.get("DUBBING_PY")
    or (str(_sibling_dubbing_py) if _sibling_dubbing_py.exists() else None))

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
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        # NOT the same failure as a detector that loads and then misbehaves,
        # and the two need different things from whoever reads it. This one is
        # "the server is running in the wrong interpreter", and the only fact
        # that identifies which is `sys.executable` — so say it.
        #
        # Seen on this host 2026-10-21: `python -m ui.server` was started from
        # `.venv-diarization` (a uv venv with torch and cv2 but no
        # insightface) rather than from `envs/dubbing`, and every photo upload
        # answered "ModuleNotFoundError" with nothing pointing at the cause.
        hint = (f"{_DUBBING_PY} -m ui.server" if _DUBBING_PY
                else "the dubbing env's python -m ui.server "
                     "(set DUBBING_PY in .env if it cannot be found)")
        raise ImportError(
            f"insightface is not installed in the interpreter running this "
            f"server ({sys.executable}). Start ui.server from the dubbing "
            f"env: {hint}"
        ) from e
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
    except ImportError as e:
        # The wrong-interpreter case, kept separate because it is the only
        # one the reader can act on and `type(e).__name__` is useless for it:
        # "ModuleNotFoundError" names the exception class, not the mistake.
        # `_detector` has already built the sentence, naming the python that
        # is running and the one that should be, so pass it through verbatim.
        log.error("talking head: %s", e)
        return {"ok": False, "reason": "detector_not_installed", "width": w,
                "height": h, "message": str(e)}
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


# ---------------------------------------------------------------------------
# the render
# ---------------------------------------------------------------------------
#
# Shaped after `ui/bbox_video.py`, which is already this repo's answer to "a
# GPU job a web request cannot wait for": a daemon thread, a status json
# beside the output, and a GET the page polls. A render here is 8-21x the
# clip length, so doing it inline dies to a browser timeout while the work
# carries on -- the worst of both.
#
# MuseTalk is the model, and not a choice: it is the one that takes a single
# image natively (`get_file_type` accepts .jpg/.png and the frame cycle
# reuses the one frame for every audio chunk), verified on this host at
# 1280x720 + 7 s of audio -> 7.0 s at 25 fps. Wav2Lip's vendored S3FD refuses
# a large frame outright and LatentSync wants 512x512 it has no VRAM for.

RENDER_MODEL = "musetalk"
RENDERS_SUBDIR = "renders"
RENDER_STATUS_NAME = "render.status.json"
RENDER_VIDEO_NAME = "out.mp4"
RENDER_FRAME_NAME = "source.png"

#: Where photos, clips and renders live. A module constant so a test can
#: point it somewhere disposable; `ui/routes_talkinghead.py` reads this one
#: rather than keeping a second copy. Goes through the shared `_io.data_dir()`
#: so a relocated `DUBBING_UI_DATA_DIR` moves this page's files along with
#: everything else — this used to be a hardcoded `REPO_ROOT / "data"` that
#: ignored the setting entirely.
DATA_DIR = _data_dir()

_RENDER_LOCKS: Dict[str, Any] = {}


#: A render id is a folder name, and it arrives from the client on the poll.
#: Built here from a uuid prefix and a catalogue id, both of which match
#: this — but `GET /api/talking-head/render/{id}` takes whatever is sent, and
#: `Path / ".."` is not a string operation. Rejecting beats sanitising: a
#: cleaned-up id would read a DIFFERENT render's status and report it as this
#: one's.
_RENDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def render_paths(render_id: str) -> Dict[str, Path]:
    if not _RENDER_ID_RE.match(render_id or ""):
        raise ValueError(f"not a render id: {render_id!r}")
    d = DATA_DIR / "talking_head" / RENDERS_SUBDIR / render_id
    return {"dir": d, "video": d / RENDER_VIDEO_NAME,
            "frame": d / RENDER_FRAME_NAME,
            "status": d / RENDER_STATUS_NAME}


def _write_render_status(render_id: str, **kw) -> Dict[str, Any]:
    import os
    import time

    p = render_paths(render_id)
    p["dir"].mkdir(parents=True, exist_ok=True)
    doc = {"render_id": render_id, "updated_at": time.time(), **kw}
    tmp = p["status"].with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, p["status"])
    return doc


def render_status(render_id: str) -> Dict[str, Any]:
    """State of one render, with the FILE as the arbiter.

    Not the status field. A killed process leaves a stale `running` behind
    forever, and a `done` with no mp4 is the exact failure CLAUDE.md §4
    records for lip-sync: `run_lipsync` used to log "done -> <path>" and
    record the path it had ASKED for, so a 40 s run exited 0 and left a
    manifest naming a file that was never created.
    """
    try:
        p = render_paths(render_id)
    except ValueError:
        return {"render_id": render_id, "state": "absent", "ready": False,
                "error": "not a render id"}
    try:
        doc = json.loads(p["status"].read_text())
    except (OSError, ValueError):
        doc = {}
    ready = p["video"].is_file() and p["video"].stat().st_size > 0
    state = doc.get("state") or ("done" if ready else "absent")
    if state == "done" and not ready:
        state = "failed"
        doc.setdefault("error", "the renderer reported success but wrote no "
                                "video")
    out = {**doc, "render_id": render_id, "state": state, "ready": ready}
    if ready:
        out["url"] = _files_url(p["video"])
    return out


def _files_url(p: Path) -> Optional[str]:
    try:
        return "/files/" + p.resolve().relative_to(
            DATA_DIR.resolve()).as_posix()
    except ValueError:                                  # pragma: no cover
        return None


def render(render_id: str, *, photo: Path, audio: Path,
           model: str = RENDER_MODEL) -> Dict[str, Any]:
    """Animate `photo` against `audio`. Blocking; `start_render` wraps it."""
    from models.lipsync import LipSyncRequest, get_lipsync

    p = render_paths(render_id)
    p["dir"].mkdir(parents=True, exist_ok=True)

    audio = Path(audio)
    if not audio.is_file():
        raise FileNotFoundError(f"no audio clip at {audio}")

    # Downscale and even up the sides BEFORE the model loads. Both failures
    # this guards are post-load: a 4K still makes S3FD return `[None]` and
    # MuseTalk then divides by zero in a print statement, and an odd
    # dimension makes x264 refuse to open the encoder. Either one costs the
    # whole GPU job to discover.
    frame = prepare_image(photo, p["frame"])
    _write_render_status(render_id, state="running", model=model,
                         frame=str(frame), audio=str(audio))

    ls = get_lipsync(model)
    res = ls.sync(LipSyncRequest(video=frame, audio=audio,
                                 out_path=p["video"]))

    # The file is the arbiter, again and for the same reason. `rc=0` with no
    # output is the Wav2Lip shape -- `inference.py` muxes with
    # `subprocess.call` and discards the return code -- so an adapter can
    # finish cleanly and have produced nothing.
    produced = p["video"].is_file() and p["video"].stat().st_size > 0
    if res.error or not produced:
        err = res.error or ("the renderer exited cleanly but wrote no video "
                            f"to {p['video'].name}")
        log.error("talking head: render %s failed: %s", render_id, err)
        return _write_render_status(render_id, state="failed", model=model,
                                    error=err)

    log.info("talking head: render %s -> %s (%.2fs @ %.1f fps)",
             render_id, p["video"], res.duration_s, res.fps)
    return _write_render_status(render_id, state="done", model=model,
                                duration_s=res.duration_s, fps=res.fps,
                                note=ANIMATION_NOTE)


def start_render(render_id: str, *, photo: Path, audio: Path,
                 model: str = RENDER_MODEL) -> Dict[str, Any]:
    """Kick a render off in a thread, unless one is already going for this id."""
    import threading

    lock = _RENDER_LOCKS.setdefault(render_id, threading.Lock())
    if not lock.acquire(blocking=False):
        return render_status(render_id)

    def _run():
        try:
            render(render_id, photo=photo, audio=audio, model=model)
        except Exception as e:                          # noqa: BLE001
            log.exception("talking head: render %s failed", render_id)
            _write_render_status(render_id, state="failed", model=model,
                                 error=f"{type(e).__name__}: {e}")
        finally:
            lock.release()

    _write_render_status(render_id, state="running", model=model)
    threading.Thread(target=_run, daemon=True,
                     name=f"talking-head-{render_id[:8]}").start()
    return render_status(render_id)


def audio_path_for(audio_url: str) -> Optional[Path]:
    """`/files/talking_head/audio/x.wav` -> the file on disk.

    The catalogue stores a URL because the page plays it directly. The
    renderer needs a path, and resolving it has to stay inside `DATA_DIR`:
    these rows are data, and a `../` in one must not read an arbitrary file.
    """
    url = (audio_url or "").strip()
    if not url.startswith("/files/"):
        return None
    candidate = (DATA_DIR / url[len("/files/"):]).resolve()
    try:
        candidate.relative_to(DATA_DIR.resolve())
    except ValueError:
        log.warning("talking head: audio_url escapes the data dir: %r", url)
        return None
    return candidate if candidate.is_file() else None
