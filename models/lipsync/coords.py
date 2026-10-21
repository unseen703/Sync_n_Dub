"""Per-frame face-box cache, shared by the lip-sync adapters.

Face detection is the expensive preprocessing step: MuseTalk runs DWPose
landmarks plus S3FD over every frame before a single mouth is generated, and
Wav2Lip runs S3FD over every frame too. Both recompute it on every run, so
re-rendering the same clip -- which is what iterating on a voice, a tone or a
batch size actually does -- pays for it again each time.

MuseTalk already supports `--use_saved_coord` / `--saved_coord`, but writes
the pickle to `<result_dir>/../<basename>.pkl`, which for us is inside a
`TemporaryDirectory` and is destroyed with it. Wav2Lip has no caching at all.

WHY THE KEY IS NOT THE FILENAME
-------------------------------
The obvious key -- the video's basename -- is actively dangerous here. Every
sharded run writes `part00.mp4` and `part01.mp4`, so a basename key would
serve one job's boxes to a different job's frames. The boxes would be
plausible, the render would succeed, and the mouths would be drawn in the
wrong places: the same shape as the voice-reference trap in CLAUDE.md, where
a clip filed under the wrong name produced a confident, wrong answer.

So the key is content-ish (size + mtime_ns) plus every parameter that moves a
box. A hit is additionally validated against the frame count before use, and
a mismatch recomputes rather than guessing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

#: Cache root. Inside `data/` so it is covered by the repo's own layout
#: conventions and can be deleted wholesale without touching a job folder.
DEFAULT_CACHE_DIR = Path("data/cache/lipsync_coords")


def cache_dir() -> Path:
    raw = os.getenv("DUBBING_COORD_CACHE_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_CACHE_DIR


def enabled() -> bool:
    """On by default; `DUBBING_COORD_CACHE=0` turns it off."""
    return os.getenv("DUBBING_COORD_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def key_for(video: Path, model: str, params: dict) -> str:
    """Stable id for (this exact file, these box-affecting parameters).

    size + mtime_ns rather than a content hash: the inputs here are hundreds
    of megabytes and are written once by our own ffmpeg cut, so a full read
    on every run would cost more than the detection it saves. The trade-off
    is that a file rewritten with an identical size AND identical mtime would
    hit a stale entry; the frame-count check below is the backstop.
    """
    try:
        st = Path(video).stat()
        stamp = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        stamp = "nostat"
    payload = json.dumps({"v": str(Path(video).name), "s": stamp,
                          "m": model, "p": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _paths(video: Path, model: str, params: dict):
    root = cache_dir() / model
    k = key_for(video, model, params)
    return root / f"{k}.pkl", root / f"{k}.json"


def load(video: Path, model: str, params: dict,
         expect_frames: Optional[int] = None) -> Optional[Any]:
    """Cached coords, or None. Validates the frame count when given one."""
    if not enabled():
        return None
    pkl, meta_p = _paths(Path(video), model, params)
    if not pkl.is_file():
        return None
    try:
        coords = pickle.loads(pkl.read_bytes())
        meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    except Exception as e:                               # noqa: BLE001
        log.warning("coord cache: unreadable entry %s (%s); recomputing",
                    pkl.name, e)
        return None
    n = meta.get("frames")
    if expect_frames is not None and n is not None and n != expect_frames:
        # Never silently reuse boxes computed for a different frame count:
        # that is how one clip's boxes end up drawn on another clip's frames.
        log.warning("coord cache: %s holds %s frames but this clip has %d; "
                    "recomputing", pkl.name, n, expect_frames)
        return None
    log.info("coord cache: hit (%s, %s frames) -- skipping face detection",
             pkl.name, n if n is not None else "?")
    return coords


def save(video: Path, model: str, params: dict, coords: Any,
         frames: Optional[int] = None) -> Optional[Path]:
    """Store coords. Best-effort: a cache failure must never fail a render."""
    if not enabled():
        return None
    pkl, meta_p = _paths(Path(video), model, params)
    try:
        pkl.parent.mkdir(parents=True, exist_ok=True)
        tmp = pkl.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(coords))
        os.replace(tmp, pkl)                 # atomic: no half-written entry
        meta_p.write_text(json.dumps(
            {"source": str(video), "model": model, "params": params,
             "frames": frames}, indent=2))
        log.info("coord cache: stored %s (%s frames)",
                 pkl.name, frames if frames is not None else "?")
        return pkl
    except Exception as e:                               # noqa: BLE001
        log.warning("coord cache: could not store (%s); continuing", e)
        return None
