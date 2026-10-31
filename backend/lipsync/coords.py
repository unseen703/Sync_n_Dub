"""Per-frame face-box cache, shared by the lip-sync adapters.

Face detection is the expensive preprocessing step and every adapter redoes it
on each run, so re-rendering the same clip pays for it again. MuseTalk's own
`--saved_coord` writes into a TemporaryDirectory that is destroyed with it;
Wav2Lip has no caching at all.

The key is NOT the filename: every sharded run writes `part00.mp4`, so a
basename key would serve one job's boxes for another job's frames and draw
plausible mouths in the wrong places. It is size + mtime_ns plus every
parameter that moves a box, and a hit is validated against the frame count
before use.
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

#: Cache root. Inside `data/` so it can be deleted wholesale without touching
#: a job folder.
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

    size + mtime_ns rather than a content hash: the inputs run to hundreds of
    megabytes, so a full read would cost more than the detection it saves. A
    file rewritten with identical size and mtime would hit a stale entry; the
    frame-count check in `load` is the backstop.
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
        # Never reuse boxes computed for a different frame count — that is
        # how one clip's boxes end up on another clip's frames.
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
