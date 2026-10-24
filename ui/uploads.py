"""Local video upload — the other way a job gets its source.

Everything else here starts from a URL that `check_url` has matched against
an allowlist, and `ytmeta.probe` reads the title and duration without
downloading. An uploaded file has neither: no host to allow, and nothing
yt-dlp can describe. So this module supplies both halves.

**Why the allowlist is not simply widened.** `check_url` exists as an SSRF
control: it decides which hosts this server will *fetch from*. An upload is
not a fetch — the bytes arrive in the request body and the server never
opens an outbound connection — so it is not the control's business, and
loosening it to let a `file://` through would turn an SSRF guard into a
local-file-read. Uploads therefore bypass `check_url` entirely and are
constrained by their own rules instead:

  * the file lands in `<job>/source/` under a name this module chooses, so
    a crafted filename cannot traverse out of the job folder;
  * the extension must be one of `VIDEO_EXTS`;
  * the size is capped (`DUBBING_UI_MAX_UPLOAD_MB`, default 4096);
  * `ffprobe` has to agree it holds a decodable video stream, which is the
    only check that looks at the bytes rather than at the name.

The probe is `ffprobe`, not yt-dlp, and it reads the *running interpreter's*
ffprobe via `media_io._ffprobe_bin()` — CLAUDE.md section 1 is explicit that
a bare `ffprobe` is not on PATH here and that the old `None` answer was
silent, reporting `dur=0.0` for a file that plays. A duration of 0 is
therefore treated as a failure to read the file, not as an empty video.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict

from ._io import env_int as _env_int

log = logging.getLogger("ui.uploads")

#: Containers worth accepting. Deliberately short: every one of these is
#: something ffmpeg on this host decodes without a extra codec package.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def max_upload_mb() -> int:
    return max(1, _env_int("DUBBING_UI_MAX_UPLOAD_MB", 4096))


def safe_stem(filename: str) -> str:
    """A filename the server chose, derived from the one it was given.

    The uploaded name is *data*: it reaches us from a browser and may hold
    separators, `..`, or nothing usable at all. Only the extension is taken
    from it after validation, and the stem is sanitised and truncated, so
    the destination path cannot leave `<job>/source/`.
    """
    stem = Path(filename or "").stem
    stem = _SAFE.sub("-", stem).strip("-.")[:60]
    return stem or f"upload-{uuid.uuid4().hex[:8]}"


def check_ext(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise ValueError(
            f"{ext or 'that file'} is not a video this server reads. "
            f"Use one of: {', '.join(VIDEO_EXTS)}.")
    return ext


def probe_file(path: Path) -> Dict[str, Any]:
    """Title, duration and dimensions for a local video, via ffprobe.

    Raises `RuntimeError` with something readable rather than returning a
    zero-filled dict: a 0.0 duration here would bound the range picker at
    nothing and quote a 0-second render.
    """
    from backend.media_io import _ffprobe_bin

    ff = _ffprobe_bin()
    if not ff:
        raise RuntimeError(
            "ffprobe was not found next to the running interpreter, so the "
            "upload cannot be measured. See CLAUDE.md section 1.")
    cmd = [str(ff), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe could not read that file: "
            f"{(proc.stderr or '').strip()[:300]}")
    try:
        meta = json.loads(proc.stdout or "{}")
    except ValueError as e:
        raise RuntimeError(f"ffprobe returned unreadable JSON: {e}") from None

    video = next((s for s in meta.get("streams") or []
                  if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("That file has no video stream.")

    dur = 0.0
    for src in (meta.get("format") or {}, video):
        try:
            dur = float(src.get("duration") or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        if dur > 0:
            break
    if dur <= 0:
        raise RuntimeError(
            "ffprobe read that file but reported no duration, so it cannot "
            "be cut into a window. Re-encode it and try again.")

    return {
        "duration_s": round(dur, 3),
        "width": video.get("width"),
        "height": video.get("height"),
        "vcodec": video.get("codec_name"),
        "has_audio": any(s.get("codec_type") == "audio"
                         for s in meta.get("streams") or []),
        "size_bytes": path.stat().st_size,
    }
