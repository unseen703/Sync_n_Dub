"""Local video upload — the other way a job gets its source.

An upload has no host to allowlist and nothing yt-dlp can describe, so it
bypasses `check_url` (an SSRF control over outbound fetches, which an upload
is not) and is constrained by its own rules: the file lands in `<job>/source/`
under a name this module chooses, the extension must be in `VIDEO_EXTS`, the
size is capped by `DUBBING_UI_MAX_UPLOAD_MB`, and ffprobe must agree it holds
a decodable video stream.

A reported duration of 0 is treated as a failure to read the file, not as an
empty video.
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

#: Containers worth accepting — each decodes on this host's ffmpeg with no
#: extra codec package.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def max_upload_mb() -> int:
    return max(1, _env_int("DUBBING_UI_MAX_UPLOAD_MB", 4096))


def safe_stem(filename: str) -> str:
    """A filename the server chose, derived from the one it was given.

    The uploaded name is data and may hold separators, `..`, or nothing
    usable. The stem is sanitised and truncated so the destination path cannot
    leave `<job>/source/`.
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
    """Duration and dimensions for a local video, via ffprobe.

    Raises `RuntimeError` rather than returning a zero-filled dict — a 0.0
    duration would bound the range picker at nothing and quote a 0-second
    render.
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
