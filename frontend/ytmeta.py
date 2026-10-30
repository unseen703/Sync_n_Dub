"""Probe what YouTube knows about a URL before the pipeline touches it.

Returns the title (for the job card), the duration (which bounds the range
picker and the lip-sync cap), and the live status, written to
`<job>/meta.json`. Nothing downstream sees this — the downloader throws
yt-dlp's `info` away and a pre-cut window never calls it at all.

`fetch_thumbnail` caches the poster frame into the job folder so a loopback
page never makes the browser talk to a CDN.
"""
from __future__ import annotations

import logging
import re
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ._io import env_float as _env_float

log = logging.getLogger("frontend.ytmeta")

THUMB_NAME = "meta.thumb.jpg"
META_NAME = "meta.json"

#: Hosts a thumbnail may be fetched from. The URL comes out of yt-dlp's JSON,
#: which is attacker-influenced exactly as the submitted URL is.
_THUMB_HOSTS = (
    "ytimg.com", "ggpht.com", "googleusercontent.com",
    "youtube.com", "yt3.ggpht.com",
)

#: yt-dlp will spend a long time on a slow host. The UI is synchronous from
#: the operator's point of view, so cap it.
_SOCKET_TIMEOUT = _env_float("DUBBING_UI_PROBE_TIMEOUT_S", 20.0)


def probe(url: str) -> Dict[str, Any]:
    """Metadata for `url` without downloading a byte.

    Raises `RuntimeError` when yt-dlp is missing or extraction fails; the
    caller turns that into a 4xx carrying the message, since "this video is
    private" is for the operator to read.
    """
    try:
        import yt_dlp
    except ImportError as e:                                # pragma: no cover
        raise RuntimeError("yt-dlp not installed. `pip install yt-dlp`") from e

    opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": _SOCKET_TIMEOUT,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise RuntimeError(f"Could not read that video: {e}") from e

    if info is None:
        raise RuntimeError("Could not read that video: yt-dlp returned nothing.")
    # `noplaylist` is honoured by the watch-URL extractor, but a bare playlist
    # URL still comes back as one. Take its first entry.
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise RuntimeError("That link is an empty playlist.")
        info = entries[0]

    dur = info.get("duration")
    return {
        "video_id":      info.get("id"),
        "title":         info.get("title"),
        "uploader":      info.get("uploader") or info.get("channel"),
        "duration_s":    float(dur) if dur is not None else None,
        "thumbnail_url": info.get("thumbnail"),
        "webpage_url":   info.get("webpage_url") or url,
        "live_status":   info.get("live_status"),
        "language":      info.get("language"),
        "original_language": info.get("original_language"),
        "description":   info.get("description"),
    }


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def name_from_url(url: str) -> str:
    """A usable name the instant a URL is pasted, before anything is fetched.

    The probe can fail or take seconds, and a job created meanwhile still needs
    a name that is not its id. The video id stands in until the real title
    arrives.
    """
    u = str(url or "").strip()
    m = _YT_ID_RE.search(u)
    if m:
        return f"YouTube {m.group(1)}"
    tail = u.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
    return tail[:60] or "Untitled video"


def display_name(meta: Dict[str, Any], *, url: str = "") -> str:
    """The name shown on the card: the fetched title, else the URL-derived
    placeholder. Never the job id."""
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    return name_from_url(url or meta.get("webpage_url") or meta.get("url") or "")


def is_live(meta: Dict[str, Any]) -> bool:
    """True for a stream that has no fixed end — nothing here can dub one."""
    return str(meta.get("live_status") or "") in ("is_live", "is_upcoming", "post_live")


def fetch_thumbnail(meta: Dict[str, Any], job_root: Path) -> Optional[Path]:
    """Cache the poster frame as `<job>/meta.thumb.jpg`.

    Best-effort: every failure is a warning and a None, since a missing
    thumbnail only degrades the card.
    """
    url = (meta or {}).get("thumbnail_url")
    if not url:
        return None
    parsed = urlparse(str(url))
    if parsed.scheme != "https":
        log.warning("refusing non-https thumbnail %r", url)
        return None
    host = (parsed.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in _THUMB_HOSTS):
        log.warning("refusing thumbnail from unexpected host %r", host)
        return None

    dest = Path(job_root) / THUMB_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        req = urllib.request.Request(str(url), headers={"User-Agent": "dubbingstudio-ui"})
        with urllib.request.urlopen(req, timeout=10) as r:   # noqa: S310 - host checked above
            if r.status != 200:
                log.warning("thumbnail HTTP %s for %s", r.status, url)
                return None
            tmp = dest.with_suffix(".jpg.tmp")
            with open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 16)
        os.replace(tmp, dest)
        return dest
    except Exception as e:
        log.warning("thumbnail fetch failed (%s): %s", url, e)
        return None
