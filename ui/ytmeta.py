"""What YouTube knows about a URL, before the pipeline touches it.

The title is the reason this module exists. `backend.downloader.download_youtube`
sets `filename_template="%(id)s.%(ext)s"` and throws yt-dlp's `info` away, and
with a pre-cut window the pipeline never calls `download_youtube` at all — it
is handed a local clip. So nothing downstream has ever seen the title, and
nothing downstream can. It is probed here, UI-side, before the job starts, and
written to `<job>/meta.json` (which `ui/server.py::_job_meta` already reads).

The probe does triple duty:

  * the **title**, for the job card;
  * **validation** — live/upcoming streams are refused, and the post-probe
    `webpage_url` is re-checked against the host allowlist so a shortener
    cannot bounce past the pre-probe check;
  * the **duration**, which bounds the range picker and the lip-sync cap.

`fetch_thumbnail` caches the poster frame into the job folder so the browser
never reaches googleusercontent — a page served from loopback should not make
the operator's browser talk to a CDN to render a job list.
"""
from __future__ import annotations

import logging
import re
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

log = logging.getLogger("ui.ytmeta")

THUMB_NAME = "meta.thumb.jpg"
META_NAME = "meta.json"

#: Hosts a thumbnail may be fetched from. The URL comes out of yt-dlp's JSON,
#: which is attacker-influenced in exactly the same way the submitted URL is,
#: so it gets the same treatment rather than being trusted because it arrived
#: in a "safe" field.
_THUMB_HOSTS = (
    "ytimg.com", "ggpht.com", "googleusercontent.com",
    "youtube.com", "yt3.ggpht.com",
)

#: yt-dlp will spend a long time on a slow host. The UI is synchronous from
#: the operator's point of view, so cap it.
_SOCKET_TIMEOUT = float(os.environ.get("DUBBING_UI_PROBE_TIMEOUT_S", "20"))


def probe(url: str) -> Dict[str, Any]:
    """Metadata for `url` without downloading a byte.

    Raises `RuntimeError` when yt-dlp is missing or the extraction fails —
    the caller turns that into a 4xx with the message, because "this video is
    private" is something the operator needs to read, not a 500.
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
    # URL still comes back as a playlist. Take its first entry rather than
    # failing on a missing duration three lines later.
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
    }


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def name_from_url(url: str) -> str:
    """A usable name the instant a URL is pasted, before anything is fetched.

    The probe takes a second or two and can fail outright; a job created in
    that window still has to be called something, and "20261017-032044__
    L3oOldViIgY__pyannote.whisperx.mms_fa.bodhan.hi" is not a name. The video
    id is short, stable and recognisable, so it is what stands in until the
    real title arrives.
    """
    u = str(url or "").strip()
    m = _YT_ID_RE.search(u)
    if m:
        return f"YouTube {m.group(1)}"
    tail = u.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
    return tail[:60] or "Untitled video"


def display_name(meta: Dict[str, Any], *, url: str = "") -> str:
    """The name shown on the card: the real title once we have it.

    Order matters and is the whole point — a fetched title replaces the
    URL-derived placeholder, and neither is ever the job id.
    """
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    return name_from_url(url or meta.get("webpage_url") or meta.get("url") or "")


def is_live(meta: Dict[str, Any]) -> bool:
    """True for a stream that has no fixed end — nothing here can dub one."""
    return str(meta.get("live_status") or "") in ("is_live", "is_upcoming", "post_live")


def fetch_thumbnail(meta: Dict[str, Any], job_root: Path) -> Optional[Path]:
    """Cache the poster frame as `<job>/meta.thumb.jpg`.

    Best-effort: a missing thumbnail degrades the card, it does not fail the
    job, so every failure here is a warning and a `None`.
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
