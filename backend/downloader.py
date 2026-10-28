"""YouTube (and any yt-dlp-supported site) downloader."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

# Supported subtitle formats; prefer VTT for its cue timings.
SUBTITLE_EXTS = (".vtt", ".srt")

#: Metadata saved beside the downloaded video. Treat uploader-written text as
#: untrusted input; `categories` is YouTube's own classification field.
METADATA_FIELDS = (
    "id", "title", "description", "categories", "tags",
    "channel", "channel_id", "uploader", "channel_follower_count",
    "duration", "upload_date", "language", "age_limit",
    "availability", "live_status", "view_count", "webpage_url",
)


def video_metadata(info: Dict) -> Dict:
    """Extract selected metadata; missing fields are set to None.

    Chapters are reduced to titles and start times. Treat uploader text as
    untrusted data, not as instructions to a model.
    """
    meta = {k: info.get(k) for k in METADATA_FIELDS}
    chapters = info.get("chapters") or []
    meta["chapters"] = [
        {"title": c.get("title"), "start_time": c.get("start_time")}
        for c in chapters if isinstance(c, dict)
    ] or None
    return meta


def _ffmpeg_dir() -> Optional[str]:
    """Return the directory yt-dlp should search for ffmpeg, if needed.

    Prefer system ffmpeg; otherwise use the binary bundled by imageio-ffmpeg.
    """
    if shutil.which("ffmpeg"):
        return None   # let yt-dlp find it on PATH
    try:
        import imageio_ffmpeg
        return str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    except Exception:
        return None

DEFAULT_FORMAT = (
    # Prefer browser-friendly MP4; fall back if no suitable MP4 pair exists.
    "best[ext=mp4][acodec!=none][vcodec!=none]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best"
)


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def download_youtube(
    url: str,
    out_dir: Path,
    *,
    format_spec: str = DEFAULT_FORMAT,
    filename_template: str = "%(id)s.%(ext)s",
    overwrite: bool = False,
    quiet: bool = False,
    playlist: bool = False,
    force_mp4: bool = True,
    subtitle_langs: Optional[Iterable[str]] = None,
    subtitles_out: Optional[Dict[str, Dict[str, str]]] = None,
    metadata_out: Optional[Dict] = None,
    write_metadata: bool = True,
) -> Path:
    """Download a video, with optional subtitles and metadata.

    Returns the downloaded file path. `playlist` enables playlist downloads;
    `force_mp4` converts the output to MP4. Optional output dictionaries are
    populated with subtitle and metadata details.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp not installed. `pip install yt-dlp`") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / filename_template)

    ydl_opts = {
        "format": format_spec,
        "outtmpl": outtmpl,
        "quiet": quiet,
        "no_warnings": quiet,
        "noprogress": quiet,
        "overwrites": overwrite,
        # Avoid downloading a whole playlist unless requested.
        "noplaylist": not playlist,
        # Merge split streams into one MP4.
        "merge_output_format": "mp4",
        # Media-host 403s can be transient; retry downloads and DASH fragments.
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "retry_sleep_functions": {
            # Back off between retries, with a maximum delay.
            "http": lambda n: min(2 ** n, 60),
            "fragment": lambda n: min(2 ** n, 30),
            "extractor": lambda n: min(2 ** n, 30),
        },
        # Let retries handle stalled connections instead of hanging indefinitely.
        "socket_timeout": 30,
    }
    if force_mp4:
        # Convert the result to browser-compatible MP4.
        ydl_opts["postprocessors"] = [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ]

    langs: List[str] = [l.strip() for l in (subtitle_langs or []) if l and l.strip()]
    if langs:
        # Prefer manual subtitles; use auto-generated captions as a fallback.
        ydl_opts.update({
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": langs,
            "subtitlesformat": "vtt/best",
            "skip_download": False,
        })

    ff_dir = _ffmpeg_dir()
    if ff_dir is not None:
        ydl_opts["ffmpeg_location"] = ff_dir

    log.info("yt-dlp: downloading %s -> %s (playlist=%s)", url, out_dir, playlist)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Use the first entry when a playlist was requested.
        if info.get("_type") == "playlist":
            entries = [e for e in info.get("entries") or [] if e]
            if not entries:
                raise RuntimeError(f"No downloadable entries in playlist {url}")
            info = entries[0]

        # Prefer yt-dlp's final output path; fall back to filename lookup.
        filename: Optional[Path] = None
        for rd in info.get("requested_downloads") or []:
            fp = rd.get("filepath") or rd.get("_filename")
            if fp:
                filename = Path(fp)
                break
        if filename is None:
            filename = Path(ydl.prepare_filename(info))
        if not filename.exists():
            for candidate in out_dir.glob(f"{filename.stem}.*"):
                if candidate.is_file():
                    filename = candidate
                    break

    if not filename.exists():
        raise RuntimeError(
            f"Download reported success but file not found under {out_dir}. "
            f"Expected stem {filename.stem!r}."
        )
    log.info("yt-dlp: saved %s (%.1f MB)", filename, filename.stat().st_size / 1e6)

    if subtitles_out is not None and langs:
        # Record each subtitle's path and whether it was manual or auto-generated.
        req_subs = info.get("requested_subtitles") or {}
        manual = set((info.get("subtitles") or {}).keys())
        for lang, sub in req_subs.items():
            fp = sub.get("filepath")
            if not fp:
                continue
            path = Path(fp)
            if not path.exists():
                for cand in out_dir.glob(f"{filename.stem}*.{lang}.*"):
                    if cand.suffix.lower() in SUBTITLE_EXTS:
                        path = cand
                        break
            if path.exists():
                subtitles_out[lang] = {
                    "path": str(path.resolve()),
                    "source": "manual" if lang in manual else "auto",
                }
        missing = [l for l in langs if l not in subtitles_out]
        if missing:
            log.info("yt-dlp: no subtitles found for lang(s): %s", ", ".join(missing))

    if metadata_out is not None or write_metadata:
        meta = video_metadata(info)
        if metadata_out is not None:
            metadata_out.update(meta)
        if write_metadata:
            meta_path = filename.with_suffix(".meta.json")
            try:
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
                )
                log.info("yt-dlp: metadata -> %s (categories=%s)",
                         meta_path.name, meta.get("categories"))
            except OSError as e:
                # Metadata sidecar failure should not fail the download.
                log.warning("yt-dlp: could not write %s: %s", meta_path.name, e)

    return filename


def find_local_subtitles(video: Path, langs: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Find language-tagged subtitle files beside `video`.

    Also accept a bare `.vtt` or `.srt` when only one language is requested.
    """
    out: Dict[str, Dict[str, str]] = {}
    stem = video.stem
    folder = video.parent
    lang_list = [l.strip() for l in langs if l and l.strip()]
    for lang in lang_list:
        for ext in SUBTITLE_EXTS:
            cand = folder / f"{stem}.{lang}{ext}"
            if cand.exists():
                out[lang] = {"path": str(cand.resolve()), "source": "local"}
                break
    # Accept an untagged subtitle when only one language was requested.
    if len(lang_list) == 1 and lang_list[0] not in out:
        for ext in SUBTITLE_EXTS:
            cand = folder / f"{stem}{ext}"
            if cand.exists():
                out[lang_list[0]] = {"path": str(cand.resolve()), "source": "local"}
                break
    return out


def resolve_input(
    input_: str,
    download_dir: Path,
    *,
    subtitle_langs: Optional[Iterable[str]] = None,
    subtitles_out: Optional[Dict[str, Dict[str, str]]] = None,
    metadata_out: Optional[Dict] = None,
) -> Path:
    """Resolve a URL or local path and optionally collect subtitles."""
    langs = list(subtitle_langs or [])
    if is_url(input_):
        return download_youtube(
            input_, download_dir,
            subtitle_langs=langs or None,
            subtitles_out=subtitles_out,
            metadata_out=metadata_out,
        )
    p = Path(input_).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local input not found: {p}")
    if subtitles_out is not None and langs:
        subtitles_out.update(find_local_subtitles(p, langs))
    return p
