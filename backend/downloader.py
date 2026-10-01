"""YouTube (and any yt-dlp-supported site) downloader."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _ffmpeg_dir() -> Optional[str]:
    """Directory containing an ffmpeg binary that yt-dlp can call.

    yt-dlp's postprocessors need ffmpeg. Prefer the system binary; fall back
    to the one imageio-ffmpeg bundles so `sudo apt install ffmpeg` is not a
    prerequisite for the pipeline to work.
    """
    if shutil.which("ffmpeg"):
        return None   # let yt-dlp find it on PATH
    try:
        import imageio_ffmpeg
        return str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    except Exception:
        return None

DEFAULT_FORMAT = (
    # Prefer H.264/AAC mp4 (universally browser-playable) so downstream
    # gr.Video and simple ffmpeg pipelines never see a webm/mkv container.
    # Fall back to best-video+best-audio if no compliant mp4 pair exists.
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
) -> Path:
    """Download a YouTube (or other yt-dlp-supported) video to `out_dir`.

    Args:
      playlist: if False (default), treat a watch-URL that carries a `list=`
        parameter as a single video — the yt-dlp default is to expand it and
        download every entry, which is almost never what a caller wants here.
      force_mp4: if True (default), remux/transcode the final file to mp4 so
        that gr.Video in the frontend can always play it back inline. Turn off
        if you only care about the audio and want to skip the extra ffmpeg pass.

    Returns the local path of the downloaded (and possibly remuxed) file.
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
        # A watch-URL with a `list=` query param defaults to downloading the
        # whole playlist. Suppress that unless the caller explicitly opts in.
        "noplaylist": not playlist,
        # Ask for a single merged file whenever a split format is chosen.
        "merge_output_format": "mp4",
    }
    if force_mp4:
        # Remux to mp4 when codecs allow, otherwise transcode. Guarantees the
        # returned path is browser-playable.
        ydl_opts["postprocessors"] = [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ]

    ff_dir = _ffmpeg_dir()
    if ff_dir is not None:
        ydl_opts["ffmpeg_location"] = ff_dir

    log.info("yt-dlp: downloading %s -> %s (playlist=%s)", url, out_dir, playlist)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Playlists — only reachable when the caller opted in.
        if info.get("_type") == "playlist":
            entries = [e for e in info.get("entries") or [] if e]
            if not entries:
                raise RuntimeError(f"No downloadable entries in playlist {url}")
            info = entries[0]

        # yt-dlp records the true output path (after post-processing) in
        # `requested_downloads[0]['filepath']`. Fall back to prepare_filename
        # and a glob probe when that key is missing.
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
    return filename


def resolve_input(input_: str, download_dir: Path) -> Path:
    """Accept a URL or a local path; return a local file path."""
    if is_url(input_):
        return download_youtube(input_, download_dir)
    p = Path(input_).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local input not found: {p}")
    return p
