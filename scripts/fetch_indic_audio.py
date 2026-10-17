"""Download Indic-language audio tracks from a YouTube video.

Many YouTube videos ship with multiple audio tracks (the original plus
official dubs). This script enumerates the available audio-only formats,
keeps the ones whose `language` tag matches a known Indic language, and
writes each selected track to disk as a separate file.

Usage:
    python scripts/fetch_indic_audio.py <URL> [--out data/indic_audio]
                                              [--langs hi,ta,te]
                                              [--format m4a|mp3|wav|opus]
                                              [--list-only]
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("fetch_indic_audio")

# ISO-639-1/639-3 codes YouTube uses for Indian-subcontinent dubs.
# Keep this as the authoritative default set; `--langs` overrides it.
INDIC_LANGS: Dict[str, str] = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
    "ne": "Nepali",
    "si": "Sinhala",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "bho": "Bhojpuri",
    "kok": "Konkani",
    "brx": "Bodo",
    "sat": "Santali",
    "mni": "Manipuri",
    "doi": "Dogri",
}


def _ffmpeg_dir() -> Optional[str]:
    if shutil.which("ffmpeg"):
        return None
    try:
        import imageio_ffmpeg
        return str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    except Exception:
        return None


def _normalise_lang(code: Optional[str]) -> Optional[str]:
    """Reduce 'hi-IN', 'hi_IN', 'Hindi (India)' etc. to a lowercase base code."""
    if not code:
        return None
    base = re.split(r"[-_ (]", code.strip(), maxsplit=1)[0].lower()
    return base or None


def list_audio_tracks(url: str) -> List[dict]:
    """Return every audio-only format yt-dlp reports for `url`."""
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp not installed. `pip install yt-dlp`") from e

    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = info.get("formats") or []
    audio_only = [
        f for f in formats
        if (f.get("vcodec") in (None, "none")) and f.get("acodec") not in (None, "none")
    ]
    return audio_only


def pick_indic_tracks(
    tracks: List[dict], wanted: Iterable[str]
) -> Dict[str, dict]:
    """Group audio tracks by Indic language code; keep the best per language.

    "Best" = highest reported `abr` (audio bitrate), tie-broken by `filesize`.
    """
    wanted_set = {w.lower() for w in wanted}
    best: Dict[str, dict] = {}
    for f in tracks:
        lang = _normalise_lang(f.get("language"))
        if lang is None or lang not in wanted_set:
            continue
        prev = best.get(lang)
        if prev is None:
            best[lang] = f
            continue
        key_new = (f.get("abr") or 0, f.get("filesize") or f.get("filesize_approx") or 0)
        key_old = (prev.get("abr") or 0, prev.get("filesize") or prev.get("filesize_approx") or 0)
        if key_new > key_old:
            best[lang] = f
    return best


def download_track(
    url: str,
    format_id: str,
    lang: str,
    out_dir: Path,
    *,
    audio_format: str = "m4a",
) -> Path:
    """Download one audio format id and transcode it to `audio_format`."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    # %(id)s.<lang>.%(ext)s — language tag survives the postprocessor rename.
    outtmpl = str(out_dir / f"%(id)s.{lang}.%(ext)s")

    opts = {
        "format": format_id,
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "overwrites": False,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": "0",
        }],
    }
    ff_dir = _ffmpeg_dir()
    if ff_dir is not None:
        opts["ffmpeg_location"] = ff_dir

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        for rd in info.get("requested_downloads") or []:
            fp = rd.get("filepath") or rd.get("_filename")
            if fp:
                return Path(fp)
        # Fallback: scan out_dir for the <id>.<lang>.<ext> we just produced.
        vid = info.get("id") or ""
        for cand in out_dir.glob(f"{vid}.{lang}.*"):
            if cand.suffix.lstrip(".").lower() == audio_format.lower():
                return cand
    raise RuntimeError(f"Could not resolve downloaded file for lang={lang}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="YouTube (or any yt-dlp-supported) video URL")
    ap.add_argument("--out", default=str(ROOT / "data" / "indic_audio"),
                    help="Output directory for audio files")
    ap.add_argument("--langs", default=",".join(INDIC_LANGS.keys()),
                    help=f"Comma-separated ISO codes to accept (default: all known Indic). "
                         f"Known: {', '.join(sorted(INDIC_LANGS))}")
    ap.add_argument("--format", default="m4a",
                    choices=("m4a", "mp3", "wav", "opus", "flac", "aac"),
                    help="Target audio container (ffmpeg transcode)")
    ap.add_argument("--list-only", action="store_true",
                    help="Only enumerate per-language audio tracks; do not download")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-7s %(message)s",
    )

    wanted = [w.strip().lower() for w in args.langs.split(",") if w.strip()]
    if not wanted:
        log.error("No languages requested — pass --langs hi,ta,...")
        return 2

    log.info("Probing %s for audio tracks ...", args.url)
    tracks = list_audio_tracks(args.url)
    if not tracks:
        log.error("No audio-only formats reported — nothing to extract.")
        return 1

    # Report everything we saw, so the user can tell "no Hindi dub available"
    # apart from "language tag was wrong" if nothing matches below.
    seen = sorted({_normalise_lang(f.get("language")) or "?" for f in tracks})
    log.info("Audio-track languages reported by source: %s", ", ".join(seen))

    picks = pick_indic_tracks(tracks, wanted)
    if not picks:
        log.warning("No Indic dubs found. Requested: %s", ", ".join(wanted))
        return 0

    for lang, f in sorted(picks.items()):
        name = INDIC_LANGS.get(lang, lang)
        abr = f.get("abr")
        size = f.get("filesize") or f.get("filesize_approx")
        log.info("  %s (%s): format_id=%s codec=%s abr=%s size=%s",
                 lang, name, f.get("format_id"), f.get("acodec"), abr, size)

    if args.list_only:
        return 0

    out_dir = Path(args.out)
    written: List[Path] = []
    for lang, f in sorted(picks.items()):
        try:
            path = download_track(args.url, f["format_id"], lang, out_dir,
                                  audio_format=args.format)
        except Exception as e:
            log.exception("Failed to download %s track: %s", lang, e)
            continue
        log.info("saved %s (%.2f MB)", path, path.stat().st_size / 1e6)
        written.append(path)

    log.info("Done. %d/%d tracks saved under %s", len(written), len(picks), out_dir)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
