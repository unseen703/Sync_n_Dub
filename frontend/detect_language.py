"""Detect the spoken language of a video source.

Uses uploader metadata when present, otherwise sends a few seconds of mono
16 kHz audio to an LLM. Title and description are deliberately never read —
they are untrusted text and often disagree with the speech. Best-effort
throughout: anything that fails returns None and the caller falls back to
`auto`.
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ._io import env_float as _env_float

log = logging.getLogger("frontend.detect_language")

DETECT_MODEL = "google/gemini-3.1-flash-lite"

PROMPT = (
    "Identify the spoken language. "
    "Return only its language name and ISO 639-1 code.\n"
    "Output format: {\"language\": \"<name>\", \"code\": \"<iso-639-1>\"}"
)

#: Where in the video to listen. Thirty seconds in skips intros and preambles,
#: which are the parts most likely to be in a different language from the talk.
CLIP_OFFSET_S = 30.0
CLIP_DURATION_S = 8.0

#: A probe is synchronous for the operator, so the remote clip fetch gets the
#: same cap `frontend.ytmeta` puts on the metadata probe.
_SOCKET_TIMEOUT = _env_float("DUBBING_UI_PROBE_TIMEOUT_S", 20.0)


def _clip_offset(duration_s: Optional[float]) -> float:
    """`CLIP_OFFSET_S`, or 0 for a video too short to seek that far into."""
    try:
        dur = float(duration_s or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return CLIP_OFFSET_S if dur >= CLIP_OFFSET_S + CLIP_DURATION_S else 0.0


def _wav_b64(src: str | Path, *, offset_s: float, duration_s: float) -> Optional[str]:
    """Transcode a window of `src` to mono 16 kHz WAV, base64-encoded.

    Returns None on any failure: a too-short file, a seek past the end that
    the retry at 0 cannot save, a missing codec.
    """
    from backend.media_io import _ffmpeg_bin

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            _ffmpeg_bin(), "-hide_banner", "-loglevel", "error",
            "-ss", str(offset_s),
            "-i", str(src),
            "-t", str(duration_s),
            "-ac", "1", "-ar", "16000",
            "-y", tmp_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0:
            cmd[cmd.index("-ss") + 1] = "0"      # offset was past the end
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode != 0:
                return None

        data = Path(tmp_path).read_bytes()
        if len(data) < 1000:
            return None
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        log.warning("audio clip extraction failed: %s", e)
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_audio_clip(
    video_path: str | Path,
    *,
    offset_s: float = CLIP_OFFSET_S,
    duration_s: float = CLIP_DURATION_S,
) -> Optional[str]:
    """A short base64 WAV clip from a video already on disk, or None."""
    video_path = Path(video_path)
    if not video_path.is_file():
        return None
    return _wav_b64(video_path, offset_s=offset_s, duration_s=duration_s)


def fetch_remote_audio_clip(
    url: str,
    *,
    offset_s: float = CLIP_OFFSET_S,
    duration_s: float = CLIP_DURATION_S,
) -> Optional[str]:
    """Pull a few seconds of audio off a URL without downloading the video.

    `download_ranges` fetches only the requested window of the smallest
    audio-only format — a couple of hundred kilobytes. yt-dlp cuts the range
    with ffmpeg, so it is pointed at the binary `backend.media_io` resolves;
    the bare `ffmpeg` it would otherwise call is often not on PATH here.

    Best-effort: a 429, a private video or a missing yt-dlp all return None.
    """
    try:
        import yt_dlp
    except ImportError:                                      # pragma: no cover
        return None

    from backend.media_io import _ffmpeg_bin

    try:
        ffmpeg_dir = str(Path(_ffmpeg_bin()).parent)
    except RuntimeError:
        return None

    end = offset_s + duration_s
    with tempfile.TemporaryDirectory(prefix="dubbing-langclip-") as tmpdir:
        opts = {
            "format": "worstaudio/bestaudio/best",
            "outtmpl": str(Path(tmpdir) / "clip.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": _SOCKET_TIMEOUT,
            "ffmpeg_location": ffmpeg_dir,
            "download_ranges": yt_dlp.utils.download_range_func(
                None, [(offset_s, end)]),
            "force_keyframes_at_cuts": False,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            log.warning("language-detection clip fetch failed (%s): %s",
                        url, e)
            return None

        got = sorted(p for p in Path(tmpdir).iterdir() if p.is_file())
        if not got:
            return None
        # yt-dlp already cut the window, so take the file whole.
        return _wav_b64(got[0], offset_s=0.0, duration_s=duration_s)


def detect_language_llm(audio_b64: str) -> Optional[Dict[str, str]]:
    """Ask the model what language the clip is in.

    Returns `{"language": "Hindi", "code": "hi"}`, or None on any failure
    (missing key, network, bad parse).
    """
    from backend.llm import LLMClient

    try:
        client = LLMClient(model=DETECT_MODEL, max_tokens=64)
    except Exception:
        return None
    if not client.available():
        return None

    messages = [{"role": "user", "content": [
        {"type": "input_audio",
         "input_audio": {"data": audio_b64, "format": "wav"}},
        {"type": "text", "text": PROMPT},
    ]}]

    try:
        raw = client.chat(messages, temperature=0.0, max_tokens=64,
                          trace_name="detect_language")
    except Exception as e:
        log.warning("language detection LLM call failed: %s", e)
        return None

    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        parsed = json.loads(text)
        code = str(parsed.get("code", "")).strip().lower()
        lang = str(parsed.get("language", "")).strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        log.warning("language detection: could not parse LLM response: %s",
                    text[:200])
        return None
    if code and lang and len(code) <= 3:
        return {"language": lang, "code": code}
    return None


def detect_from_probe(
    meta: Dict[str, Any],
    *,
    video_path: Optional[str | Path] = None,
    url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The source language of a video: uploader metadata, else its audio.

    `video_path` is the local file when there is one. `url` is used only when
    there is not and metadata came up empty, costing a short ranged fetch.
    Returns `{"code", "language", "method"}` or None, where `method` is
    `yt_metadata` or `llm_audio`.
    """
    meta = meta or {}

    for key in ("language", "original_language"):
        code = meta.get(key)
        if isinstance(code, str) and code.strip():
            return {"code": code.strip().lower()[:2],
                    "language": "", "method": "yt_metadata"}

    offset = _clip_offset(meta.get("duration_s"))
    audio_b64 = None
    if video_path:
        audio_b64 = extract_audio_clip(video_path, offset_s=offset)
    elif url:
        audio_b64 = fetch_remote_audio_clip(url, offset_s=offset)
    if not audio_b64:
        return None

    result = detect_language_llm(audio_b64)
    if result:
        return {**result, "method": "llm_audio"}
    return None
