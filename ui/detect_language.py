"""Detect the spoken language of a video source.

Two strategies, in order:

  1. **yt-dlp metadata** — ``info["language"]`` or ``info["original_language"]``
     from the YouTube page.  Free, instant, and right when present.
  2. **LLM with audio** — extract a short clip, base64-encode it, and ask
     Gemini 3.1 Flash Lite via OpenRouter.  Used when (1) is absent, and
     always for uploaded files (no yt-dlp metadata at all).

The LLM call is best-effort: if the key is unset or the call fails, the
caller falls back to ``auto`` and the ASR adapter detects per-turn.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("ui.detect_language")

DETECT_MODEL = "google/gemini-3.1-flash-lite"

PROMPT = (
    "Identify the spoken language. "
    "Return only its language name and ISO 639-1 code.\n"
    "Output format: {\"language\": \"<name>\", \"code\": \"<iso-639-1>\"}"
)


def _ffmpeg_bin() -> str:
    """Resolve ffmpeg the same way the rest of the codebase does."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    interp = Path(os.sys.executable)
    candidate = interp.parent / "ffmpeg"
    if candidate.is_file():
        return str(candidate)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found")


def extract_audio_clip(
    video_path: str | Path,
    *,
    offset_s: float = 30.0,
    duration_s: float = 8.0,
) -> Optional[str]:
    """Extract a short mono WAV clip and return it as a base64 string.

    Seeks to ``offset_s`` (falling back to 0 for short videos) and grabs
    ``duration_s`` seconds.  Returns None on any failure.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        return None

    ffmpeg = _ffmpeg_bin()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", str(offset_s),
            "-i", str(video_path),
            "-t", str(duration_s),
            "-ac", "1", "-ar", "16000",
            "-y", tmp_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0:
            # offset was past the end — try from the start
            cmd[3] = "0"
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


def detect_language_llm(
    *,
    description: str = "",
    audio_b64: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Ask Gemini Flash Lite to identify the spoken language.

    Returns ``{"language": "Hindi", "code": "hi"}`` on success, None on
    any failure (missing key, network, bad parse).
    """
    from backend.llm import LLMClient, LLMUnavailable

    try:
        client = LLMClient(model=DETECT_MODEL, max_tokens=64)
    except Exception:
        return None

    if not client.available():
        return None

    content_parts: list[Dict[str, Any]] = []
    if audio_b64:
        content_parts.append({
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"},
        })
    if description:
        content_parts.append({
            "type": "text",
            "text": f"Video description: {description[:500]}",
        })
    content_parts.append({"type": "text", "text": PROMPT})

    messages = [{"role": "user", "content": content_parts}]

    try:
        raw = client.chat(messages, temperature=0.0, max_tokens=64,
                          trace_name="detect_language")
    except (LLMUnavailable, Exception) as e:
        log.warning("language detection LLM call failed: %s", e)
        return None

    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        code = str(parsed.get("code", "")).strip().lower()
        lang = str(parsed.get("language", "")).strip()
        if code and lang and len(code) <= 3:
            return {"language": lang, "code": code}
    except (json.JSONDecodeError, KeyError, TypeError):
        log.warning("language detection: could not parse LLM response: %s",
                    raw[:200])
    return None


def detect_from_yt_meta(meta: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Try to get the language from yt-dlp metadata fields."""
    for key in ("language", "original_language"):
        code = (meta.get(key) or "")
        if isinstance(code, str) and code.strip():
            code = code.strip().lower()[:2]
            return {"code": code, "method": "yt_metadata"}
    return None


def detect_from_probe(
    meta: Dict[str, Any],
    *,
    video_path: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Detect source language: yt-dlp metadata first, LLM fallback.

    Returns ``{"code": "hi", "language": "Hindi", "method": "..."}`` or None.
    """
    # Strategy 1: yt-dlp metadata
    yt = detect_from_yt_meta(meta)
    if yt:
        return {"code": yt["code"], "language": "", "method": "yt_metadata"}

    # Strategy 2: LLM with audio clip + description
    description = " ".join(filter(None, [
        str(meta.get("title") or ""),
        str(meta.get("description") or ""),
    ]))[:500]

    audio_b64 = None
    if video_path:
        audio_b64 = extract_audio_clip(video_path)

    result = detect_language_llm(description=description,
                                 audio_b64=audio_b64)
    if result:
        return {**result, "method": "llm"}
    return None
