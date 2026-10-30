"""Render a job's config as label/value/hint lines a user can read.

Done server-side because mapping a config key to readable English is domain
knowledge that belongs next to the pipeline; duplicating it in TypeScript
guarantees drift. The frontend renders the triples without interpreting them.
The raw config comes back too, for the "Advanced" disclosure.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import lipsync_models as _ls_models
from .langs import describe as describe_lang

log = logging.getLogger("ui.spec")


def _line(key: str, label: str, value: Any, *, hint: str = "",
          tone: str = "default") -> Dict[str, Any]:
    return {"key": key, "label": label, "value": str(value),
            "hint": hint or None, "tone": tone}


def _clock(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _relative(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        t = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    delta = (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds()
    if delta < 90:
        return "just now"
    for unit, size in (("minute", 60), ("hour", 3600), ("day", 86400)):
        n = int(delta // size)
        if delta < size * (60 if unit == "minute" else 24 if unit == "hour" else 30):
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return t.strftime("%Y-%m-%d")


SCRIPT_WORDS = {
    "codemix": ("Target language, English terms kept in English",
                "Technical terms and acronyms stay in Latin script, the way "
                "they are actually spoken"),
    "native":  ("Target language script throughout", ""),
    "roman":   ("Romanised (Latin script)", ""),
    "latin":   ("Romanised (Latin script)", ""),
}

#: Removed engines stay listed: old jobs still carry them in their config and
#: would otherwise render the raw id.
TTS_WORDS = {
    
    "bodhan":      "Bodhan Indic voices",
    "indic-speak": "Bodhan Indic voices",
}

#: How stage 9b1 decided the register, in operator terms. The tier matters
#: because confidence differs sharply between them.
REGISTER_TIER_WORDS = {
    "categories":  "From the video's own YouTube category.",
    "channel":     "From a channel this install has a rule for.",
    "description": "Matched wording in the uploader's description.",
    "llm":         "Inferred from the transcript — the least certain of the "
                   "four signals, and worth a listen before trusting it.",
}

#: Derived from `ui/lipsync_models.py` rather than written out, so a model
#: cannot be missing or matched by a prefix of a longer name.
LIPSYNC_WORDS = {k: f"On — {v['label']}"
                 for k, v in _ls_models.FACTS.items()}


def build(job_root: Path, *, manifest: Dict[str, Any],
          meta: Optional[Dict[str, Any]] = None,
          progress: Optional[Dict[str, Any]] = None,
          n_speakers: Optional[int] = None) -> Dict[str, Any]:
    cfg = dict(manifest.get("config") or {})
    result = dict(manifest.get("result") or {})
    meta = meta or {}
    lines: List[Dict[str, Any]] = []

    dur = meta.get("duration_s")
    win = meta.get("window") or cfg.get("window_abs_s")
    if win and isinstance(win, (list, tuple)) and len(win) == 2:
        clip_len = max(0.0, float(win[1]) - float(win[0]))
        part = f"{_clock(win[0])} – {_clock(win[1])} ({_clock(clip_len)})"
    elif dur:
        part = _clock(dur)
    else:
        part = "—"
    lines.append(_line("part", "Clip dubbed", part))

    lang = cfg.get("target_language") or cfg.get("language")
    script = (cfg.get("target_script") or "").lower()
    lang_value = describe_lang(lang)
    script_hint = ""
    if script in SCRIPT_WORDS:
        _, script_hint = SCRIPT_WORDS[script]
        lang_value = f"{lang_value} ({script})"
    lines.append(_line("language", "Dubbing into", lang_value, hint=script_hint))

    if n_speakers is not None:
        lines.append(_line("speakers", "Speakers found",
                           n_speakers if n_speakers else "—"))

    tts = (cfg.get("tts") or cfg.get("voice_engine") or "").lower()
    if tts:
        lines.append(_line("voices", "Voices from",
                           TTS_WORDS.get(tts, tts),
                           hint=cfg.get("dub_provenance") or ""))
        # State the absence: the tone control is hidden on these engines, and
        # an unexplained missing feature reads as a bug.
        if tts not in ("bodhan", "indic-speak"):
            lines.append(_line(
                "tone", "Tone control",
                f"Not available with {TTS_WORDS.get(tts, tts)}",
                hint="Only the Bodhan voices accept a delivery style.",
                tone="warn"))
        else:
            lines.append(_line("tone", "Tone control", "Available"))

    # Stage 9b1 picks the register from the uploader's metadata, so it is a
    # decision the operator never made and could not otherwise see.
    register = result.get("content_register")
    if register:
        lines.append(_line("register", "Video type", register,
                           hint=REGISTER_TIER_WORDS.get(
                               result.get("content_register_tier") or "", "")))
    elif tts in ("bodhan", "indic-speak") and cfg.get("tts_style") == "auto":
        # Asked for, and nothing was confident enough to answer.
        lines.append(_line("register", "Video type", "Neutral",
                           hint="Nothing in the video's metadata identified "
                                "the programme type, so the voice uses its "
                                "default delivery."))

    ls = (cfg.get("lipsync") or "").lower()
    # `normalise` resolves the variant the clip runner records and matches the
    # longest key first.
    ls_word = LIPSYNC_WORDS.get(_ls_models.normalise(ls) or "")
    lines.append(_line("lipsync", "Lip-sync",
                       ls_word or "Off (dubbed audio only)"))
    if ls and cfg.get("lipsync_auto_reason"):
        lines.append(_line("lipsync_why", "Model chosen because",
                           cfg["lipsync_auto_reason"],
                           hint="Picked automatically from the face size; "
                                "you can override it on a new job."))


    started = manifest.get("started_at") or (progress or {}).get("started_at") \
        or (progress or {}).get("updated_at")
    lines.append(_line("started", "Started", _relative(started)))
    took = _took(manifest)
    if took:
        lines.append(_line("took", "Took", took))

    rev = int(manifest.get("revision") or 0)
    if rev > 1:
        lines.append(_line("revision", "Version",
                           f"v{rev} — re-rendered after an edit"))

    return {
        "title": meta.get("title"),
        "thumbnail_url": meta.get("thumbnail_url"),
        "lines": lines,
        "raw_config": cfg,
    }


def _took(manifest: Dict[str, Any]) -> Optional[str]:
    a, b = manifest.get("started_at"), manifest.get("finished_at")
    if not (a and b):
        return None
    try:
        t0 = _dt.datetime.fromisoformat(str(a).replace("Z", "+00:00"))
        t1 = _dt.datetime.fromisoformat(str(b).replace("Z", "+00:00"))
    except ValueError:
        return None
    s = int((t1 - t0).total_seconds())
    if s < 0:
        return None
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h} h {m:02d} m"
    return f"{m} min {sec:02d} s" if m else f"{sec} s"


def _short_url(url: str) -> str:
    u = str(url)
    if len(u) <= 46:
        return u
    return u[:28] + "…" + u[-14:]
