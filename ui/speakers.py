"""Who speaks, in which voice — reconstructed from whatever the run left.

The problem this solves
-----------------------
Stage 9d writes `voices/<stem>.<engine>.voices.json` **only inside
`if voice_engine:`**. A cloning run (IndicF5, VC) takes the `else` branch,
builds `speaker_voice_map` in memory, and persists nothing. Older jobs on
disk predate the catalog default and have no `*.voices.json` at all, so a UI
that only read that file would show an empty speaker table on most of them.

Four sources are tried in order of how directly they record the decision:

  1. `voices/*.voices.json`                   — 9d's `if` branch
  2. merged JSON `extra.speaker_voice_map`    — added by the pipeline patch
  3. per-segment `tts_voice`                  — whenever `backend.edits` ran
  4. deterministic replay of 9d's `else`      — always available post-9d

Step 4 is exact rather than a guess. Since Edge was removed the `else`
branch assigns no voice at all -- it is the cloning path, where the speaker's
own reference clip is the voice -- so the replay reports every speaker in
that branch with `voice: None`, which is what actually happened. The speaker
ids come from the stems of the files in `voices/<stem>.refs/`, which are
precisely the keys 9d enumerated.

Before 9d has run at all the table still appears — ids and speech time come
from the diarization RTTM — so the user sees their cast as soon as
diarization finishes, with voices filling in later.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("ui.speakers")

#: Engines whose adapter forwards `style` to the API. This is the exact
#: predicate `backend.edits._h_set_tone` warns on, kept identical so the UI
#: hides the tone control precisely when the engine would ignore it.
TONE_ENGINES = ("bodhan", "indic-speak")

#: Stable colours for speaker chips, assigned by rank so the same speaker
#: keeps its colour across reloads.
SPEAKER_COLORS = ("#6366f1", "#06b6d4", "#f59e0b", "#ec4899",
                  "#10b981", "#8b5cf6", "#ef4444", "#14b8a6")


LABELS_FILE = "speaker_labels.json"


def read_labels(job_root: Path) -> Dict[str, str]:
    """User-given names for speakers, e.g. SPEAKER_01 -> "Prof. Mitra".

    A job-local file rather than the merged JSON: renaming a speaker is a
    display preference and must not dirty a pipeline stage. Nothing
    downstream reads it, so a rename never triggers a re-render.
    """
    doc = _read_json(Path(job_root) / LABELS_FILE)
    return {str(k): str(v) for k, v in doc.items()} if isinstance(doc, dict) else {}


def write_label(job_root: Path, speaker_id: str, label: Optional[str]) -> Dict[str, str]:
    """Set or clear one speaker's name. Returns the full map."""
    job_root = Path(job_root)
    labels = read_labels(job_root)
    clean = (label or "").strip()
    if clean:
        labels[str(speaker_id)] = clean[:80]
    else:
        labels.pop(str(speaker_id), None)
    tmp = job_root / (LABELS_FILE + ".tmp")
    tmp.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    tmp.replace(job_root / LABELS_FILE)
    return labels


def supports_tone(config: Dict[str, Any]) -> bool:
    eng = (config.get("tts") or config.get("voice_engine") or "").lower()
    return eng in TONE_ENGINES


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _from_voices_json(job_root: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    vdir = job_root / "voices"
    if not vdir.is_dir():
        return out
    for p in sorted(vdir.glob("*.voices.json")):
        rows = _read_json(p)
        if not isinstance(rows, list):
            continue
        for r in rows:
            sid = r.get("speaker")
            if sid and r.get("voice"):
                out[str(sid)] = {"engine": r.get("engine"), "voice": r["voice"],
                                 "score": r.get("score"),
                                 "rationale": r.get("rationale"),
                                 "source": "assigned"}
    return out


def speaker_genders(job_root: Path, merged: Dict[str, Any]) -> Dict[str, str]:
    """Predicted gender per speaker, from whatever recorded it.

    Stage 9d computes this from the reference clips with ECAPA and now
    writes it beside the voice. Re-deriving it here would mean loading a
    speaker-embedding model inside a web request, which is not a thing a
    GET should do.
    """
    out: Dict[str, str] = {}
    for info in (_read_json(p) or {} for p in
                 (job_root / "voices").glob("*.speaker_voice_map.json")):
        if isinstance(info, dict):
            for sid, rec in info.items():
                g = (rec or {}).get("gender") if isinstance(rec, dict) else None
                if g:
                    out.setdefault(str(sid), str(g))
    m = (merged.get("extra") or {}).get("speaker_voice_map")
    if isinstance(m, dict):
        for sid, rec in m.items():
            g = (rec or {}).get("gender") if isinstance(rec, dict) else None
            if g:
                out.setdefault(str(sid), str(g))
    # An explicit gender edit wins over the prediction.
    for seg in merged.get("segments") or []:
        g, spk = seg.get("edit_gender"), seg.get("speaker")
        if g and spk:
            out[str(spk)] = str(g)
    return out


def _from_speaker_voice_map(job_root: Path, merged: Dict[str, Any]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    candidates: List[Any] = [(merged.get("extra") or {}).get("speaker_voice_map")]
    for p in (job_root / "voices").glob("*.speaker_voice_map.json"):
        candidates.append(_read_json(p))
    for m in candidates:
        if not isinstance(m, dict):
            continue
        for sid, info in m.items():
            if isinstance(info, dict) and info.get("voice"):
                out.setdefault(str(sid), {"engine": info.get("engine"),
                                          "voice": info["voice"],
                                          "score": None, "rationale": None,
                                          "source": "assigned"})
    return out


def _from_segments(merged: Dict[str, Any]) -> Dict[str, dict]:
    """Majority `tts_voice` per speaker — present whenever edits ran."""
    votes: Dict[str, Counter] = defaultdict(Counter)
    for seg in merged.get("segments") or []:
        v, spk = seg.get("tts_voice"), seg.get("speaker")
        if v and spk:
            votes[str(spk)][v] += 1
    return {sid: {"engine": None, "voice": c.most_common(1)[0][0],
                  "score": None, "rationale": None, "source": "edit"}
            for sid, c in votes.items() if c}


def _replay_else_branch(job_root: Path, config: Dict[str, Any]) -> Dict[str, dict]:
    """Reproduce 9d's `else` branch exactly. See the module docstring."""
    refs = _reference_clips(job_root)
    if not refs:
        return {}
    tts = (config.get("tts") or "").lower()
    # Since Edge was removed, that branch assigns no voice at all: it is
    # reached only by the cloning engines (IndicF5, VC), where the reference
    # clip IS the voice and there is no name to report. Say so rather than
    # inventing one.
    return {sid: {"engine": tts or None, "voice": None, "score": None,
                  "rationale": None, "source": "derived"}
            for sid in refs}


def _reference_clips(job_root: Path) -> Dict[str, Path]:
    """`<job>/voices/<stem>.refs/<SPEAKER>.wav`, keyed by speaker id."""
    out: Dict[str, Path] = {}
    vdir = job_root / "voices"
    if not vdir.is_dir():
        return out
    for refs in sorted(vdir.glob("*.refs")):
        if refs.is_dir():
            for w in sorted(refs.glob("*.wav")):
                out[w.stem] = w
    return out


_RTTM_RE = re.compile(r"^SPEAKER\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+"
                      r"\S+\s+\S+\s+(\S+)")


def _from_rttm(job_root: Path) -> Dict[str, float]:
    """Speaker -> total speech seconds, straight from the diarization RTTM."""
    totals: Dict[str, float] = defaultdict(float)
    d = job_root / "diarization"
    if not d.is_dir():
        return {}
    for p in sorted(d.glob("*.rttm")):
        try:
            text = p.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            m = _RTTM_RE.match(line.strip())
            if m:
                totals[m.group(3)] += float(m.group(2))
        if totals:
            break
    return dict(totals)


def _from_merged_segments(merged: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Speech time and line count per speaker, from the merged transcript."""
    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"speech_s": 0.0, "n_segments": 0})
    for seg in merged.get("segments") or []:
        spk = seg.get("speaker")
        if not spk:
            continue
        a = agg[str(spk)]
        a["speech_s"] += max(0.0, float(seg.get("end", 0) or 0)
                             - float(seg.get("start", 0) or 0))
        a["n_segments"] += 1
    return dict(agg)


# ---------------------------------------------------------------------------
# voice catalogue for the picker
# ---------------------------------------------------------------------------

def voice_options(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    eng = (config.get("tts") or config.get("voice_engine") or "").lower()
    lang = (config.get("target_language") or "").split("-")[0].lower()
    try:
        from backend.voices import load_voice_catalog
    except Exception:                                # pragma: no cover
        return []
    cat_engine = "indic-speak" if eng in ("", "bodhan", "indic-speak") else eng
    rows = []
    for v in load_voice_catalog(cat_engine, language=lang or None):
        rows.append({"voice": v.voice, "engine": v.engine, "gender": v.gender,
                     "age": v.age, "language": v.language, "tier": v.tier,
                     "display": v.voice})
    return rows


# ---------------------------------------------------------------------------
# the endpoint payload
# ---------------------------------------------------------------------------

def resolve_speaker_voices(job_root: Path, config: Dict[str, Any],
                           merged: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, dict]:
    """Best available voice assignment per speaker, with its provenance."""
    merged = merged or {}
    out: Dict[str, dict] = {}
    for step in (_from_voices_json(job_root),
                 _from_speaker_voice_map(job_root, merged),
                 _from_segments(merged),
                 _replay_else_branch(job_root, config)):
        for sid, info in step.items():
            out.setdefault(sid, info)
    return out


def build_table(job_root: Path, config: Dict[str, Any], *,
                data_dir: Path,
                merged_json: Optional[Path] = None) -> Dict[str, Any]:
    job_root = Path(job_root)
    merged = (_read_json(merged_json) or {}) if merged_json else {}
    voices = resolve_speaker_voices(job_root, config, merged)
    genders = speaker_genders(job_root, merged)
    labels = read_labels(job_root)
    refs = _reference_clips(job_root)
    stats = _from_merged_segments(merged)
    rttm = _from_rttm(job_root)
    tones = _segment_tones(merged)

    ids = sorted(set(voices) | set(refs) | set(stats) | set(rttm))
    rows = []
    for sid in ids:
        st = stats.get(sid, {})
        speech = st.get("speech_s") or rttm.get(sid) or 0.0
        v = voices.get(sid) or {}
        ref = refs.get(sid)
        rows.append({
            "id": sid,
            "label": labels.get(sid),
            "speech_s": round(float(speech), 2),
            "n_segments": int(st.get("n_segments") or 0),
            "gender": genders.get(sid),
            "voice": v.get("voice"),
            "voice_source": v.get("source"),
            "voice_score": v.get("score"),
            "voice_rationale": v.get("rationale"),
            "tone": tones.get(sid),
            "reference_url": (_files_url(ref, data_dir) if ref else None),
        })
    # Protagonist first, the way `assign_voices` already ranks speakers.
    rows.sort(key=lambda r: -r["speech_s"])
    for i, r in enumerate(rows):
        r["color"] = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]

    eng = (config.get("tts") or config.get("voice_engine") or "")
    opts = voice_options(config)
    # Tag each voice with which speakers it suits, so the picker can open on
    # the right gender instead of making the user scan 44 names. Not a
    # filter: a user who wants a male voice for a speaker ECAPA called
    # female must still be able to pick one, because the prediction is a
    # prediction.
    for o in opts:
        o["recommended_for"] = [r["id"] for r in rows
                                if r["gender"] and o.get("gender") == r["gender"]]
    return {
        "engine": eng or None,
        "engine_label": _engine_label(eng),
        "language": config.get("target_language"),
        "supports_tone": supports_tone(config),
        "tone_options": (tone_vocabulary() if supports_tone(config) else []),
        "voice_options": opts,
        "speakers": rows,
    }


def tone_vocabulary() -> List[Dict[str, str]]:
    """The delivery styles the engine actually accepts.

    Read from `models.tts.bodhan_adapter.INDIC_SPEAK_STYLES` rather than
    written out here. Case and spacing are significant and an unknown value
    is a **422, not a silent fallback** — an invented list (which this module
    shipped at first: "formal", "casual", "polite") would have made every
    tone change fail at the API with nothing in the UI to explain it.

    Returned as {value, label} so the wire value stays verbatim while the
    dropdown can show "AIR style news" as "AIR-style news".
    """
    try:
        from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES
    except Exception as e:                               # pragma: no cover
        log.warning("bodhan style list unavailable: %s", e)
        return []
    return [{"value": v, "label": _tone_label(v),
             "group": _tone_group(v)} for v in INDIC_SPEAK_STYLES]


#: The six Ekman emotions in the list come from AI4Bharat's Rasa corpus; the
#: rest are delivery registers. Grouping them keeps a 14-item dropdown
#: scannable, and tells the user which ones are moods rather than formats.
_EMOTIONS = ("anger", "disgust", "fear", "happy", "sad", "surprise")


def _tone_group(style: str) -> str:
    return "Emotion" if style in _EMOTIONS else "Delivery"


def _tone_label(style: str) -> str:
    return {
        "AIR style news": "AIR-style news",
        "TV style news": "TV-style news",
        "single person narration audiobook": "Audiobook narration",
        "children's stories": "Children's stories",
        "educational lecture": "Educational lecture",
        "Customer Care": "Customer care",
    }.get(style, style[:1].upper() + style[1:])


def _segment_tones(merged: Dict[str, Any]) -> Dict[str, str]:
    votes: Dict[str, Counter] = defaultdict(Counter)
    for seg in merged.get("segments") or []:
        t, spk = seg.get("tts_style"), seg.get("speaker")
        if t and spk:
            votes[str(spk)][t] += 1
    return {sid: c.most_common(1)[0][0] for sid, c in votes.items() if c}


def _engine_label(engine: str) -> str:
    return {"bodhan": "Bodhan Indic voices",
            "indic-speak": "Bodhan Indic voices",
            "indicf5": "IndicF5 (cloned from the speaker)"}.get(
                (engine or "").lower(), engine or "—")


def _files_url(p: Path, data_dir: Path) -> str:
    try:
        rel = Path(p).resolve().relative_to(Path(data_dir).resolve())
    except ValueError:
        return ""
    return "/files/" + rel.as_posix()
