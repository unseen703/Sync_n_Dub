"""Cue list for the review screen.

The operator reviews a dub by reading along with it, so this serves one cue
per segment with everything needed to decide "is this line right?" — when it
is spoken, who the pipeline thinks says it, what was heard, what was written
in the target language, and which clip to play.

It is a READ of the merged ASR+MT JSON, not a second source of truth. Every
field comes straight out of the segment the pipeline wrote; nothing here
re-derives, re-translates or re-times anything, so a cue cannot disagree
with the dub it describes.

WHY `speaker` IS WORTH SHOWING EVEN THOUGH IT LOOKS LIKE PLUMBING
-----------------------------------------------------------------
`backend/tts.py` resolves every segment's voice through `seg["speaker"]`, so
a mis-attributed cue is read in the wrong person's voice while every artifact
still looks correct. Diarization is where that goes wrong and stage 4c only
improves the odds. Showing the label per cue is what lets an operator catch
it, and `set_speaker` is what lets them fix it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

#: Cues above this are paged rather than sent at once. A 40-minute lecture is
#: a few thousand cues; shipping them all makes the review screen the
#: slowest page in the UI for no benefit.
DEFAULT_LIMIT = 500


def _read(p: Optional[Path]) -> Dict[str, Any]:
    if not p:
        return {}
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def _url(p: Optional[str], data_dir: Path) -> Optional[str]:
    if not p:
        return None
    try:
        rel = Path(p).resolve().relative_to(Path(data_dir).resolve())
    except (ValueError, OSError):
        return None
    return "/files/" + rel.as_posix()


def cues(merged_json: Optional[Path], *, data_dir: Path,
         offset: int = 0, limit: int = DEFAULT_LIMIT,
         labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One row per segment, in time order, plus what the whole list needs.

    `labels` renames speakers for DISPLAY only — it is the operator's
    `speaker_labels.json`, kept out of the merged JSON deliberately so that
    renaming a caption never dirties a pipeline stage.
    """
    data = _read(merged_json)
    segs = data.get("segments") or []
    labels = labels or {}

    rows: List[Dict[str, Any]] = []
    for seg in segs[offset:offset + max(0, limit)]:
        spk = seg.get("speaker")
        spk = str(spk) if spk is not None else None
        qc = seg.get("qc") or {}
        rows.append({
            "id": str(seg.get("id")),
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "speaker": spk,
            "speaker_label": labels.get(spk or "", spk),
            # What was heard, and what will be spoken. Both, because the
            # operator is checking a translation as much as a transcription
            # and one without the other cannot be judged.
            "source_text": seg.get("text") or "",
            "dub_text": (seg.get("final_text") or seg.get("mt_text")
                         or seg.get("text") or ""),
            "edited": bool(seg.get("edit_backup")),
            # The clip that actually plays for this line, so the operator
            # hears what shipped rather than a re-synthesis of it.
            "dub_audio_url": _url(seg.get("tts_audio"), data_dir),
            # Evidence the operator can act on. `tts_error` means this line
            # has NO audio at all -- a Bodhan outage leaves the stage green
            # with every segment failed, so it has to be visible per cue.
            "tts_error": qc.get("tts_error"),
            "budget_violation": bool(qc.get("budget_violation")),
            # Which face ASD attributed this line to, when stage 6 ran. It is
            # how an operator checks the active speaker against the picture.
            "face_track": seg.get("face_track"),
            "on_screen": seg.get("on_screen"),
        })

    speakers = sorted({r["speaker"] for r in rows if r["speaker"]})
    return {
        "total": len(segs),
        "offset": offset,
        "limit": limit,
        "cues": rows,
        "speakers": speakers,
        "source_language": data.get("language"),
        "target_language": (data.get("extra") or {}).get("mt_target_language"),
        # Counts the operator would otherwise have to tally by scrolling.
        "n_edited": sum(1 for r in rows if r["edited"]),
        "n_failed": sum(1 for r in rows if r["tts_error"]),
    }
