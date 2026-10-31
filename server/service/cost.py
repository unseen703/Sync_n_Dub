"""Turn an edit's dirty stages into a cost preview the user can consent to.

Produces `stages` (what re-runs, labelled, GPU-flagged, with an estimate where
a rate is known), `unaffected_labels` (what does not) and `total_est_s`.
Labels come from `server.progress.STAGE_LABELS` via `_PROGRESS_KEY`, which joins
the two stage vocabularies. A stage with no measured rate gets `None`, not a
guess.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import lipsync_models as _ls_models
from ..progress import STAGE_LABELS as _PROGRESS_LABELS

#: `backend.edits.STAGE_ORDER` name -> the `server.progress` marker key with the
#: same meaning. This is the join between the two vocabularies.
_PROGRESS_KEY: Dict[str, str] = {
    "codemix":  "9b",
    "duration": "9c",
    "voices":   "9d",
    "tts":      "9e",
    "timing":   "9f2",
    "mix":      "9g",
    "lipsync":  "9h",
}

#: Stages that hold a GPU for minutes. Only lip-sync: every other post-MT
#: stage is a network call or ffmpeg.
_GPU_STAGES = frozenset({"lipsync"})

#: Upstream of the merged JSON. `run_post_mt_stages` (from 9d onward in a
#: cast run, or the full chain in an edit replay) never touches these, so
#: they are what the edit cannot put at risk.
_UPSTREAM_KEYS: Sequence[str] = ("1", "2", "3", "4", "5", "6", "7", "8", "9")

#: Lip-sync wall time as a multiple of clip length: midpoints of the bands in
#: `server/service/lipsync_models.py`. Derived rather than written out again, so a row
#: cannot go missing; a model with no band is absent here and reads as
#: "not measured".
LIPSYNC_RATES: Dict[str, float] = {
    k: (v["rate"][0] + v["rate"][1]) / 2.0
    for k, v in _ls_models.FACTS.items() if v.get("rate")
}

#: Speech generation wall time as a multiple of clip length. Bodhan is
#: hosted, so one request per cue plus upload/download dominates; measured
#: at ~3× the clip's duration end-to-end.
TTS_RATE = 3.0

#: Seconds of TTS per segment, kept as a floor for very short clips where
#: 3× clip time underestimates per-cue request overhead.
TTS_S_PER_SEGMENT = 0.6

#: The mix is one ffmpeg pass — fast, but not free on a long video.
MIX_RATE = 0.05
MIX_FLOOR_S = 5.0

#: Voice casting is catalogue work plus, at worst, one embedding per speaker.
VOICES_FLAT_S = 5.0


def label_for(stage: str) -> str:
    """User-facing text for a `backend.edits.STAGE_ORDER` name."""
    key = _PROGRESS_KEY.get(stage)
    if key and key in _PROGRESS_LABELS:
        return _PROGRESS_LABELS[key]
    return stage.replace("_", " ").capitalize()


def _lipsync_rate(model: Optional[str]) -> Optional[float]:
    # Through `normalise`, so a recorded variant resolves instead of falling
    # through to "no measured rate".
    key = _ls_models.normalise(model)
    return LIPSYNC_RATES.get(key) if key else None


def _estimate(stage: str, *, clip_s: Optional[float],
              n_segments: Optional[int],
              config: Dict[str, Any]) -> Optional[float]:
    """Seconds, or None when this host has no measured rate for the stage."""
    if stage == "tts":
        if clip_s is None:
            if not n_segments:
                return None
            return round(n_segments * TTS_S_PER_SEGMENT, 1)
        floor = (n_segments or 0) * TTS_S_PER_SEGMENT
        return round(max(clip_s * TTS_RATE, floor), 1)
    if stage == "voices":
        return VOICES_FLAT_S
    if stage == "mix":
        if clip_s is None:
            return None
        return round(max(MIX_FLOOR_S, clip_s * MIX_RATE), 1)
    if stage == "lipsync":
        rate = _lipsync_rate(config.get("lipsync") or config.get("lipsync_model"))
        if clip_s is None or rate is None:
            return None
        total = clip_s * rate
        return round(total, 1)
    # codemix, duration and timing have no measured rate on this host. Say so
    # by omission.
    return None


def estimate(dirty_stages: Sequence[str], *,
             config: Optional[Dict[str, Any]] = None,
             clip_s: Optional[float] = None,
             n_segments: Optional[int] = None,
             available: Optional[Iterable[str]] = None,
             warnings: Iterable[str] = ()) -> Dict[str, Any]:
    """Cost preview for a set of dirty stages.

    `available` is the post-MT stages this job can replay
    (`backend.edit_agent.available_stages`). Anything in it that is not dirty
    is listed as unaffected too, so the user sees it named rather than absent.
    """
    config = dict(config or {})
    dirty = [s for s in dirty_stages]
    warns: List[str] = list(warnings)

    rows: List[Dict[str, Any]] = []
    for s in dirty:
        est = _estimate(s, clip_s=clip_s, n_segments=n_segments, config=config)
        rows.append({
            "id": s,
            "label": label_for(s),
            "gpu": s in _GPU_STAGES,
            "est_s": est,
        })

    if "lipsync" in dirty and _lipsync_rate(
            config.get("lipsync") or config.get("lipsync_model")) is None:
        warns.append(
            "no measured rate for lip-sync model "
            f"{config.get('lipsync') or config.get('lipsync_model') or 'unset'!r} "
            "on this host — the lip-sync line has no estimate")

    known = [r["est_s"] for r in rows if r["est_s"] is not None]
    total = round(sum(known), 1) if known else None

    unaffected = [_PROGRESS_LABELS[k] for k in _UPSTREAM_KEYS
                  if k in _PROGRESS_LABELS]
    # De-duplicate while keeping order: "1" and "fetch" share a label.
    seen = set()
    unaffected = [x for x in unaffected if not (x in seen or seen.add(x))]
    for s in (available or ()):
        if s not in dirty:
            lbl = label_for(s)
            if lbl not in seen:
                seen.add(lbl)
                unaffected.append(lbl)

    return {
        "stages": rows,
        "unaffected_labels": unaffected,
        "total_est_s": total,
        "gpu_required": any(r["gpu"] for r in rows),
        "warnings": warns,
    }
