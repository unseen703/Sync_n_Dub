"""What a staged edit will cost to re-render, in terms a user can consent to.

`backend.edits` answers "which stages are dirty" in pipeline vocabulary —
`["voices", "tts", "mix", "lipsync"]`. That is the right answer and the wrong
words: nobody approves "voices", and nobody can tell from that list whether
the hour of diarization, ASR and translation that produced the transcript is
about to be thrown away. It is not — `run_pipeline(post_mt_only_from=…)`
replays stages 9b..9h and nothing above them — but the list does not say so.

So this module turns `EditResult.dirty_stages` into three things:

  `stages`             what re-runs, labelled in the UI's own words, with a
                       `gpu` flag on the lines that are minutes rather than
                       seconds and an `est_s` where a rate has been measured;
  `unaffected_labels`  what does NOT re-run — the part the user is actually
                       afraid of losing, and the reason the dialog is safe to
                       confirm;
  `total_est_s`        the sum of the estimates that exist.

The labels come from `ui.progress.STAGE_LABELS` wherever the keys line up.
They do not line up directly — `backend.edits.STAGE_ORDER` names the stages
`codemix, duration, voices, tts, vc, timing, mix, lipsync` while the progress
table keys them by the pipeline's own marker ids (`9b`, `9c`, `9d`, …) — so
`_PROGRESS_KEY` maps one onto the other and the label is read from the single
table rather than written out a second time here. One table, two spellings.

Where the numbers come from
---------------------------
`.claude/skills/clip-run`, measured on this host's 2x16 GiB Blackwell cards,
as multiples of clip length: wav2lip 3–5x, musetalk 8–21x, latentsync ~83x,
wav2lip_hd 6–24x (Wav2Lip plus a per-frame restoration pass on top, so its
cost tracks frame SIZE as well as length). TTS is roughly 0.6 s per segment.
The midpoint of each range is used, and an estimate is omitted (`None`)
rather than invented for a stage with no measured rate — a made-up number in
a consent dialog is worse than a blank, because the user will plan around it.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import lipsync_models as _ls_models
from .progress import STAGE_LABELS as _PROGRESS_LABELS

#: `backend.edits.STAGE_ORDER` name -> the `ui.progress` marker key with the
#: same meaning. This is the join between the two vocabularies.
_PROGRESS_KEY: Dict[str, str] = {
    "codemix":  "9b",
    "duration": "9c",
    "voices":   "9d",
    "tts":      "9e",
    "vc":       "9f",
    "timing":   "9f2",
    "mix":      "9g",
    "lipsync":  "9h",
}

#: Stages that hold a GPU for minutes. Lip-sync is the headline; voice
#: conversion also runs a model on the card, and a dialog that marked only
#: one of them would understate a VC job.
_GPU_STAGES = frozenset({"lipsync", "vc"})

#: Upstream of the merged JSON. `run_post_mt_stages` never touches these, so
#: they are listed verbatim as the things the edit cannot put at risk.
_UPSTREAM_KEYS: Sequence[str] = ("1", "2", "3", "4", "5", "6", "7", "8", "9")

#: Lip-sync wall time as a multiple of clip length, measured 2026-10-12.
#: Midpoints of the bands in `ui/lipsync_models.py`.
#:
#: Derived, not written out again. `wav2lip_hd` was missing here — the line
#: below this table was an orphan comment with no entry after it, which is
#: what a dropped row looks like. A restoration pass is a *second* pass over
#: the render, not a cheaper one, and it now carries its own measured band.
LIPSYNC_RATES: Dict[str, float] = {
    k: (v["rate"][0] + v["rate"][1]) / 2.0
    for k, v in _ls_models.FACTS.items() if v.get("rate")
}

#: Seconds of TTS per segment, measured over the Edge and Bodhan paths.
TTS_S_PER_SEGMENT = 0.6

#: The mix is one ffmpeg pass: fast, but not free on a long video.
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
    # Through `normalise`, so a recorded variant (`musetalk-v15`,
    # `wav2lip-hd`) resolves instead of falling through to "no measured rate".
    key = _ls_models.normalise(model)
    return LIPSYNC_RATES.get(key) if key else None


def _estimate(stage: str, *, clip_s: Optional[float],
              n_segments: Optional[int],
              config: Dict[str, Any]) -> Optional[float]:
    """Seconds, or None when this host has no measured rate for the stage."""
    if stage == "tts":
        if not n_segments:
            return None
        return round(n_segments * TTS_S_PER_SEGMENT, 1)
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
    # codemix (one network call per cue), duration, timing, vc: no rate has
    # been measured on this host. Say so by omission.
    return None


def estimate(dirty_stages: Sequence[str], *,
             config: Optional[Dict[str, Any]] = None,
             clip_s: Optional[float] = None,
             n_segments: Optional[int] = None,
             available: Optional[Iterable[str]] = None,
             warnings: Iterable[str] = ()) -> Dict[str, Any]:
    """Cost preview for a set of dirty stages.

    `available` is `backend.edit_agent.available_stages(...)` — the post-MT
    stages this job can actually replay. Anything in it that is not dirty is
    named as unaffected too, so "voice conversion is NOT re-run" is visible
    on a job that has voice conversion.
    """
    config = dict(config or {})
    dirty = [s for s in dirty_stages]
    warns: List[str] = list(warnings)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for s in dirty:
        est = _estimate(s, clip_s=clip_s, n_segments=n_segments, config=config)
        if est is None:
            missing.append(label_for(s))
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
    if total is not None and missing:
        warns.append("estimate excludes " + ", ".join(missing)
                     + " (no measured rate on this host)")

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
