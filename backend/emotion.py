"""Stage 9b2 — per-segment delivery style from the source speaker's emotion.

Reads how each line was delivered in the ORIGINAL audio and writes a
`tts_style` onto the matching segment, which `backend/tts.py` already picks
up (`style=seg.get("tts_style") or info.get("style")`). The synthesized
Hindi line then carries the English speaker's delivery instead of a flat
read.

Why the policy lives here and not in the adapter
------------------------------------------------
Only 6 of Bodhan's 14 styles are moment-level. `anger`, `happy`, `sad`,
`fear`, `disgust` and `surprise` describe a few seconds of speech; the other
eight (`educational lecture`, `news`, `advertisements`, …) describe a whole
*programme* and cannot sensibly change mid-video. So an emotion label can
only ever select from that six, and every other segment falls back to the
run's global `--tts-style`. That split is the reason this module maps by
name rather than passing the model's label through.

Three guards, each of which exists because the naive version is worse
-------------------------------------------------------------------
1. **Confidence floor** (`min_confidence`). A 4-way or 9-way classifier is
   wrong often enough that acting on every argmax would repaint a neutral
   lecture with random affect. Below the floor the segment keeps the global
   style.
2. **Run-length hysteresis** (`min_run`). A lecture is overwhelmingly
   neutral; isolated single-segment emotions are usually classifier noise,
   and even when real, flipping style every few seconds is audible as
   jitter and re-renders the voice in a different mode mid-paragraph. A
   label must hold for `min_run` consecutive segments to be applied.
3. **Neutral is not a style.** `neutral`, `other` and `unknown` map to None,
   leaving the global register in place. Bodhan has no "neutral" style --
   omitting the key IS the neutral reading.

The honest caveat: none of this is worth running until the styles are shown
to differ audibly. `scripts/probe_emotion.py` prints the label distribution
for a job without touching the dub, which is the cheap way to find out
whether there is any signal on your content before wiring it in.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

#: Model label -> Bodhan style. Keys are normalised (lowercase, English side
#: of FunASR's "生气/angry"). Anything absent is deliberately unmapped.
#: `neutral`/`other`/`unknown` map to None so the global style survives --
#: Bodhan has no neutral style, and omitting the key is the neutral reading.
EMOTION_TO_STYLE: Dict[str, Optional[str]] = {
    "angry": "anger",
    "anger": "anger",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "fearful": "fear",
    "fear": "fear",
    "disgusted": "disgust",
    "disgust": "disgust",
    "surprised": "surprise",
    "surprise": "surprise",
    # explicit no-ops, listed so a reader can see they were considered.
    # `unk` is what FunASR actually emits (`<unk>/unk`); `unknown` is what
    # the docs say. Measured 2026-10-17 against emotion2vec_plus_large, whose
    # real vocabulary is the 9-class set -- angry, disgusted, fearful, happy,
    # neutral, other, sad, surprised, unk -- NOT the 5 classes its HF card
    # lists. That is why labels are read at runtime and mapped by name.
    "neutral": None,
    "other": None,
    "unk": None,
    "unknown": None,
}

DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_RUN = 2
#: Clips shorter than this are not worth scoring -- an utterance-level model
#: given 300 ms of speech returns noise. They keep the global style.
MIN_CLIP_S = 0.8


def map_label(label: str) -> Optional[str]:
    """Model label -> Bodhan style, or None to keep the global style."""
    return EMOTION_TO_STYLE.get((label or "").strip().lower())


def _runs(labels: List[Optional[str]]) -> List[int]:
    """Length of the constant run each index belongs to.

    `[a,a,b,a]` -> `[2,2,1,1]`. Used for hysteresis: a style is only applied
    where its own run is long enough, so one stray frame in the middle of a
    neutral stretch is suppressed while a genuine sustained shift survives.
    """
    out = [1] * len(labels)
    i = 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        for k in range(i, j + 1):
            out[k] = j - i + 1
        i = j + 1
    return out


def annotate_segments_with_emotion(
    merged_json: Path,
    *,
    source_audio: Path,
    model: str = "emotion2vec",
    clips_dir: Path,
    device: Optional[str] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_run: int = DEFAULT_MIN_RUN,
    global_style: Optional[str] = None,
    overwrite_existing: bool = False,
    model_kwargs: Optional[dict] = None,
) -> dict:
    """Score every segment's source audio and write `tts_style` in place.

    Returns a summary dict for the manifest: the label histogram, how many
    segments were actually styled, and why the rest were not. That summary
    is the point -- if it says 98% neutral, this stage is buying nothing on
    this content and should be turned back off rather than tuned.

    `overwrite_existing=False` (default) never clobbers a `tts_style` that
    `backend/edits.py` put on a segment by hand: an explicit human edit
    outranks a classifier.
    """
    import soundfile as sf
    from models.emotion import get_emotion

    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segments = data.get("segments") or []
    if not segments:
        return {"enabled": False, "reason": "no segments"}

    clips_dir = Path(clips_dir); clips_dir.mkdir(parents=True, exist_ok=True)

    # One read, then slice -- same shape as `voices.extract_speaker_references`
    # and the viseme path in `align.py`. N ffmpeg calls would dominate the
    # stage's wall time for no benefit.
    audio, sr = sf.read(str(source_audio), always_2d=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)

    clips: Dict[str, Path] = {}
    too_short: List[str] = []
    for seg in segments:
        sid = str(seg.get("id"))
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if end - start < MIN_CLIP_S:
            too_short.append(sid)
            continue
        s0 = max(0, int(round(start * sr)))
        s1 = min(len(audio), int(round(end * sr)))
        if s1 <= s0:
            too_short.append(sid)
            continue
        dest = clips_dir / f"{sid}.wav"
        if not dest.exists():
            sf.write(str(dest), audio[s0:s1], sr)
        clips[sid] = dest

    if not clips:
        log.warning("emotion: every segment shorter than %.1fs; nothing scored",
                    MIN_CLIP_S)
        return {"enabled": False, "reason": "all segments too short",
                "too_short": len(too_short)}

    adapter = get_emotion(model, device=device, **(model_kwargs or {}))
    log.info("emotion: scoring %d/%d segments with %s",
             len(clips), len(segments), model)
    batch = adapter.score(clips)
    by_id = batch.by_id()

    # --- map, then apply the two guards -------------------------------
    raw_labels = [by_id[str(s.get("id"))].label if str(s.get("id")) in by_id
                  else None for s in segments]
    mapped = [map_label(l) if l else None for l in raw_labels]
    runs = _runs(mapped)

    applied = 0
    skipped_conf = skipped_run = skipped_manual = 0
    for seg, style, run in zip(segments, mapped, runs):
        sid = str(seg.get("id"))
        res = by_id.get(sid)
        if res is not None:
            # Always record what the model saw, even when it is not acted
            # on -- that is what makes the histogram below trustworthy and
            # lets a later run re-tune the thresholds without re-scoring.
            seg["source_emotion"] = res.label
            seg["source_emotion_confidence"] = round(res.confidence, 4)
        if style is None:
            continue
        if seg.get("tts_style") and not overwrite_existing:
            skipped_manual += 1
            continue
        if res is None or res.confidence < min_confidence:
            skipped_conf += 1
            continue
        if run < min_run:
            skipped_run += 1
            continue
        seg["tts_style"] = style
        seg["tts_style_source"] = "emotion"
        applied += 1

    merged_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    hist = Counter(l for l in raw_labels if l)
    summary = {
        "enabled": True,
        "model": batch.model_name,
        "labels_reported": batch.labels,
        "segments_total": len(segments),
        "segments_scored": len(clips),
        "segments_styled": applied,
        "skipped_low_confidence": skipped_conf,
        "skipped_short_run": skipped_run,
        "skipped_manual_style": skipped_manual,
        "segments_too_short": len(too_short),
        "label_histogram": dict(hist.most_common()),
        "min_confidence": min_confidence,
        "min_run": min_run,
        "global_style_fallback": global_style,
        "coverage_note": batch.extra.get("coverage"),
    }
    top = hist.most_common(1)
    if top and top[0][1] / max(len(raw_labels), 1) > 0.9:
        # The outcome worth surfacing loudly: one label everywhere means the
        # stage is paying for itself in nothing.
        log.warning("emotion: %r on %d/%d segments — this stage is adding "
                    "almost no information on this content",
                    top[0][0], top[0][1], len(raw_labels))
    log.info("emotion: styled %d/%d segments (%s)",
             applied, len(segments), dict(hist.most_common(5)))
    return summary
