"""Stage 9b2 — per-segment delivery style from the source speaker's emotion.

Scores each segment's original audio and writes a `tts_style` that
`backend/tts.py` picks up, guarded by a confidence floor and run-length
hysteresis. Only the six moment-level Bodhan styles can be selected; anything
else keeps the run's global `--tts-style`.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

#: Model label -> Bodhan style. Keys are normalised (lowercase, English side
#: of FunASR's "生气/angry"); anything absent is deliberately unmapped.
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
    # Explicit no-ops. `unk` is what FunASR emits, `unknown` is what the docs
    # say. emotion2vec_plus_large really has 9 classes (angry, disgusted,
    # fearful, happy, neutral, other, sad, surprised, unk), not the 5 on its HF
    # card — hence mapping by name at runtime.
    "neutral": None,
    "other": None,
    "unk": None,
    "unknown": None,
}

DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MIN_RUN = 2
#: Shorter clips are not worth scoring — an utterance-level model given
#: 300 ms of speech returns noise. They keep the global style.
MIN_CLIP_S = 0.8


def map_label(label: str) -> Optional[str]:
    """Model label -> Bodhan style, or None to keep the global style."""
    return EMOTION_TO_STYLE.get((label or "").strip().lower())


def _runs(labels: List[Optional[str]]) -> List[int]:
    """Length of the constant run each index belongs to: `[a,a,b,a]` ->
    `[2,2,1,1]`. Drives the hysteresis guard."""
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

    Returns a manifest summary: the label histogram, how many segments were
    styled and why the rest were not. A histogram that is 98% neutral means
    the stage is buying nothing on this content.

    `overwrite_existing=False` never clobbers a `tts_style` set by hand in
    `backend/edits.py` — an explicit human edit outranks the classifier.
    """
    import soundfile as sf
    from models.emotion import get_emotion

    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segments = data.get("segments") or []
    if not segments:
        return {"enabled": False, "reason": "no segments"}

    clips_dir = Path(clips_dir); clips_dir.mkdir(parents=True, exist_ok=True)

    # One read, then slice: N ffmpeg calls would dominate the stage's wall
    # time for no benefit.
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
            # Record what the model saw even when it is not acted on, so the
            # thresholds can be re-tuned without re-scoring.
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
    from .mt_versions import snapshot as _mt_snapshot
    _mt_snapshot(merged_json, "emotion", data=data)

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
        # One label everywhere means the stage is adding nothing.
        log.warning("emotion: %r on %d/%d segments — this stage is adding "
                    "almost no information on this content",
                    top[0][0], top[0][1], len(raw_labels))
    log.info("emotion: styled %d/%d segments (%s)",
             applied, len(segments), dict(hist.most_common(5)))
    return summary
