"""Backend-facing ASR entry point.

One call site for the CLI and pipeline over the `models.asr` registry,
turning diarized turns into per-segment windows when they are given.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from models.asr import ASRResult, available_models, get_asr

log = logging.getLogger(__name__)


def transcribe(
    audio: Path,
    *,
    model: str = "bodhan",
    device: Optional[str] = None,
    language: Optional[str] = None,
    turns: Optional[list] = None,          # List[models.diarization.base.Turn]
    **model_kwargs,
) -> ASRResult:
    """Transcribe `audio` with `model`, optionally sliced to diarized turns.

    With `turns`, each turn becomes one `ASRSegment` id'd `SEG_NNNN`, which is
    the shape the manifest's `segments[]` expects.
    """
    audio = Path(audio)
    if not audio.exists():
        raise FileNotFoundError(audio)
    if model not in available_models():
        raise ValueError(
            f"Unknown ASR model {model!r}. Available: {', '.join(available_models())}"
        )
    log.info("asr: %s with %s (device=%s, lang=%s, turns=%s)",
             audio.name, model, device or "auto", language or "auto",
             "yes" if turns else "no")

    segments: Optional[List[Tuple[str, float, float, Optional[str]]]] = None
    if turns:
        segments = [
            (f"SEG_{i:04d}", float(t.start), float(t.end), t.speaker)
            for i, t in enumerate(turns)
        ]

    asr = get_asr(model, device=device, **model_kwargs)
    result = asr.transcribe(audio, language=language, segments=segments)
    log.info("asr: %d segments (model=%s, lang=%s)",
             len(result.segments), result.model_name, result.language)
    return result
