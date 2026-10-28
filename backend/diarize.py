"""Thin wrapper over models.diarization — the backend-facing entry point.

Keeps the model registry out of pipeline.py and gives the CLI/frontend one
place to import from.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from models.diarization import (
    DiarizationResult,
    available_models,
    get_diarizer,
)

log = logging.getLogger(__name__)


def diarize(
    audio: Path,
    *,
    model: str = "pyannote",
    device: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    rttm_out: Optional[Path] = None,
    **model_kwargs,
) -> DiarizationResult:
    """Diarize `audio` with the named model.

    Args:
      audio: mono 16 kHz WAV.
      model: one of `available_models()` — pyannote.
      device: 'cuda' | 'cpu' | 'mps' | None (auto).
      num_speakers / min_speakers / max_speakers: hints, honored per-adapter.
      rttm_out: if given, also writes an RTTM file to this path.
      **model_kwargs: forwarded to the adapter constructor. Unknown kwargs
        are dropped with a warning, so it is safe to pass the union of what
        any adapter might want. Per-adapter accepted keys:
          pyannote:  checkpoint, hf_token

    Returns a `DiarizationResult`.
    """
    audio = Path(audio)
    if not audio.exists():
        raise FileNotFoundError(audio)
    if model not in available_models():
        raise ValueError(
            f"Unknown diarization model {model!r}. "
            f"Available: {', '.join(available_models())}"
        )

    log.info("diarize: %s with %s on %s", audio.name, model, device or "auto")
    diarizer = get_diarizer(model, device=device, **model_kwargs)
    result = diarizer.diarize(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    log.info(
        "diarize: %d turns, %d speakers (model=%s)",
        len(result.turns), result.num_speakers, result.model_name,
    )

    if rttm_out is not None:
        result.write_rttm(Path(rttm_out))
        log.info("diarize: wrote %s", rttm_out)

    return result
