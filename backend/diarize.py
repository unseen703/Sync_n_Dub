"""Backend-facing entry point for diarization.

Keeps the model registry out of `pipeline.py` and gives the CLI and UI one
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
    """Diarize a mono 16 kHz WAV with the named model.

    The speaker-count arguments are hints, honoured per adapter. `rttm_out`
    also writes an RTTM there. `model_kwargs` go to the adapter constructor,
    with unknown keys dropped and warned about, so passing the union of what
    any adapter might want is safe.
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
