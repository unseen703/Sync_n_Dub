"""Evaluation layer for the timing-alignment study.

`reference` builds the ground truth every other metric is scored against.
Re-exported here so callers write `from backend.eval import vad, align_words`.
"""
from __future__ import annotations

from backend.eval.reference import (  # noqa: F401
    RefQC,
    Tracks,
    VadResult,
    Word,
    agreement_median_ms,
    align_words,
    build_reference,
    main,
    reference_qc,
    resolve_tracks,
    vad,
)

__all__ = [
    "Word", "VadResult", "RefQC", "Tracks",
    "vad", "align_words", "reference_qc", "build_reference",
    "resolve_tracks", "agreement_median_ms", "main",
]
