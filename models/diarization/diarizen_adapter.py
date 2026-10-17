"""DiariZen diarization adapter.

Repo: https://github.com/BUTSpeechFIT/DiariZen
Weights (default): BUT-FIT/diarizen-wavlm-large-s80-md — CC-BY-NC-4.0.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import BaseDiarizer, DiarizationResult, Turn, canonicalize_speaker_labels


class DiariZenDiarizer(BaseDiarizer):
    name = "diarizen"
    default_checkpoint = "BUT-FIT/diarizen-wavlm-large-s80-md"
    licence = "CC-BY-NC-4.0"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint = checkpoint or self.default_checkpoint
        # Accepted for symmetry with the pyannote adapter and forwarded to HF
        # Hub if the DiariZen checkpoint is ever gated.
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
        except ImportError as e:
            raise RuntimeError(
                "DiariZen not installed. `pip install diarizen>=0.1.0` "
                "(or install from git: pip install git+https://github.com/BUTSpeechFIT/DiariZen)"
            ) from e

        try:
            pipeline = DiariZenPipeline.from_pretrained(
                self.checkpoint, use_auth_token=self.hf_token,
            )
        except TypeError:
            # older DiariZen releases do not accept use_auth_token — fall back.
            pipeline = DiariZenPipeline.from_pretrained(self.checkpoint)
        try:
            import torch
            pipeline = pipeline.to(torch.device(self.device))
        except Exception:
            # Not every DiariZen version exposes .to() at the pipeline level.
            pass

        self._pipeline = pipeline
        self._loaded = True

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        self._ensure_loaded()
        audio_path = Path(audio_path)

        # DiariZen pipeline call returns a pyannote-style Annotation object.
        annotation = self._pipeline(str(audio_path))

        turns: list[Turn] = []
        # Duck-type: Annotation has itertracks; otherwise expect iterable of (segment, label).
        if hasattr(annotation, "itertracks"):
            for seg, _track, label in annotation.itertracks(yield_label=True):
                turns.append(Turn(start=float(seg.start), end=float(seg.end), speaker=str(label)))
        else:
            for row in annotation:
                seg, label = row[0], row[-1]
                turns.append(Turn(start=float(seg.start), end=float(seg.end), speaker=str(label)))

        turns.sort(key=lambda t: (t.start, t.end))
        turns = canonicalize_speaker_labels(turns)

        return DiarizationResult(
            turns=turns,
            audio_path=audio_path,
            model_name=self.checkpoint,
            extra={"backend": "diarizen", "licence": self.licence},
        )
