"""NVIDIA NeMo Sortformer diarization adapter (what this project calls
`nemotron` for CLI brevity).

Model: `nvidia/diar_sortformer_4spk-v1` — end-to-end neural diarization,
4-speaker cap, Apache-2.0 **code** and CC-BY-NC weights per the plan §8
licence column. The HF card calls out that mathematical cap: anything beyond
4 concurrent speakers is folded into the top-4 by the model itself.

Install: NeMo (`nemo-toolkit[asr]>=2.0`); weights are pulled on first use via
`SortformerEncLabelModel.from_pretrained` into the HF cache.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from .base import BaseDiarizer, DiarizationResult, Turn, canonicalize_speaker_labels

log = logging.getLogger(__name__)


class NemotronDiarizer(BaseDiarizer):
    name = "nemotron"
    default_checkpoint = "nvidia/diar_sortformer_4spk-v1"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        batch_size: int = 1,
    ):
        super().__init__(device=device)
        self.checkpoint = checkpoint or self.default_checkpoint
        self.batch_size = batch_size
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            raise RuntimeError(
                "`pip install nemo-toolkit[asr]>=2.0` for the NeMo Sortformer "
                "diarizer."
            ) from e

        # Pick a map_location that matches our target device so the first
        # forward doesn't blow up with device-mismatched buffers.
        map_location = "cuda" if self.device.startswith("cuda") else "cpu"
        # Local .nemo path, or HF repo id.
        if os.path.exists(self.checkpoint):
            model = SortformerEncLabelModel.restore_from(
                self.checkpoint, map_location=map_location,
            )
        else:
            model = SortformerEncLabelModel.from_pretrained(
                self.checkpoint, map_location=map_location,
            )
        model.eval()
        self._model = model
        self._loaded = True
        log.info("nemotron: loaded %s on %s", self.checkpoint, self.device)

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        if any(x is not None for x in (num_speakers, min_speakers, max_speakers)):
            log.info(
                "nemotron: speaker-count hints are ignored (Sortformer has a "
                "fixed 4-speaker head)."
            )
        self._ensure_loaded()
        audio_path = Path(audio_path)

        # One-click API. Returns List[List[str]]: outer per input file,
        # inner strings of "begin_s end_s speaker_idx".
        results = self._model.diarize(
            audio=str(audio_path),
            batch_size=self.batch_size,
            verbose=False,
        )
        if not results:
            lines: List[str] = []
        elif isinstance(results[0], str):
            lines = results                  # single-file flattened
        else:
            lines = results[0]

        turns: List[Turn] = []
        for line in lines:
            parts = str(line).split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue
            speaker = f"SPK_{int(parts[2]):02d}" if parts[2].isdigit() else str(parts[2])
            if end > start:
                turns.append(Turn(start=start, end=end, speaker=speaker))

        turns.sort(key=lambda t: (t.start, t.end))
        turns = canonicalize_speaker_labels(turns)

        return DiarizationResult(
            turns=turns,
            audio_path=audio_path,
            model_name=self.checkpoint,
            extra={"backend": "nemo.sortformer", "batch_size": self.batch_size},
        )
