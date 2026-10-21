"""AI4Bharat IndicF5 adapter.

IndicF5 is a near-human expressive, zero-shot cloning TTS covering 11
Indian languages. It is published on HuggingFace as
`ai4bharat/IndicF5` and loads via `AutoModel.from_pretrained(..., trust_remote_code=True)`.

Interface:
  model(text, ref_audio_path=<wav>, ref_text=<transcript of ref clip>)

The adapter requires each `TTSRequest` to carry `reference_audio` AND
`reference_text` (zero-shot cloning needs both) — upstream fills these
from the per-speaker reference clips `backend.voices.extract_speaker_references`
produces, with `reference_text` set to a short ASR transcript of that
clip (first sentence of the longest turn is a good default).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .base import BaseTTS, TTSRequest, TTSResult, TTSSegment

log = logging.getLogger(__name__)


SUPPORTED = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_REPO = "ai4bharat/IndicF5"


class IndicF5TTS(BaseTTS):
    name = "indicf5"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_id: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.repo_id = repo_id or DEFAULT_REPO
        self.dtype = dtype
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModel
        kwargs = {"trust_remote_code": True}
        if self.dtype in ("float16", "bfloat16"):
            kwargs["torch_dtype"] = getattr(torch, self.dtype)
        log.info("indicf5: loading %s onto %s", self.repo_id, self.device)
        self._model = AutoModel.from_pretrained(self.repo_id, **kwargs).to(self.device)
        self._model.eval()
        self._loaded = True

    # ------------------------------------------------------------------

    def synthesize(
        self,
        requests: List[TTSRequest],
        *,
        out_dir: Path,
        language: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> TTSResult:
        self._ensure_loaded()
        lang = (language or "").split("-")[0].lower()
        if lang and lang not in SUPPORTED:
            raise ValueError(
                f"IndicF5 does not support language {language!r}. "
                f"Supported: {sorted(SUPPORTED)}"
            )
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import soundfile as sf

        segments: List[TTSSegment] = []
        for req in requests:
            text = (req.text or "").strip()
            dest = out_dir / f"{req.id}.wav"
            if not text:
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=sample_rate, voice=req.voice,
                    speaker=req.speaker, error="empty text",
                ))
                continue
            if not req.reference_audio or not req.reference_text:
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=sample_rate, voice=req.voice,
                    speaker=req.speaker,
                    error="IndicF5 needs reference_audio + reference_text",
                ))
                continue
            try:
                audio = self._model(
                    text,
                    ref_audio_path=str(req.reference_audio),
                    ref_text=req.reference_text,
                )
                audio = np.asarray(audio, dtype="float32")
                if audio.dtype.kind == "i":
                    audio = audio.astype("float32") / 32768.0
                sf.write(str(dest), audio, sample_rate)
                dur = len(audio) / float(sample_rate)
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=dur,
                    sample_rate=sample_rate, voice=req.voice,
                    speaker=req.speaker,
                ))
            except Exception as e:
                log.warning("indicf5: synth failed for %s: %s", req.id, e)
                segments.append(TTSSegment(
                    id=req.id, audio_path=dest, duration_s=0.0,
                    sample_rate=sample_rate, voice=req.voice,
                    speaker=req.speaker, error=str(e),
                ))

        return TTSResult(
            model_name=f"indicf5/{self.repo_id}",
            out_dir=out_dir, language=language, segments=segments,
        )
