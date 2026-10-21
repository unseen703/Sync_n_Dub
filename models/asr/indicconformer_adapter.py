"""AI4Bharat IndicConformer adapter.

HF: ai4bharat/indic-conformer-600m-multilingual

The repo ships as a HuggingFace "custom code" model (not a NeMo .nemo
checkpoint): it exposes `IndicASRConfig` + `IndicASRModel` through its own
`model_onnx.py`, and the actual weights are per-language ONNX files under
`assets/`. Loading goes through `transformers.AutoModel` with
`trust_remote_code=True`; the model's `forward(wav, lang, decoding='ctc', ...)`
runs one clip at a time and returns the transcript string.

A language code is REQUIRED — the multilingual model uses it to select the
per-language joint head + vocab mask. Supported codes:
    as, bn, brx, doi, gu, hi, kn, kok, ks, mai, ml, mni, mr, ne, or, pa,
    sa, sat, sd, ta, te, ur
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import ASRResult, ASRSegment, BaseASR

log = logging.getLogger(__name__)

SUPPORTED_LANGS = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml",
    "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}


class IndicConformerASR(BaseASR):
    name = "indicconformer"
    default_checkpoint = "ai4bharat/indic-conformer-600m-multilingual"
    supported_languages = SUPPORTED_LANGS

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        batch_size: int = 1,              # kept for CLI compat; model is per-clip
        language: Optional[str] = None,
        decoding: str = "ctc",            # "ctc" or "rnnt"
    ):
        super().__init__(device=device)
        self.checkpoint = checkpoint or self.default_checkpoint
        self.batch_size = batch_size
        self.language = language
        self.decoding = decoding
        self._model = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for IndicConformer. "
                "`pip install transformers huggingface_hub onnxruntime-gpu`."
            ) from e
        # This model builds its own onnxruntime sessions inside the bundled
        # `model_onnx.py` and lets the EPs pick themselves, so the CUDA EP has
        # to be loadable *before* the model is constructed or it quietly runs
        # on CPU. See backend.gpu.ensure_onnxruntime_cuda.
        from backend.gpu import ensure_onnxruntime_cuda
        ensure_onnxruntime_cuda()

        # The repo bundles `model_onnx.py` — trust_remote_code is required.
        self._model = AutoModel.from_pretrained(
            self.checkpoint, trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True
        log.info("indicconformer: loaded %s (device picks itself per onnxruntime EPs)",
                 self.checkpoint)

    # ------------------------------------------------------------------
    # transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        segments: Optional[List[tuple[str, float, float, Optional[str]]]] = None,
    ) -> ASRResult:
        self._ensure_loaded()
        lang = (language or self.language or "").lower()
        if lang not in SUPPORTED_LANGS:
            raise ValueError(
                f"IndicConformer requires a supported language code; got {lang!r}. "
                f"Supported: {sorted(SUPPORTED_LANGS)}"
            )

        import soundfile as sf
        import torch

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(
                f"IndicConformer expects 16 kHz; got {sr}. Resample upstream."
            )

        if segments is None:
            text = self._run_clip(audio, lang, torch)
            return ASRResult(
                audio_path=Path(audio_path),
                model_name=self.checkpoint,
                language=lang,
                segments=[ASRSegment(
                    id="SEG_0000",
                    start=0.0, end=float(len(audio)) / sr,
                    text=text, language=lang,
                )],
            )

        out: List[ASRSegment] = []
        for sid, start_s, end_s, speaker in segments:
            s0 = max(0, int(round(start_s * sr)))
            s1 = min(len(audio), int(round(end_s * sr)))
            clip = audio[s0:s1]
            text = self._run_clip(clip, lang, torch) if clip.size else ""
            out.append(ASRSegment(
                id=sid, start=start_s, end=end_s, text=text,
                speaker=speaker, language=lang,
            ))
        return ASRResult(
            audio_path=Path(audio_path),
            model_name=self.checkpoint,
            language=lang,
            segments=out,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _run_clip(self, clip: np.ndarray, lang: str, torch_mod) -> str:
        if clip.size == 0:
            return ""
        wav = torch_mod.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0)
        with torch_mod.no_grad():
            out = self._model(wav, lang, decoding=self.decoding)
        # forward returns a string (CTC) or a (hyp, timestamps) tuple when
        # compute_timestamps is set. We don't request timestamps.
        if isinstance(out, tuple):
            out = out[0]
        return (out or "").strip()
