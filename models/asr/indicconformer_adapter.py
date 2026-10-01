"""AI4Bharat IndicConformer adapter (NeMo Conformer-CTC).

HF cards:
  ai4bharat/indic-conformer-600m-multilingual  (recommended default)
  ai4bharat/indic-conformer-200m-{lang}        (per-language variants)

The 600M multilingual model needs a language code at inference time via the
NeMo config knob exposed as `set_decoding_strategy` on the model. For the
per-language 200M checkpoints the code is baked in.

Install: `pip install nemo-toolkit[asr]>=2.0.0` (already in requirements for
the Sortformer diarizer).
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import ASRResult, ASRSegment, BaseASR, Word

log = logging.getLogger(__name__)


class IndicConformerASR(BaseASR):
    name = "indicconformer"
    default_checkpoint = "ai4bharat/indic-conformer-600m-multilingual"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        batch_size: int = 1,
        language: Optional[str] = None,
    ):
        super().__init__(device=device)
        self.checkpoint = checkpoint or self.default_checkpoint
        self.batch_size = batch_size
        self.language = language
        self._model = None

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as e:
            raise RuntimeError(
                "nemo-toolkit[asr] is required for IndicConformer. "
                "`pip install nemo-toolkit[asr]`."
            ) from e
        model = ASRModel.from_pretrained(self.checkpoint)
        if self.device.startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                model = model.cuda()
        model.eval()
        self._model = model
        self._loaded = True
        log.info("indicconformer: loaded %s on %s", self.checkpoint, self.device)

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
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(
                f"IndicConformer expects 16 kHz; got {sr}. Resample upstream."
            )

        self._set_language(language or self.language)

        if segments is None:
            # Full-file inference. NeMo's `transcribe` accepts a list of
            # paths; we give it the one we have.
            texts = self._model.transcribe([str(audio_path)],
                                           batch_size=self.batch_size)
            text = _flatten(texts)
            return ASRResult(
                audio_path=Path(audio_path),
                model_name=self.checkpoint,
                language=language,
                segments=[ASRSegment(
                    id="SEG_0000",
                    start=0.0, end=len(audio) / sr,
                    text=text,
                    language=language,
                )],
            )

        # Per-turn: slice to a scratch WAV per window — NeMo wants file paths
        # or a dataset manifest, so a tempfile per segment is the simplest
        # route and keeps GPU usage low on short lecture clips.
        out: List[ASRSegment] = []
        tmp_wavs: List[Path] = []
        try:
            for sid, start_s, end_s, speaker in segments:
                s0 = max(0, int(round(start_s * sr)))
                s1 = min(len(audio), int(round(end_s * sr)))
                clip = audio[s0:s1]
                if clip.size == 0:
                    out.append(ASRSegment(
                        id=sid, start=start_s, end=end_s, text="",
                        speaker=speaker, language=language,
                    ))
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                tmp_wavs.append(Path(tmp.name))
                sf.write(tmp.name, clip, sr)

            batch_paths = [str(p) for p in tmp_wavs]
            if batch_paths:
                texts = self._model.transcribe(batch_paths, batch_size=self.batch_size)
                flat = [_flatten([t]) for t in texts]
            else:
                flat = []

            idx = 0
            for sid, start_s, end_s, speaker in segments:
                s0 = max(0, int(round(start_s * sr)))
                s1 = min(len(audio), int(round(end_s * sr)))
                if s1 - s0 <= 0:
                    continue
                text = flat[idx] if idx < len(flat) else ""
                idx += 1
                out.append(ASRSegment(
                    id=sid, start=start_s, end=end_s, text=text,
                    speaker=speaker, language=language,
                ))
        finally:
            for p in tmp_wavs:
                try: p.unlink()
                except OSError: pass

        return ASRResult(
            audio_path=Path(audio_path),
            model_name=self.checkpoint,
            language=language,
            segments=out,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _set_language(self, language: Optional[str]) -> None:
        """The 600M multilingual checkpoint exposes `cur_decoder` + a language
        token through its config; the 200M per-language ones ignore the call.
        """
        if not language:
            return
        try:
            # NeMo >=2.0 way
            self._model.cur_decoder = "ctc"
            if hasattr(self._model, "change_decoding_strategy"):
                self._model.change_decoding_strategy(decoding_cfg=None,
                                                    decoder_type="ctc")
            # Multilingual IndicConformer accepts a `language_id` kwarg on
            # transcribe in some releases; others want it baked into the model
            # cfg. We mutate the cfg if available.
            cfg = getattr(self._model, "cfg", None)
            if cfg is not None and hasattr(cfg, "language"):
                cfg.language = language
        except Exception as e:  # noqa: BLE001
            log.warning("indicconformer: could not set language=%s (%s)", language, e)


def _flatten(x):
    """NeMo returns `list[str]` or `list[tuple[str, _]]` depending on version."""
    if not x:
        return ""
    first = x[0]
    if isinstance(first, tuple):
        return first[0] or ""
    if isinstance(first, list):
        return _flatten(first)
    return str(first)
