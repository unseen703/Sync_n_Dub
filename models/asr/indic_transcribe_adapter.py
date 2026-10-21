"""Local HuggingFace adapter for Bodhan Indic-Transcribe (Core / Flex).

HF repos:
  bodhan-ai/indic-transcribe-core  — Canary 1.2B, best accuracy (8.7 WER)
  bodhan-ai/indic-transcribe-flex  — Canary 1.2B, extra modes (11.3 WER)

Both repos ship their own Python inference code (``indic_transcribe``
module) wrapping a FastConformer encoder + Transformer decoder (NVIDIA
Canary architecture, 1.2B params). Loading adds the downloaded snapshot
to ``sys.path``, imports ``IndicTranscribe``, and calls
``from_pretrained``.

Single-pass decoding caps at 45 s. The bundled ``long_form.transcribe_long``
helper splits at natural pauses into ≤30 s chunks with anti-repetition
guards, and is what we use for segments longer than 30 s.

Output modes (both models):
  native    — target-language script (default)
  mixed     — native script with English loanwords/digits in Latin
  romanized — all Latin transliteration (alias: romanised)

Code-mixed speech (e.g. Hinglish) is transcribed as spoken in every mode.
``mixed`` mode further writes English tokens in Latin script, which is the
natural match for the code-mix stage downstream.

Language identification: ``asr.identify(audio)`` returns ranked candidates.
When ``lang=None`` is passed, identification runs automatically before
transcription (costs one decoder step, not a second encoder pass).

Gated model — ``hf auth login`` once, or set ``HF_TOKEN`` in ``.env``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .base import ASRResult, ASRSegment, BaseASR

log = logging.getLogger(__name__)

CORE_REPO = "bodhan-ai/indic-transcribe-core"
FLEX_REPO = "bodhan-ai/indic-transcribe-flex"

# The 27 language codes the canary checkpoint was trained on (read from the
# model's own LANGUAGES constant). Both Core and Flex share this set.
TRAINED_LANGS = {
    "as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "hne",
    "kn", "kok", "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa",
    "sat", "sd", "ta", "te", "ur",
}

# Modes the models accept: maps CLI / user-facing names to what the model expects.
VALID_MODES = {"native", "mixed", "romanized", "romanised", "itn"}

MAX_SINGLE_PASS_S = 45.0
PREFER_LONG_FORM_S = 30.0


class IndicTranscribeASR(BaseASR):
    """Local canary-architecture ASR via bodhan-ai/indic-transcribe-*."""

    name = "indictranscribe"
    supported_languages = TRAINED_LANGS

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_id: Optional[str] = None,
        mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        language: Optional[str] = None,
        return_lid: bool = False,
    ):
        super().__init__(device=device)
        self.repo_id = repo_id or CORE_REPO
        # Accept CLI's --render-mode and map to model vocabulary.
        raw = mode or render_mode
        if raw == "roman":
            raw = "romanized"
        if raw and raw not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(VALID_MODES - {'romanised', 'itn'})}, "
                f"got {raw!r}"
            )
        self.mode = raw or "native"
        self.language = language
        self.return_lid = return_lid
        self._model = None
        self._model_dir: Optional[str] = None
        self._transcribe_long_fn = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required. "
                "`pip install huggingface_hub torch torchaudio transformers "
                "sentencepiece soundfile`"
            ) from e

        log.info("indic_transcribe: downloading %s (gated — needs HF_TOKEN) …",
                 self.repo_id)
        self._model_dir = snapshot_download(self.repo_id)

        if self._model_dir not in sys.path:
            sys.path.insert(0, self._model_dir)

        from indic_transcribe import IndicTranscribe  # type: ignore[import-untyped]
        self._model = IndicTranscribe.from_pretrained(
            self._model_dir, device=self.device,
        )

        try:
            from long_form import transcribe_long  # type: ignore[import-untyped]
            self._transcribe_long_fn = transcribe_long
        except ImportError:
            log.warning("indic_transcribe: long_form.transcribe_long not found "
                        "in %s; long-audio chunking disabled", self._model_dir)

        self._loaded = True
        log.info("indic_transcribe: loaded %s from %s (device=%s, mode=%s)",
                 self.repo_id, self._model_dir, self.device, self.mode)

    # ------------------------------------------------------------------
    # inference helpers
    # ------------------------------------------------------------------

    def _call_model(self, audio_path: str, lang: Optional[str]) -> tuple[str, Optional[dict]]:
        """Transcribe one clip (≤45 s). Returns (text, lid_info_or_None)."""
        text = self._model(
            audio_path,
            lang=lang,
            mode=self.mode,
            return_lid=self.return_lid,
        )
        if isinstance(text, tuple):
            return (text[0] or "").strip(), text[1]
        return (text or "").strip(), None

    def _call_long(self, audio_path: str, lang: Optional[str]) -> str:
        """Transcribe audio of any length via pause-based chunking."""
        if self._transcribe_long_fn is None:
            return self._call_model(audio_path, lang)[0]
        text = self._transcribe_long_fn(
            self._model, audio_path, lang, mode=self.mode,
        )
        return (text or "").strip()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        segments: Optional[List[tuple[str, float, float, Optional[str]]]] = None,
    ) -> ASRResult:
        self._ensure_loaded()
        audio_path = Path(audio_path)
        lang = (language or self.language or "").lower() or None

        # Full-file mode (no diarized turns).
        if segments is None:
            duration = self._audio_duration(audio_path)
            if duration > PREFER_LONG_FORM_S:
                text = self._call_long(str(audio_path), lang)
                lid = None
            else:
                text, lid = self._call_model(str(audio_path), lang)

            detected_lang = lang
            if lid and lid.get("source") == "lid":
                detected_lang = lid.get("lang", lang)

            return ASRResult(
                audio_path=audio_path,
                model_name=self.repo_id,
                language=detected_lang,
                segments=[ASRSegment(
                    id="SEG_0000",
                    start=0.0, end=duration,
                    text=text, language=detected_lang,
                )],
                extra={"mode": self.mode, "lid": lid},
            )

        # Per-segment mode (diarized turns).
        import soundfile as sf
        import tempfile

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        out_segments: List[ASRSegment] = []
        detected_lang: Optional[str] = lang
        for sid, start_s, end_s, speaker in segments:
            s0 = max(0, int(round(start_s * sr)))
            s1 = min(len(audio), int(round(end_s * sr)))
            clip = audio[s0:s1]
            if clip.size == 0:
                out_segments.append(ASRSegment(
                    id=sid, start=start_s, end=end_s, text="",
                    speaker=speaker, language=lang,
                ))
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip, sr)
                clip_dur = len(clip) / sr
                if clip_dur > PREFER_LONG_FORM_S and self._transcribe_long_fn:
                    text = self._call_long(tmp.name, lang)
                    lid = None
                else:
                    text, lid = self._call_model(tmp.name, lang)

            seg_lang = lang
            if lid and lid.get("source") == "lid":
                seg_lang = lid.get("lang", lang)
                detected_lang = detected_lang or seg_lang

            out_segments.append(ASRSegment(
                id=sid, start=start_s, end=end_s,
                text=text, speaker=speaker, language=seg_lang,
            ))

        return ASRResult(
            audio_path=audio_path,
            model_name=self.repo_id,
            language=detected_lang,
            segments=out_segments,
            extra={"mode": self.mode},
        )

    # ------------------------------------------------------------------
    # language identification
    # ------------------------------------------------------------------

    def identify(
        self,
        audio_path: Path,
        topk: int = 5,
        allowed_langs: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """Identify the spoken language without transcribing.

        Returns [(lang_code, probability), ...] ranked best-first.
        """
        self._ensure_loaded()
        return self._model.identify(
            str(audio_path), topk=topk, allowed_langs=allowed_langs,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _audio_duration(path: Path) -> float:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return float(info.duration)
        except Exception:
            return 0.0


class IndicTranscribeCoreASR(IndicTranscribeASR):
    """Indic-Transcribe Core — canary architecture, best accuracy.

    bodhan-ai/indic-transcribe-core (1.2B params, 8.7 WER on VOI benchmark).
    Supports 27 languages including code-mixed speech. Output in native script
    by default; ``mode="mixed"`` gives Latin-script English tokens.
    """
    name = "indictranscribe_core"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_id: Optional[str] = None,
        mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        language: Optional[str] = None,
        return_lid: bool = False,
    ):
        super().__init__(
            device=device, repo_id=repo_id or CORE_REPO,
            mode=mode, render_mode=render_mode,
            language=language, return_lid=return_lid,
        )


class IndicTranscribeFlexASR(IndicTranscribeASR):
    """Indic-Transcribe Flex — canary architecture, extra output modes.

    bodhan-ai/indic-transcribe-flex (1.2B params, 11.3 WER on VOI benchmark).
    Same 27 languages as Core. Adds romanized output mode for full Latin-script
    transcription. Trade-off: slightly lower accuracy than Core.
    """
    name = "indictranscribe_flex"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        repo_id: Optional[str] = None,
        mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        language: Optional[str] = None,
        return_lid: bool = False,
    ):
        super().__init__(
            device=device, repo_id=repo_id or FLEX_REPO,
            mode=mode, render_mode=render_mode,
            language=language, return_lid=return_lid,
        )
