"""WhisperX adapter.

WhisperX wraps Whisper (CTranslate2 build) with wav2vec2 CTC forced
alignment, which is exactly the word-timestamp output the lip-sync and
duration stages need. Install: `pip install whisperx`.

Model sizes: tiny / base / small / medium / large-v2 / large-v3. For Indic
content, large-v3 is the only size that remains competitive; smaller models
drop to 20%+ WER on Hindi lecture audio.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import ASRResult, ASRSegment, BaseASR, Word

log = logging.getLogger(__name__)


class WhisperXASR(BaseASR):
    name = "whisperx"
    default_checkpoint = "large-v3"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        checkpoint: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: int = 16,
        language: Optional[str] = None,
        align: bool = True,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.4,
        vad_onset: float = 0.3,
        vad_offset: float = 0.25,
    ):
        super().__init__(device=device)
        self.checkpoint = checkpoint or self.default_checkpoint
        # float16 on CUDA, float32 elsewhere unless caller overrides.
        self.compute_type = compute_type or (
            "float16" if self.device.startswith("cuda") else "float32"
        )
        self.batch_size = batch_size
        self.language = language
        self.align = align
        # condition_on_previous_text=False stops the "repeat-the-same-phrase"
        # hallucination Whisper falls into on weak audio regions.
        # Lower vad_onset / no_speech_threshold = less aggressive silence
        # pruning, so quiet intros and over-music speech are not dropped.
        self.asr_options = {
            "condition_on_previous_text": condition_on_previous_text,
            "no_speech_threshold": no_speech_threshold,
        }
        self.vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}
        self._asr = None
        self._align_model = None
        self._align_meta = None
        self._align_lang: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import whisperx
        except ImportError as e:
            raise RuntimeError("`pip install whisperx` for the WhisperX adapter.") from e
        wx_device = "cuda" if self.device.startswith("cuda") else "cpu"
        # Older whisperx builds reject unknown kwargs; fall back gracefully.
        try:
            self._asr = whisperx.load_model(
                self.checkpoint,
                device=wx_device,
                compute_type=self.compute_type,
                language=self.language,
                asr_options=self.asr_options,
                vad_options=self.vad_options,
            )
        except TypeError:
            self._asr = whisperx.load_model(
                self.checkpoint,
                device=wx_device,
                compute_type=self.compute_type,
                language=self.language,
            )
            log.warning("whisperx: this build ignored asr_options/vad_options; "
                        "update whisperx to apply hallucination/VAD tuning.")
        self._loaded = True
        log.info("whisperx: loaded %s (compute=%s, device=%s)",
                 self.checkpoint, self.compute_type, wx_device)

    def _load_aligner(self, language: str):
        import whisperx
        if self._align_lang == language and self._align_model is not None:
            return
        wx_device = "cuda" if self.device.startswith("cuda") else "cpu"
        model, meta = whisperx.load_align_model(
            language_code=language, device=wx_device,
        )
        self._align_model, self._align_meta, self._align_lang = model, meta, language

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
        import whisperx
        audio = whisperx.load_audio(str(audio_path))
        lang = language or self.language

        wx_result = self._asr.transcribe(audio, batch_size=self.batch_size,
                                         language=lang)
        detected_lang = wx_result.get("language", lang)

        if self.align and detected_lang:
            try:
                self._load_aligner(detected_lang)
                import whisperx as _wx
                wx_result = _wx.align(
                    wx_result["segments"],
                    self._align_model, self._align_meta,
                    audio, "cuda" if self.device.startswith("cuda") else "cpu",
                    return_char_alignments=False,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("whisperx: alignment failed for lang=%s (%s); "
                            "returning segment-level timings only",
                            detected_lang, e)

        raw_segments = wx_result["segments"]
        asr_segments = _to_asr_segments(raw_segments, detected_lang)

        if segments is not None:
            # Re-slice to caller-provided turns. Anything outside a turn is
            # dropped — that's the lossy behaviour the pipeline historically
            # relied on. For full-audio coverage, callers pass segments=None
            # and layer speaker labels on afterwards.
            asr_segments = _intersect_with_turns(asr_segments, segments)

        return ASRResult(
            audio_path=Path(audio_path),
            model_name=f"whisperx/{self.checkpoint}",
            language=detected_lang,
            segments=asr_segments,
            extra={"compute_type": self.compute_type},
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_asr_segments(raw, lang: Optional[str]) -> List[ASRSegment]:
    out = []
    for i, s in enumerate(raw):
        words = []
        for w in s.get("words", []) or []:
            try:
                ws = float(w["start"]); we = float(w["end"])
            except (KeyError, TypeError):
                continue
            words.append(Word(
                text=str(w.get("word", "")).strip(),
                start=ws, end=we,
                score=float(w["score"]) if "score" in w else None,
            ))
        out.append(ASRSegment(
            id=f"SEG_{i:04d}",
            start=float(s["start"]),
            end=float(s["end"]),
            text=(s.get("text") or "").strip(),
            words=words,
            language=lang,
        ))
    return out


def _intersect_with_turns(
    asr_segments: List[ASRSegment],
    turns: List[tuple[str, float, float, Optional[str]]],
) -> List[ASRSegment]:
    """Re-slice ASR segments into caller-provided turns.

    Each turn becomes one `ASRSegment` whose `text` is the concatenation of
    words that fall inside the window. If the ASR had no word timestamps,
    we fall back to overlapping segment text with proportional cut.
    """
    out = []
    for sid, ts, te, spk in turns:
        words_in: List[Word] = []
        text_bits: List[str] = []
        for s in asr_segments:
            if s.end < ts or s.start > te:
                continue
            if s.words:
                for w in s.words:
                    if w.end < ts or w.start > te:
                        continue
                    words_in.append(w)
                    text_bits.append(w.text)
            else:
                text_bits.append(s.text)
        out.append(ASRSegment(
            id=sid,
            start=ts, end=te,
            text=" ".join(bit for bit in text_bits if bit),
            speaker=spk,
            words=words_in,
            language=asr_segments[0].language if asr_segments else None,
        ))
    return out
