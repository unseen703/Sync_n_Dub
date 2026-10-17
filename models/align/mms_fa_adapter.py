"""Meta MMS Forced Aligner (torchaudio bundle).

Uses `torchaudio.pipelines.MMS_FA` — a wav2vec2-based CTC emission model
plus a shared 1,100+ language romanized dictionary. Alignment is produced
by running the emissions through torchaudio.functional.forced_align over
each segment's text.

The MMS dictionary is **character-level over romanized text**. For Indic
scripts (Devanagari, Tamil, …) we must transliterate the text to Latin
first; otherwise every character maps to the CTC blank. We use the `uroman`
package when present — it is what the MMS authors use. Non-Latin segments
without uroman installed are skipped (empty `words`), not crashed, so a
mixed-script transcript still aligns the Latin parts.

Install: `pip install uroman-python` (optional) for Indic script support.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from models.asr.base import Word

from .base import AlignRequest, AlignResult, AlignedSegment, BaseAligner

log = logging.getLogger(__name__)

# Characters torchaudio MMS_FA's dictionary accepts (lowercase a-z + ').
_LATIN_WORD_RE = re.compile(r"[A-Za-z']+")


class MMSForcedAligner(BaseAligner):
    name = "mms_fa"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        sample_rate: int = 16000,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self._model = None
        self._tokenizer = None
        self._aligner = None
        self._dictionary = None
        self._uroman = None
        self._uroman_tried = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch  # noqa: F401
            from torchaudio.pipelines import MMS_FA
        except ImportError as e:
            raise RuntimeError(
                "torchaudio>=2.2 is required for MMS forced alignment."
            ) from e
        bundle = MMS_FA
        import torch
        self._model = bundle.get_model(with_star=False).to(self.device).eval()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()
        self._dictionary = bundle.get_dict()
        self._loaded = True
        log.info("mms_fa: loaded torchaudio MMS_FA on %s", self.device)

    def _get_uroman(self):
        if self._uroman is not None or self._uroman_tried:
            return self._uroman
        self._uroman_tried = True
        try:
            import uroman as ur
            self._uroman = ur.Uroman()
        except ImportError:
            log.warning(
                "mms_fa: `uroman` not installed; non-Latin segments will be "
                "skipped. `pip install uroman` to enable Indic alignment."
            )
        return self._uroman

    # ------------------------------------------------------------------
    # alignment
    # ------------------------------------------------------------------

    def align(
        self,
        audio_path: Path,
        requests: List[AlignRequest],
        *,
        language: Optional[str] = None,
    ) -> AlignResult:
        self._ensure_loaded()
        import soundfile as sf
        import torch

        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            raise ValueError(
                f"mms_fa expects {self.sample_rate} Hz; got {sr}. Resample upstream."
            )

        out: List[AlignedSegment] = []
        for req in requests:
            text = (req.text or "").strip()
            if not text or req.end <= req.start:
                out.append(AlignedSegment(
                    id=req.id, start=req.start, end=req.end, text=text,
                    speaker=req.speaker, language=req.language or language,
                ))
                continue

            native_tokens = _split_words(text)
            paired = _romanize_pairs(native_tokens, self._get_uroman())
            if not paired:
                out.append(AlignedSegment(
                    id=req.id, start=req.start, end=req.end, text=text,
                    speaker=req.speaker, language=req.language or language,
                ))
                continue
            words_latin = [latin for _, latin in paired]

            s0 = max(0, int(round(req.start * sr)))
            s1 = min(len(audio), int(round(req.end * sr)))
            clip = audio[s0:s1]
            if clip.size < sr // 20:    # <50 ms
                out.append(AlignedSegment(
                    id=req.id, start=req.start, end=req.end, text=text,
                    speaker=req.speaker, language=req.language or language,
                ))
                continue

            try:
                word_spans, num_frames = self._align_clip(clip, words_latin, torch)
            except Exception as e:  # noqa: BLE001
                log.warning("mms_fa: segment %s failed (%s); keeping empty words",
                            req.id, e)
                word_spans, num_frames = [], 0

            words: List[Word] = []
            if word_spans and num_frames:
                sec_per_frame = (clip.shape[0] / sr) / num_frames
                for (native_tok, _latin), spans in zip(paired, word_spans):
                    if not spans:
                        continue
                    ws = req.start + spans[0].start * sec_per_frame
                    we = req.start + spans[-1].end * sec_per_frame
                    score = float(np.mean([sp.score for sp in spans]))
                    words.append(Word(text=native_tok, start=ws, end=we, score=score))

            out.append(AlignedSegment(
                id=req.id, start=req.start, end=req.end, text=text,
                words=words, speaker=req.speaker,
                language=req.language or language,
            ))

        return AlignResult(
            audio_path=audio_path,
            model_name=self.name,
            language=language,
            segments=out,
        )

    # ------------------------------------------------------------------

    def _align_clip(self, clip: np.ndarray, words_latin: List[str], torch_mod):
        """Returns (one TokenSpan list per word, number of emission frames)."""
        wav = torch_mod.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0).to(self.device)
        with torch_mod.inference_mode():
            emission, _ = self._model(wav)
        token_lists = self._tokenizer(words_latin)
        token_spans = self._aligner(emission[0], token_lists)
        return token_spans, int(emission.shape[1])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _split_words(text: str) -> List[str]:
    """Whitespace-tokenise; keep the original surface forms for display."""
    return [w for w in text.split() if w.strip()]


def _romanize_pairs(words: List[str], uroman_inst) -> List[tuple[str, str]]:
    """Return (native_token, latin_token) pairs, dropping tokens that
    collapse to the empty string after romanisation + normalisation.
    """
    out: List[tuple[str, str]] = []
    for w in words:
        norm = _normalise(w)
        if not norm and uroman_inst is not None:
            try:
                norm = _normalise(uroman_inst.romanize_string(w))
            except Exception:
                norm = ""
        if norm:
            out.append((w, norm))
    return out


def _normalise(s: str) -> str:
    s = s.lower()
    kept = _LATIN_WORD_RE.findall(s)
    return "".join(kept)
