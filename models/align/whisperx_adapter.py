"""WhisperX forced-alignment adapter.

Reuses WhisperX's `load_align_model` + `align` so a non-WhisperX ASR (Bodhan,
IndicConformer, …) can still get wav2vec2-CTC word boundaries.

WhisperX ships a per-language table of HF wav2vec2 CTC heads — one per
language it supports. Coverage of Indic scripts is partial; the aligner
raises if asked for a language whose head is not registered. We catch and
log that case, returning an empty-words segment rather than killing the run.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from models.asr.base import Word

from .base import AlignRequest, AlignResult, AlignedSegment, BaseAligner

log = logging.getLogger(__name__)


class WhisperXAligner(BaseAligner):
    name = "whisperx"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        model_name: Optional[str] = None,       # override HF repo id
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self._align_model = None
        self._align_meta = None
        self._align_lang: Optional[str] = None

    def _ensure_loaded(self) -> None:
        # Nothing to load until we know the language — alignment heads are
        # per-language.
        self._loaded = True

    def _load_for(self, language: str) -> bool:
        try:
            import whisperx
        except ImportError as e:
            raise RuntimeError(
                "`pip install whisperx` for the WhisperX aligner."
            ) from e
        if self._align_lang == language and self._align_model is not None:
            return True
        wx_device = "cuda" if self.device.startswith("cuda") else "cpu"
        try:
            model, meta = whisperx.load_align_model(
                language_code=language, device=wx_device,
                model_name=self.model_name,
            )
        except (ValueError, KeyError) as e:
            log.warning("whisperx aligner: no CTC head for language %r (%s)",
                        language, e)
            self._align_model = None
            self._align_lang = language
            return False
        self._align_model, self._align_meta, self._align_lang = model, meta, language
        return True

    # ------------------------------------------------------------------

    def align(
        self,
        audio_path: Path,
        requests: List[AlignRequest],
        *,
        language: Optional[str] = None,
    ) -> AlignResult:
        self._ensure_loaded()

        # Group by language — WhisperX loads a per-language model.
        by_lang: dict[Optional[str], List[int]] = {}
        for i, r in enumerate(requests):
            lang = (r.language or language or "").strip() or None
            by_lang.setdefault(lang, []).append(i)

        out: List[Optional[AlignedSegment]] = [None] * len(requests)

        import whisperx
        wx_device = "cuda" if self.device.startswith("cuda") else "cpu"
        audio = whisperx.load_audio(str(audio_path))

        for lang, idxs in by_lang.items():
            # Build the list of whisperx-style segments for this language.
            wx_segments = []
            for i in idxs:
                r = requests[i]
                text = (r.text or "").strip()
                if not text or r.end <= r.start:
                    out[i] = AlignedSegment(
                        id=r.id, start=r.start, end=r.end, text=text,
                        speaker=r.speaker, language=r.language or language,
                    )
                    continue
                wx_segments.append({
                    "_orig_idx": i,
                    "start": float(r.start),
                    "end": float(r.end),
                    "text": text,
                })

            if not wx_segments or lang is None:
                for seg in wx_segments:
                    i = seg["_orig_idx"]
                    r = requests[i]
                    out[i] = AlignedSegment(
                        id=r.id, start=r.start, end=r.end, text=r.text,
                        speaker=r.speaker, language=r.language or language,
                    )
                continue

            if not self._load_for(lang):
                for seg in wx_segments:
                    i = seg["_orig_idx"]
                    r = requests[i]
                    out[i] = AlignedSegment(
                        id=r.id, start=r.start, end=r.end, text=r.text,
                        speaker=r.speaker, language=r.language or language,
                    )
                continue

            try:
                aligned = whisperx.align(
                    [{k: v for k, v in s.items() if not k.startswith("_")}
                     for s in wx_segments],
                    self._align_model, self._align_meta,
                    audio, wx_device, return_char_alignments=False,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("whisperx aligner: align call failed for lang %r (%s)",
                            lang, e)
                for seg in wx_segments:
                    i = seg["_orig_idx"]
                    r = requests[i]
                    out[i] = AlignedSegment(
                        id=r.id, start=r.start, end=r.end, text=r.text,
                        speaker=r.speaker, language=r.language or language,
                    )
                continue

            # whisperx.align does NOT return one segment per input: it splits
            # long segments, so `aligned["segments"]` can be longer than
            # `wx_segments`. Zipping the two silently pairs segment k's words
            # with request k+1 and drops the tail -- measured on
            # 2vxQaT0Fy-c_1min, where a 12.4 s turn split in two and every
            # later segment inherited its predecessor's word stream, leaving
            # 19.3 s of words for a 60 s clip. 17 of the 142 merged JSONs on
            # disk carry that corruption, and it is invisible downstream
            # because the times are still monotonic and plausible.
            #
            # Assign by TIME instead of by position: pool every aligned word
            # and give it to the request whose [start, end] contains its
            # midpoint. Order-independent, and robust to both splits and drops.
            pooled: List[Word] = []
            for aligned_seg in aligned.get("segments", []) or []:
                for w in aligned_seg.get("words", []) or []:
                    try:
                        ws = float(w["start"]); we = float(w["end"])
                    except (KeyError, TypeError):
                        continue
                    pooled.append(Word(
                        text=str(w.get("word", "")).strip(),
                        start=ws, end=we,
                        score=float(w["score"]) if "score" in w else None,
                    ))
            pooled.sort(key=lambda w: w.start)

            buckets: Dict[int, List[Word]] = {s["_orig_idx"]: [] for s in wx_segments}
            spans = sorted(
                ((float(s["start"]), float(s["end"]), s["_orig_idx"]) for s in wx_segments),
                key=lambda t: t[0])
            unassigned = 0
            for w in pooled:
                mid = (w.start + w.end) / 2.0
                hit = None
                for a, b, idx in spans:
                    if a <= mid <= b:
                        hit = idx
                        break
                if hit is None:
                    # Outside every turn (diarization gap). Attach to the
                    # nearest span rather than discarding a real word.
                    hit = min(spans, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]
                    unassigned += 1
                buckets[hit].append(w)
            if unassigned:
                log.debug("whisperx aligner: %d/%d words fell outside every turn "
                          "and were snapped to the nearest", unassigned, len(pooled))

            for seg in wx_segments:
                i = seg["_orig_idx"]
                r = requests[i]
                out[i] = AlignedSegment(
                    id=r.id, start=r.start, end=r.end, text=r.text,
                    words=buckets[i], speaker=r.speaker,
                    language=r.language or language,
                )

        # Nothing should be None at this point, but belt-and-braces.
        return AlignResult(
            audio_path=audio_path,
            model_name=self.name,
            language=language,
            segments=[s if s is not None else AlignedSegment(
                id=requests[i].id, start=requests[i].start, end=requests[i].end,
                text=requests[i].text or "", speaker=requests[i].speaker,
                language=requests[i].language or language)
                for i, s in enumerate(out)],
        )
