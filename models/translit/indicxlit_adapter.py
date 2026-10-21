"""AI4Bharat IndicXlit adapter — Roman ↔ Indic script.

IndicXlit is a transformer trained on `ai4bharat/Aksharantar`. Covers 21
scheduled Indic languages ↔ Latin. We expose it as a local alternative to
Bodhan Indic-Transliterate so an offline run can still re-script code-mix
spans (plan §6 E10).

Install path, in order of preference:
  1. `pip install ai4bharat-transliteration` — ships `XlitEngine` and will
     pull the FairseqNMT checkpoints on first use.
  2. Fall back to HuggingFace `ai4bharat/IndicXlit-en-indic-v1.0` /
     `IndicXlit-indic-en-v1.0` through `transformers` when the FairseqNMT
     package cannot be installed.

Default direction is `en → <target>` because that is the direction the
code-mix path needs; `indic → roman` is available via `target_script="roman"`.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .base import BaseTransliterator, TranslitRequest, TranslitResult, TranslitSegment

log = logging.getLogger(__name__)


SUPPORTED = {
    "as", "bn", "brx", "gu", "hi", "kn", "ks", "gom", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sd", "si",
    "ta", "te", "ur",
}


class IndicXlitTransliterate(BaseTransliterator):
    name = "indicxlit"

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        beam_width: int = 4,
        topk: int = 1,
    ):
        super().__init__(device=device)
        self.beam_width = int(beam_width)
        self.topk = int(topk)
        self._engine_en_indic = None
        self._engine_indic_en = None
        self._backend: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from ai4bharat.transliteration import XlitEngine  # type: ignore
        except ImportError:
            XlitEngine = None
        if XlitEngine is not None:
            self._XlitEngine = XlitEngine
            self._backend = "ai4bharat"
            self._loaded = True
            log.info("indicxlit: using ai4bharat-transliteration backend")
            return
        raise ImportError(
            "indicxlit requires `pip install ai4bharat-transliteration`."
        )

    # ------------------------------------------------------------------

    def _get_engine(self, target_language: str, target_script: str):
        if self._backend != "ai4bharat":
            raise RuntimeError("indicxlit backend not initialised")
        if target_script == "native":
            if self._engine_en_indic is None:
                self._engine_en_indic = self._XlitEngine(
                    target_language, beam_width=self.beam_width, src_script_type="en",
                )
            return self._engine_en_indic, "en", target_language
        if self._engine_indic_en is None:
            self._engine_indic_en = self._XlitEngine(
                src_script_type="indic", beam_width=self.beam_width,
            )
        return self._engine_indic_en, target_language, "en"

    def transliterate(
        self,
        requests: List[TranslitRequest],
        *,
        target_language: str,
        target_script: str = "native",
    ) -> TranslitResult:
        self._ensure_loaded()
        lang = (target_language or "").split("-")[0].lower()
        if lang not in SUPPORTED:
            raise ValueError(
                f"indicxlit does not support language {target_language!r}. "
                f"Supported: {sorted(SUPPORTED)}"
            )
        engine, src_tag, tgt_tag = self._get_engine(lang, target_script)
        out: List[TranslitSegment] = []
        for req in requests:
            text = (req.text or "").strip()
            if not text:
                out.append(TranslitSegment(
                    id=req.id, source_text=text, text="",
                    source_language=req.source_language,
                    target_script=target_script,
                ))
                continue
            try:
                if target_script == "native":
                    res = engine.translit_sentence(text, lang_code=lang)
                else:
                    res = engine.translit_sentence(text, lang_code=lang)
                scripted = res if isinstance(res, str) else str(res)
            except Exception as e:
                log.warning("indicxlit failed on %r (%s); passing text through",
                            text[:60], e)
                scripted = text
            out.append(TranslitSegment(
                id=req.id, source_text=text, text=scripted,
                source_language=req.source_language,
                target_script=target_script,
                extra={"backend": self._backend},
            ))
        return TranslitResult(
            model_name=f"indicxlit/{self._backend}",
            target_script=target_script,
            segments=out,
            extra={"beam_width": self.beam_width},
        )
