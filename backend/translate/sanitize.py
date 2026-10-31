"""Text sanitization at the ASR -> MT -> TTS stage boundaries.

  sanitize_for_mt     NFC-normalize ASR output before translation
  sanitize_mt_output  strip literal <unk> and whitespace artifacts after MT
  sanitize_for_tts    final cleanup before synthesis
"""
from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

_JUNK_TOKENS = re.compile(r"<(?:unk|s|/s|pad|mask|blank)>")
_ZW_CHARS = re.compile(r"[​‌﻿]")
_SP_UNDERSCORE = "▁"
_MULTI_WS = re.compile(r"\s+")


def sanitize_for_mt(text: str) -> str:
    """NFC-normalize ASR output before it enters translation.

    SentencePiece MT vocabularies are trained on NFC, so a decomposed form of
    the same character is OOV and translates to <unk>.
    """
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


def sanitize_mt_output(text: str, seg_id: str = "") -> str:
    """Strip literal <unk> and collapse whitespace after MT decoding.

    `skip_special_tokens=True` removes tokens by ID only, so a model that
    emits the literal characters `<unk>` as text survives decoding.
    """
    cleaned = _JUNK_TOKENS.sub("", text)
    cleaned = cleaned.replace(_SP_UNDERSCORE, " ")
    cleaned = _MULTI_WS.sub(" ", cleaned).strip()
    if cleaned != text.strip():
        log.warning("sanitize: MT seg %s had junk tokens: %r -> %r",
                    seg_id, text.strip(), cleaned)
    return cleaned


def sanitize_for_tts(text: str, seg_id: str = "") -> str:
    """Final cleanup before TTS synthesis.

    Catches whatever leaked through earlier stages: special tokens, non-NFC
    Unicode, zero-width characters, SentencePiece artifacts and excess
    whitespace. Keeps ZWJ (U+200D), which Indic conjuncts need.
    """
    original = text
    text = _JUNK_TOKENS.sub("", text)
    text = text.replace(_SP_UNDERSCORE, " ")
    text = unicodedata.normalize("NFC", text)
    text = _ZW_CHARS.sub("", text)
    text = _MULTI_WS.sub(" ", text).strip()
    if text != original.strip():
        log.warning("sanitize: TTS seg %s cleaned: %r -> %r",
                    seg_id, original.strip(), text)
    return text
