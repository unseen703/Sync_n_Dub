"""Text sanitization for the ASR → MT → TTS pipeline.

Three functions, one per stage boundary:

  sanitize_for_mt(text)   – NFC-normalize ASR output before translation,
                            reducing OOV → <unk> in the MT model.
  sanitize_mt_output(text) – strip literal <unk> and whitespace artifacts
                             from MT output before codemix/duration.
  sanitize_for_tts(text)  – defense-in-depth cleanup before synthesis:
                            special tokens, NFC, zero-width chars, whitespace.
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

    SentencePiece MT vocabularies are trained on NFC text. Feeding NFD
    (decomposed) forms causes the same character to be OOV, producing
    <unk> tokens in the translation.
    """
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


def sanitize_mt_output(text: str, seg_id: str = "") -> str:
    """Strip literal <unk> and collapse whitespace after MT decoding.

    `skip_special_tokens=True` only removes tokens by ID.  When a model
    generates the literal characters '<unk>' as regular text (seen on
    rare conjuncts), they survive decoding.
    Catching them here prevents junk from reaching codemix and duration.
    """
    cleaned = _JUNK_TOKENS.sub("", text)
    cleaned = cleaned.replace(_SP_UNDERSCORE, " ")
    cleaned = _MULTI_WS.sub(" ", cleaned).strip()
    if cleaned != text.strip():
        log.warning("sanitize: MT seg %s had junk tokens: %r -> %r",
                    seg_id, text.strip(), cleaned)
    return cleaned


def sanitize_for_tts(text: str, seg_id: str = "") -> str:
    """Defense-in-depth cleanup before TTS synthesis.

    Catches anything that leaked through earlier stages: special tokens,
    non-NFC Unicode, zero-width characters, SentencePiece artifacts,
    and excess whitespace.  Keeps ZWJ (U+200D) which is needed for
    correct Indic conjunct rendering.
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
