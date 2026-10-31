"""Shared types and abstract base for machine-translation adapters.

An adapter takes source strings plus a target language (and optional script)
and returns the translations in order. Translation is per-segment, so the
manifest's `segments[].mt_text` / `mt_candidates` stays the single owner of
segment-level timing.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# Length-penalty sweep for n-best. Beam search scores logP / len**penalty, so
# <1 rewards short output and >1 long; sweeping it is what buys the length
# spread the duration ranker needs.
DEFAULT_LENGTH_PENALTIES: Tuple[float, ...] = (0.6, 1.0, 1.4)

# Diverse beam search needs groups to disagree. 1.0 is the point where Hindi
# hypotheses stop being word-for-word identical without going off-meaning.
DEFAULT_DIVERSITY_PENALTY: float = 1.0


@dataclass
class MTRequest:
    id: str
    text: str
    source_language: Optional[str] = None
    target_script: Optional[str] = None       # "native" | "roman" | None
    # How many alternative wordings the caller wants back. 1 keeps the old
    # single-hypothesis path exactly as it was.
    n_best: int = 1
    # Slot length this segment has to fit, in seconds. No adapter reads it
    # today; it rides along for the manifest and for a future length-aware
    # decoder.
    length_hint_s: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class MTSegment:
    id: str
    source_text: str
    text: str                           # the translated text
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    target_script: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class MTResult:
    model_name: str
    source_language: Optional[str]
    target_language: str
    target_script: Optional[str]
    segments: List[MTSegment]
    extra: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "model_name": self.model_name,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "target_script": self.target_script,
            "segments": [
                {
                    "id": s.id,
                    "source_text": s.source_text,
                    "text": s.text,
                    "source_language": s.source_language,
                    "target_language": s.target_language,
                    "target_script": s.target_script,
                    "extra": s.extra,
                }
                for s in self.segments
            ],
            "extra": self.extra,
        }


class BaseMT(ABC):
    """Every MT adapter subclasses this."""

    name: str = "base"
    # True only for adapters that override translate_nbest with a real
    # multi-hypothesis decode. The base implementation returns one candidate.
    supports_nbest: bool = False

    def __init__(self, device: Optional[str] = None, **kwargs):
        from backend.diarize.base import _resolve_device
        self.device = device or _resolve_device()
        self._loaded = False

    @abstractmethod
    def _ensure_loaded(self) -> None: ...

    @abstractmethod
    def translate(
        self,
        requests: List[MTRequest],
        *,
        target_language: str,
        target_script: Optional[str] = None,
        source_language: Optional[str] = None,
    ) -> MTResult:
        """Translate every request to `target_language`. Order is preserved."""

    def translate_nbest(
        self,
        requests: List[MTRequest],
        *,
        target_language: str,
        target_script: Optional[str] = None,
        source_language: Optional[str] = None,
        n_best: int = 4,
        length_penalties: Optional[Sequence[float]] = None,
        diversity_penalty: Optional[float] = None,
    ) -> MTResult:
        """Return up to `n_best` alternative wordings per request.

        The default delegates to `translate()` and reports the one hypothesis
        as a single-element candidate list. Overrides set
        `supports_nbest = True` and must honour two rules: `MTSegment.text`
        stays byte-identical to what `translate()` would return, and
        `extra["candidates"]` is a deduped `List[str]`, best first, with
        `candidates[0] == text`.
        """
        if n_best > 1 and not self.supports_nbest:
            log.info("%s: n_best=%d requested but adapter has no n-best path; "
                     "returning 1 candidate per segment", self.name, n_best)
        result = self.translate(
            requests,
            target_language=target_language,
            target_script=target_script,
            source_language=source_language,
        )
        for seg in result.segments:
            if seg is None:
                continue
            seg.extra = dict(seg.extra or {})
            seg.extra.setdefault("candidates", [seg.text] if seg.text else [])
        result.extra = {**(result.extra or {}),
                        "n_best_requested": int(n_best),
                        "n_best_supported": bool(self.supports_nbest)}
        return result


# ---------------------------------------------------------------------------
# shared n-best helpers (adapters reuse these so dedup/beam rules stay one rule)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
# Devanagari danda / double danda plus the Latin sentence enders. Hypotheses
# differing only by a trailing danda are the same for duration purposes.
_TRIM_CHARS = " \t\r\n.,;:!?।॥‘’“”\"'"


def candidate_key(text: str) -> str:
    """Normalised dedup key: NFC, case-folded, whitespace- and edge-punct-free.

    Not the stored form — callers dedupe on this but keep the original surface
    string, since TTS pronounces punctuation-driven pauses.
    """
    t = unicodedata.normalize("NFC", (text or "")).strip()
    t = _WS_RE.sub(" ", t)
    return t.strip(_TRIM_CHARS).casefold()


def dedupe_candidates(texts: Sequence[str], *, limit: Optional[int] = None) -> List[str]:
    """Order-preserving dedupe by `candidate_key`, dropping empties.

    Order in is priority out, so callers put the `translate()` top-1 first and
    the sweep behind it.
    """
    seen: set[str] = set()
    out: List[str] = []
    for raw in texts:
        text = (raw or "").strip()
        if not text:
            continue
        key = candidate_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


# transformers >= 4.57 serves group beam search from this hub repo instead of
# `generate()`. It downloads and executes third-party code, so it is opt-in via
# MT_GROUP_BEAM_REMOTE=1. Roughly triples batch time (~2.0 s against ~0.7 s) in
# exchange for more candidate length spread.
GROUP_BEAM_REMOTE_REPO = "transformers-community/group-beam-search"


def group_beam_remote_kwargs() -> dict:
    """Extra `generate()` kwargs that re-enable group beam search on
    transformers >= 4.57, or `{}` when the opt-in is off or not needed."""
    import os
    if os.getenv("MT_GROUP_BEAM_REMOTE", "").strip().lower() not in ("1", "true", "yes"):
        return {}
    try:
        import transformers
        parts = transformers.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except Exception:
        return {}
    if (major, minor) < (4, 57):
        return {}
    return {"custom_generate": GROUP_BEAM_REMOTE_REPO, "trust_remote_code": True}


def plain_beam_fallback(gen_kwargs: dict) -> dict:
    """Strip the group-beam kwargs so the pass runs as plain beam search.

    On transformers >= 4.57 `num_beam_groups=` raises unless `custom_generate=`
    and `trust_remote_code=True` are also passed, and an unhandled raise leaves
    one candidate per segment. Dropping the group knobs keeps the same
    `length_penalty`, which is the knob the duration ranker consumes: diversity
    within a pass is lost, spread across the sweep is not. Returns the kwargs
    unchanged when there was nothing to strip.
    """
    return {k: v for k, v in gen_kwargs.items()
            if k not in ("num_beam_groups", "diversity_penalty",
                         "custom_generate", "trust_remote_code")}


def plan_beam_groups(
    n_best: int, num_beams: int, *, beams_per_group: int = 1,
) -> Tuple[int, int, int]:
    """Pick (num_beams, num_beam_groups, num_return_sequences) for diverse beam.

    transformers requires `num_beams % num_beam_groups == 0` and
    `num_return_sequences <= num_beams`. One beam per group is the maximally
    diverse legal configuration; plain beam returns K near-identical
    hypotheses off one tree, with near-zero length spread.
    """
    k = max(1, int(n_best))
    groups = max(1, k)
    beams = groups * max(1, int(beams_per_group))
    # Never decode with fewer beams than the adapter's configured quality
    # setting; round up to the next whole number of groups.
    floor = max(1, int(num_beams))
    if beams < floor:
        beams = groups * -(-floor // groups)
    return beams, groups, min(k, beams)
