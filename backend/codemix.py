"""Code-mixing and script normalization (plan §6).

Renders a `final_text` per segment: detect spans that should stay English,
transliterate them into the target script, re-script any Latin word left in
the MT output, and restore terms MT translated that it should not have.
Writes back into the merged ASR+MT JSON.

The termbase is `termbase/<domain>.tsv`, two tab-separated columns
`source\treplacement`, `#` comments ignored.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from models.translit import TranslitRequest, available_models, get_transliterator

log = logging.getLogger(__name__)


# Anything in the Latin script that could hold a word.
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")
# Numbers with units, e.g. "5kg", "3.14ms", "128GB".
_NUMUNIT_RE = re.compile(r"\b\d+(?:\.\d+)?[A-Za-z]+\b")
# ALL-CAPS acronyms of length 2+, e.g. "GPU", "API", "MIT".
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
# A capitalised token in mid-sentence — rough proper-noun proxy.
_PROPER_RE = re.compile(r"(?<=[a-z\s,])[A-Z][a-z]+")


@dataclass
class RetainSpan:
    text: str
    start: int            # character offsets into the SOURCE ASR text
    end: int
    type: str             # "termbase" | "numunit" | "acronym" | "proper"
    translit: Optional[str] = None   # filled by the transliteration pass


# ---------------------------------------------------------------------------
# termbase
# ---------------------------------------------------------------------------

def load_termbase(path: Optional[Path]) -> Dict[str, str]:
    """Read a two-column TSV. Returns {source_term_lower: replacement}."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        log.warning("termbase not found: %s", p)
        return {}
    out: Dict[str, str] = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        src, rep = parts[0].strip(), parts[1].strip()
        if src and rep:
            out[src.lower()] = rep
    log.info("termbase: %d term(s) loaded from %s", len(out), p)
    return out


# ---------------------------------------------------------------------------
# span detection
# ---------------------------------------------------------------------------

def detect_retain_spans(
    text: str,
    *,
    termbase: Dict[str, str],
) -> List[RetainSpan]:
    """Non-overlapping spans to keep in English form.

    First match wins, in the order termbase > numunit > acronym > proper.
    """
    hits: List[Tuple[int, int, str, str]] = []
    low = text.lower()
    for term in sorted(termbase, key=len, reverse=True):
        start = 0
        while True:
            i = low.find(term, start)
            if i < 0:
                break
            hits.append((i, i + len(term), text[i:i + len(term)], "termbase"))
            start = i + len(term)
    for pat, kind in ((_NUMUNIT_RE, "numunit"),
                      (_ACRONYM_RE, "acronym"),
                      (_PROPER_RE, "proper")):
        for m in pat.finditer(text):
            hits.append((m.start(), m.end(), m.group(), kind))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))

    out: List[RetainSpan] = []
    cursor = 0
    for s, e, t, kind in hits:
        if s < cursor:
            continue
        out.append(RetainSpan(text=t, start=s, end=e, type=kind))
        cursor = e
    return out


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def extract_latin_tokens(text: str) -> List[str]:
    """Latin-script runs in the MT output: candidates to re-script."""
    return list({m.group() for m in _LATIN_WORD_RE.finditer(text)})


def apply_substitutions(
    mt_text: str,
    spans: Iterable[RetainSpan],
    *,
    termbase: Dict[str, str],
) -> str:
    """Replace Latin remnants in `mt_text` with their transliteration.

    Longest first, so "gradient descent" wins over "gradient". A termbase entry
    beats the transliteration — it is the domain expert's override.
    """
    out = mt_text
    pairs: List[Tuple[str, str]] = []
    for sp in spans:
        rep = termbase.get(sp.text.lower()) or sp.translit
        if rep and rep != sp.text:
            pairs.append((sp.text, rep))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    for src, rep in pairs:
        pattern = re.compile(
            r"(?<![A-Za-z])" + re.escape(src) + r"(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        out = pattern.sub(rep, out)
    return _nfc(out)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def codemix_segments(
    merged_mt_json: Path,
    *,
    target_language: str,
    target_script: str = "native",
    translit_model: str = "bodhan",
    termbase_path: Optional[Path] = None,
    out_json: Optional[Path] = None,
    translit_kwargs: Optional[dict] = None,
) -> dict:
    """Run the full code-mix and script-normalisation pass.

    Reads the merged ASR+MT JSON, writes it back in place by default, and
    returns the updated dict.
    """
    merged_mt_json = Path(merged_mt_json)
    data = json.loads(merged_mt_json.read_text())

    if translit_model not in available_models():
        raise ValueError(
            f"Unknown transliterator {translit_model!r}. "
            f"Available: {', '.join(available_models())}"
        )
    termbase = load_termbase(termbase_path) if termbase_path else {}

    # Collect every unique Latin token across all segments so the translit API
    # is called once per term, not once per occurrence.
    seg_spans: Dict[str, List[RetainSpan]] = {}
    unique_terms: Dict[str, None] = {}
    for seg in data.get("segments", []):
        src = seg.get("text") or ""
        spans = detect_retain_spans(src, termbase=termbase)
        # also catch Latin tokens left in the MT text itself
        for tok in extract_latin_tokens(seg.get("mt_text") or ""):
            if tok and not any(sp.text.lower() == tok.lower() for sp in spans):
                spans.append(RetainSpan(text=tok, start=-1, end=-1, type="latin_in_mt"))
        seg_spans[seg["id"]] = spans
        for sp in spans:
            if sp.text.lower() not in termbase:
                unique_terms.setdefault(sp.text, None)

    # 2. Transliterate the unique terms in one batch.
    translit_map: Dict[str, str] = {}
    if unique_terms:
        log.info("codemix: %d unique Latin term(s) → %s/%s via %s",
                 len(unique_terms), target_language, target_script, translit_model)
        tr = get_transliterator(translit_model, **(translit_kwargs or {}))
        requests = [
            TranslitRequest(id=f"T_{i}", text=t, target_script=target_script)
            for i, t in enumerate(unique_terms)
        ]
        tr_result = tr.transliterate(
            requests, target_language=target_language, target_script=target_script,
        )
        for req, seg in zip(requests, tr_result.segments):
            translit_map[req.text] = seg.text or req.text
    else:
        log.info("codemix: no Latin terms to transliterate")

    # 3. Rewrite each segment.
    retained_total = 0
    substituted_total = 0
    for seg in data.get("segments", []):
        spans = seg_spans.get(seg["id"], [])
        for sp in spans:
            sp.translit = translit_map.get(sp.text) or sp.text
        mt_text = seg.get("mt_text") or ""
        final = apply_substitutions(mt_text, spans, termbase=termbase)
        seg["retain_spans"] = [
            {"text": sp.text, "start": sp.start, "end": sp.end,
             "type": sp.type, "translit": sp.translit}
            for sp in spans
        ]
        seg["final_text"] = final
        seg["codemix_path"] = "B" if spans else "A"
        retained_total += len(spans)
        if final != mt_text:
            substituted_total += 1

    data.setdefault("extra", {})["codemix"] = {
        "translit_model": translit_model,
        "target_language": target_language,
        "target_script": target_script,
        "retained_terms": retained_total,
        "substituted_segments": substituted_total,
        "termbase_size": len(termbase),
    }

    dest = Path(out_json) if out_json else merged_mt_json
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    from .mt_versions import snapshot as _mt_snapshot
    _mt_snapshot(dest, "codemix", data=data)
    log.info("codemix: wrote %s (retained=%d, substituted=%d segs)",
             dest, retained_total, substituted_total)
    return data
