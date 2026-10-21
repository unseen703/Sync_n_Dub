"""Text-quality metrics over MT / codemix manifest segments.

Deliberately absent: a METEOR wrapper. NLTK's METEOR runs stemming and
WordNet synonym matching, both English-only. Point it at Devanagari and the
stem/synonym stages contribute nothing — it silently degenerates to an
exact-match unigram F-score wearing METEOR's name, which reads as a semantic
metric and is not one. chrF++ is the right surface metric for Indic output
and is what `chrfpp_report` returns.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

from .timing import akshara_count

__all__ = [
    "script_of",
    "script_profile",
    "safe_cer",
    "term_retention",
    "term_consistency",
    "chrfpp_report",
    "length_ratio",
    "roundtrip_mt_wer",
]


def _require(pkg: str, pip_name: Optional[str] = None):
    """Import `pkg` or raise an ImportError that names the pip package."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise ImportError(
            f"Text evaluation needs `{pip_name or pkg}`. "
            f"Install it with `pip install {pip_name or pkg}`."
        ) from e


def _norm(s: str) -> str:
    """NFC + casefold: the comparison form for term matching.

    NFC first because Devanagari transliteration output arrives in mixed
    normalisation and a decomposed nukta would never equal a composed one.
    """
    return unicodedata.normalize("NFC", s or "").casefold()


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def script_profile(text: str) -> Dict[str, Any]:
    """Per-script letter counts, ignoring digits, punctuation and whitespace."""
    deva = latin = other = 0
    for ch in unicodedata.normalize("NFC", text or ""):
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
            deva += 1
        elif (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A) or (0xC0 <= cp <= 0x24F):
            latin += 1
        else:
            other += 1
    total = deva + latin + other
    return {
        "deva": deva, "latin": latin, "other": other, "letters": total,
        "deva_frac": (deva / total) if total else 0.0,
        "latin_frac": (latin / total) if total else 0.0,
        "other_frac": (other / total) if total else 0.0,
        "dominant": (max((("deva", deva), ("latin", latin), ("other", other)),
                         key=lambda kv: kv[1])[0] if total else "other"),
    }


def script_of(text: str) -> str:
    """"deva" | "latin" | "mixed" | "other".

    Strict: any Devanagari AND any Latin letter makes it "mixed", because
    code-mixed output is exactly what the codemix stage is supposed to
    produce and a threshold would hide a single untransliterated term.
    """
    p = script_profile(text)
    if p["letters"] == 0:
        return "other"
    if p["deva"] and p["latin"]:
        return "mixed"
    if p["deva"]:
        return "deva"
    if p["latin"]:
        return "latin"
    return "other"


def safe_cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """CER, but REFUSED when reference and hypothesis are in different scripts.

    `data/eval/nsIMQP0QhpE_60s.indicconformer.hi.asr_eval.json` reports
    CER 0.934. That is a Latin hypothesis scored against a Devanagari
    reference: no character can ever match, so the number is a constant
    near 1.0 that says nothing about the model and everything about the
    output script. Scoring it anyway puts a meaningless figure in a results
    table where a reader will compare it to a real one.

    Dominant script is used rather than the strict label so a Devanagari
    sentence containing one Latin acronym is still scored.
    """
    rp = script_profile(reference)
    hp = script_profile(hypothesis)
    out: Dict[str, Any] = {
        "cer": None,
        "ref_script": script_of(reference),
        "hyp_script": script_of(hypothesis),
        "ref_dominant": rp["dominant"], "hyp_dominant": hp["dominant"],
        "ref_profile": rp, "hyp_profile": hp,
        "measured": False,
        "not_measured": [],
    }
    if rp["letters"] == 0 or hp["letters"] == 0:
        out["not_measured"].append({
            "metric": "cer",
            "reason": f"one side has no letters (ref={rp['letters']}, "
                      f"hyp={hp['letters']}); CER would be degenerate"})
        return out
    if rp["dominant"] != hp["dominant"]:
        out["not_measured"].append({
            "metric": "cer",
            "reason": f"script mismatch: reference is {rp['dominant']} "
                      f"({rp['deva_frac']:.2f} deva / {rp['latin_frac']:.2f} latin), "
                      f"hypothesis is {hp['dominant']} "
                      f"({hp['deva_frac']:.2f} deva / {hp['latin_frac']:.2f} latin). "
                      f"CER across scripts is near-1.0 by construction and "
                      f"carries no information — transliterate to a common "
                      f"script first, then re-score."})
        return out

    jiwer = _require("jiwer")
    out["cer"] = float(jiwer.cer(reference, hypothesis))
    out["measured"] = True
    return out


# ---------------------------------------------------------------------------
# Terminology
# ---------------------------------------------------------------------------

_OUT_TEXT_KEYS = ("final_text", "mt_text", "text")


def _out_text(seg: Any, text_key: Optional[str] = None) -> str:
    if isinstance(seg, dict):
        if text_key and seg.get(text_key):
            return str(seg[text_key])
        for k in _OUT_TEXT_KEYS:
            if seg.get(k):
                return str(seg[k])
        return ""
    return str(getattr(seg, text_key or "final_text", "") or "")


def _spans_of(seg: Any) -> List[dict]:
    if isinstance(seg, dict):
        return list(seg.get("retain_spans") or [])
    return list(getattr(seg, "retain_spans", None) or [])


def _span_forms(span: dict) -> List[str]:
    """Acceptable surface forms for a retained term, in preference order."""
    forms = []
    for k in ("translit", "text"):
        v = span.get(k) if isinstance(span, dict) else getattr(span, k, None)
        if v:
            forms.append(str(v))
    return forms


def term_retention(
    segments: Iterable[Any],
    *,
    text_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Fraction of retained terms that actually survive into the output text.

    A term counts as retained when the output contains EITHER its source form
    (`span["text"]`, left in Latin) OR its transliteration (`span["translit"]`).
    Both are legitimate outcomes of the codemix stage; requiring one specific
    form would score path A against path B's rubric.

    Spans are counted DISTINCT per segment. Matching is substring containment,
    so three occurrences of the same term in one segment cannot be told apart
    and counting them separately would inflate both numerator and denominator
    with the same decision made once.
    """
    n_segs = 0
    n_raw = 0
    n_distinct = 0
    n_matched = 0
    missing: List[Dict[str, Any]] = []
    by_term: Dict[str, Dict[str, Any]] = {}
    n_no_text = 0

    for seg in segments or []:
        n_segs += 1
        spans = _spans_of(seg)
        n_raw += len(spans)
        if not spans:
            continue
        out_text = _norm(_out_text(seg, text_key))
        if not out_text:
            n_no_text += 1
            continue
        seen: Dict[str, dict] = {}
        for sp in spans:
            forms = _span_forms(sp)
            if not forms:
                continue
            key = _norm(forms[-1])  # the source form is the term's identity
            seen.setdefault(key, sp)
        for key, sp in seen.items():
            n_distinct += 1
            hit = next((f for f in _span_forms(sp) if _norm(f) and _norm(f) in out_text),
                       None)
            rec = by_term.setdefault(
                key, {"term": key, "occurrences": 0, "matched": 0, "renderings": []})
            rec["occurrences"] += 1
            if hit is not None:
                n_matched += 1
                rec["matched"] += 1
                if hit not in rec["renderings"]:
                    rec["renderings"].append(hit)
            else:
                missing.append({
                    "segment": seg.get("id") if isinstance(seg, dict) else None,
                    "term": key,
                    "forms_tried": _span_forms(sp),
                })

    out: Dict[str, Any] = {
        "n_segments": n_segs,
        "n_spans_raw": n_raw,
        "n_spans_distinct": n_distinct,
        "n_matched": n_matched,
        "retention": (n_matched / n_distinct) if n_distinct else None,
        "n_missing": len(missing),
        "missing": missing[:50],
        "by_term": sorted(by_term.values(),
                          key=lambda r: (-r["occurrences"], r["term"])),
        "not_measured": [],
    }
    if n_distinct == 0:
        out["not_measured"].append({
            "metric": "term_retention",
            "reason": f"no retain_spans on any of {n_segs} segments; "
                      f"nothing to retain"})
    if n_no_text:
        out["not_measured"].append({
            "metric": "term_retention.segments",
            "reason": f"{n_no_text} segments had retain_spans but no output "
                      f"text and were excluded"})
    return out


def term_consistency(
    segments: Iterable[Any],
    *,
    text_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Is the same source term rendered the same way every time it appears?

    Terms occurring once are excluded — they are trivially consistent and
    would pad the score toward its ceiling. A rendering is the literal
    surface form found in the output, so a term translated once and
    transliterated once counts as two renderings.

    THREE numbers are returned, and only one of them is safely comparable:

    * `consistency` = `1 - unique_renderings / occurrences`. This is the
      formula the eval spec names, so the key is kept — but its ceiling is
      NOT 1.0. A perfectly consistent corpus scores `1 - n_terms /
      occurrences`, so a term appearing exactly twice and rendered
      identically both times scores 0.5, and two systems with different
      term-frequency distributions get different ceilings. Measured on
      `2vxQaT0Fy-c_5min.bodhan`: consistency 0.643 with ZERO inconsistent
      terms, purely because 5 terms share 14 occurrences.
    * `consistency_ceiling` — what perfect consistency would score here.
      Quote `consistency` only next to this.
    * `consistency_normalized` = `(occurrences - unique) / (occurrences -
      n_terms)`. 1.0 = every term rendered one way, 0.0 = every occurrence
      rendered differently. This is the one to rank systems on.

    `n_inconsistent_terms` is the headline for a human reader: how many
    terms were rendered more than one way at all.
    """
    occ: Dict[str, List[str]] = {}
    unresolved = 0
    for seg in segments or []:
        spans = _spans_of(seg)
        if not spans:
            continue
        raw_out = _out_text(seg, text_key)
        out_text = _norm(raw_out)
        if not out_text:
            continue
        for sp in spans:
            forms = _span_forms(sp)
            if not forms:
                continue
            key = _norm(forms[-1])
            hit = next((f for f in forms if _norm(f) and _norm(f) in out_text), None)
            if hit is None:
                # Dropped entirely: that is a retention failure, reported by
                # term_retention. Counting it as a "rendering" here would
                # double-penalise the same event in a different metric.
                unresolved += 1
                continue
            occ.setdefault(key, []).append(_norm(hit))

    multi = {k: v for k, v in occ.items() if len(v) >= 2}
    total_occ = sum(len(v) for v in multi.values())
    total_uniq = sum(len(set(v)) for v in multi.values())

    per_term = [
        {"term": k, "occurrences": len(v), "unique_renderings": len(set(v)),
         "renderings": sorted(set(v)),
         "consistency": 1.0 - len(set(v)) / len(v),
         # (occ - uniq) / (occ - 1): 1.0 iff all renderings agree.
         "consistency_normalized": ((len(v) - len(set(v))) / (len(v) - 1)
                                    if len(v) > 1 else None)}
        for k, v in sorted(multi.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]

    denom = total_occ - len(multi)
    out: Dict[str, Any] = {
        "n_terms_total": len(occ),
        "n_terms_multi": len(multi),
        "occurrences": total_occ,
        "unique_renderings": total_uniq,
        "consistency": (1.0 - total_uniq / total_occ) if total_occ else None,
        "consistency_ceiling": ((1.0 - len(multi) / total_occ)
                                if total_occ else None),
        "consistency_normalized": (((total_occ - total_uniq) / denom)
                                   if denom > 0 else None),
        "mean_per_term_consistency": (float(np.mean([t["consistency"]
                                                     for t in per_term]))
                                      if per_term else None),
        "mean_per_term_consistency_normalized": (
            float(np.mean([t["consistency_normalized"] for t in per_term]))
            if per_term else None),
        "n_inconsistent_terms": sum(1 for t in per_term
                                    if t["unique_renderings"] > 1),
        "per_term": per_term,
        "n_unresolved_spans": unresolved,
        "not_measured": [],
    }
    if denom <= 0 and total_occ:
        out["not_measured"].append({
            "metric": "consistency_normalized",
            "reason": "every multi-occurrence term occurs exactly once after "
                      "filtering; the normalised scale is undefined"})
    if not multi:
        out["not_measured"].append({
            "metric": "term_consistency",
            "reason": f"no term occurred twice or more "
                      f"({len(occ)} distinct terms found); consistency is "
                      f"undefined on single occurrences"})
    return out


# ---------------------------------------------------------------------------
# chrF++
# ---------------------------------------------------------------------------

def chrfpp_report(refs: List[str], hyps: List[str]) -> Dict[str, Any]:
    """Segment-level and corpus chrF++, via `backend.transcripts.chrf_pp_segments`.

    Median is reported alongside the mean because the segment distribution is
    long-tailed: a handful of untranslated or hallucinated segments drag the
    mean well below where most of the output sits, and the corpus score (which
    is NOT the mean — sacrebleu re-aggregates the n-gram counts) is different
    again. All three are reported so none can be quoted as the other.
    """
    refs = list(refs or [])
    hyps = list(hyps or [])
    out: Dict[str, Any] = {
        "n": len(refs),
        "corpus": None, "median": None, "mean": None,
        "p10": None, "p90": None, "min": None, "max": None,
        "segments": [], "signature": None,
        "not_measured": [],
    }
    if not refs or not hyps:
        out["not_measured"].append({
            "metric": "chrfpp_report",
            "reason": f"empty input (refs={len(refs)}, hyps={len(hyps)})"})
        return out
    if len(refs) != len(hyps):
        out["not_measured"].append({
            "metric": "chrfpp_report",
            "reason": f"length mismatch: {len(refs)} refs vs {len(hyps)} hyps"})
        return out

    from backend.transcripts import chrf_pp_segments
    r = chrf_pp_segments(refs, hyps)
    s = np.asarray(r["segments"], dtype=float)
    out.update({
        "corpus": r["corpus"],
        "signature": r["signature"],
        "segments": r["segments"],
        "median": float(np.median(s)),
        "mean": float(np.mean(s)),
        "p10": float(np.percentile(s, 10)),
        "p90": float(np.percentile(s, 90)),
        "min": float(s.min()), "max": float(s.max()),
        "n_below_0_2": int(np.count_nonzero(s < 0.2)),
    })
    return out


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

def length_ratio(
    segments: Iterable[Any],
    *,
    source_key: str = "text",
    target_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Target/source expansion per segment, in characters, words and aksharas.

    The akshara ratio is the one that predicts duration: Hindi TTS timing
    tracks syllables, not characters, and Devanagari packs a syllable into
    one to four code points depending on the conjunct, so the character
    ratio systematically over-reads expansion on consonant-heavy output.
    """
    char_r: List[float] = []
    word_r: List[float] = []
    ak_r: List[float] = []
    dur_r: List[float] = []
    n = 0
    n_skipped = 0

    for seg in segments or []:
        n += 1
        src = (seg.get(source_key) if isinstance(seg, dict)
               else getattr(seg, source_key, "")) or ""
        tgt = _out_text(seg, target_key)
        if not str(src).strip() or not str(tgt).strip():
            n_skipped += 1
            continue
        src, tgt = str(src), str(tgt)
        char_r.append(len(tgt) / len(src))
        sw = len(src.split()) or 1
        word_r.append(len(tgt.split()) / sw)
        sa = akshara_count(src)
        ta = akshara_count(tgt)
        if sa > 0:
            ak_r.append(ta / sa)
        if isinstance(seg, dict):
            est, targ = seg.get("est_duration_s"), seg.get("target_duration_s")
            if est and targ and float(targ) > 0:
                dur_r.append(float(est) / float(targ))

    def block(v: List[float]) -> Dict[str, Any]:
        if not v:
            return {"n": 0, "median": None, "mean": None, "p95": None,
                    "min": None, "max": None}
        a = np.asarray(v, dtype=float)
        return {"n": int(a.size), "median": float(np.median(a)),
                "mean": float(np.mean(a)), "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max())}

    out: Dict[str, Any] = {
        "n_segments": n,
        "n_skipped": n_skipped,
        "char_ratio": block(char_r),
        "word_ratio": block(word_r),
        "akshara_ratio": block(ak_r),
        "est_over_target_duration": block(dur_r),
        "not_measured": [],
    }
    if n_skipped:
        out["not_measured"].append({
            "metric": "length_ratio",
            "reason": f"{n_skipped}/{n} segments had an empty source or "
                      f"target text and were excluded"})
    if not ak_r:
        out["not_measured"].append({
            "metric": "akshara_ratio",
            "reason": "no segment had a non-zero source akshara count "
                      "(source is probably not Devanagari — compare the "
                      "word ratio instead)"})
    if not dur_r:
        out["not_measured"].append({
            "metric": "est_over_target_duration",
            "reason": "segments carry no est_duration_s/target_duration_s "
                      "(run the duration stage first)"})
    return out


# ---------------------------------------------------------------------------
# Round-trip MT
# ---------------------------------------------------------------------------

_MT_CACHE: Dict[Tuple[str, str], Any] = {}


def roundtrip_mt_wer(
    segments: Iterable[Any],
    *,
    model: str = "facebook/nllb-200-distilled-600M",
    source_language: str = "hi",
    target_language: str = "en",
    source_key: str = "text",
    target_key: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Back-translate the dub text to English and score it against the source.

    A weak proxy for adequacy, and labelled as such: round-trip error folds
    the back-translator's own mistakes into the score, so it ranks systems
    far better than it grades one. Its real use is catching catastrophic
    segments — dropped clauses, hallucinated sentences — which show up as a
    per-segment WER near or above 1.0 while chrF++ stays respectable.
    """
    segs = [s for s in (segments or [])]
    srcs: List[str] = []
    hyps_in: List[str] = []
    ids: List[Any] = []
    for seg in segs:
        src = (seg.get(source_key) if isinstance(seg, dict)
               else getattr(seg, source_key, "")) or ""
        tgt = _out_text(seg, target_key)
        if not str(src).strip() or not str(tgt).strip():
            continue
        ids.append(seg.get("id") if isinstance(seg, dict) else None)
        srcs.append(str(src))
        hyps_in.append(str(tgt))

    out: Dict[str, Any] = {
        "model": model,
        "n_segments": len(segs),
        "n_scored": 0,
        "wer": None, "cer": None,
        "segment_wer": [], "median_segment_wer": None, "p95_segment_wer": None,
        "n_catastrophic": None,
        "back_translations": [],
        "caveat": "round-trip error includes the back-translator's own errors; "
                  "use for ranking and for catching catastrophic segments, "
                  "not as an absolute adequacy score",
        "not_measured": [],
    }
    if not srcs:
        out["not_measured"].append({
            "metric": "roundtrip_mt_wer",
            "reason": f"no segment had both a source ({source_key}) and a "
                      f"non-empty target text (of {len(segs)})"})
        return out

    try:
        from models.mt import get_mt
        from models.mt.base import MTRequest
        key = (model, device or "auto")
        if key not in _MT_CACHE:
            log.info("roundtrip_mt_wer: loading %s", model)
            _MT_CACHE[key] = get_mt("nllb", device=device,
                                    model_id=model, batch_size=batch_size)
        mt = _MT_CACHE[key]
        res = mt.translate(
            [MTRequest(id=str(i), text=t) for i, t in enumerate(hyps_in)],
            target_language=target_language,
            source_language=source_language,
        )
        back = [s.text for s in res.segments]
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "roundtrip_mt_wer",
            "reason": f"back-translation failed ({type(e).__name__}): {e}"})
        return out

    if len(back) != len(srcs):
        out["not_measured"].append({
            "metric": "roundtrip_mt_wer",
            "reason": f"back-translator returned {len(back)} segments for "
                      f"{len(srcs)} inputs; cannot align"})
        return out

    try:
        jiwer = _require("jiwer")
        out["wer"] = float(jiwer.wer(srcs, back))
        out["cer"] = float(jiwer.cer(srcs, back))
        per = [float(jiwer.wer(r, h)) for r, h in zip(srcs, back)]
    except Exception as e:  # noqa: BLE001
        out["not_measured"].append({
            "metric": "roundtrip_mt_wer", "reason": f"jiwer scoring failed: {e}"})
        return out

    a = np.asarray(per, dtype=float)
    out["n_scored"] = len(per)
    out["segment_wer"] = per
    out["median_segment_wer"] = float(np.median(a))
    out["p95_segment_wer"] = float(np.percentile(a, 95))
    out["n_catastrophic"] = int(np.count_nonzero(a >= 0.9))
    out["back_translations"] = [
        {"id": i, "source": s, "back": b, "wer": w}
        for i, s, b, w in zip(ids, srcs, back, per)
    ][:50]
    return out
