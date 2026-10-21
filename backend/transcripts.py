"""Ground-truth transcript parsing + ASR/MT evaluation.

Parses the `.vtt` / `.srt` file the downloader fetched alongside the video
and scores the ASR / MT output against it. The scoring itself is delegated
to the canonical reference implementations — we never roll our own metric:

    - WER / CER       : jiwer (industry standard; same tool pyannote-metrics,
                        ESPnet, whisper-benchmark use)
    - BLEU / chrF     : sacrebleu (the reference BLEU; what WMT publishes)
    - ROUGE-1/2/L     : rouge-score (Google's canonical implementation)
    - METEOR          : nltk.translate.meteor_score (reference since 2005)

Each library is imported lazily so a user who only wants ASR metrics does
not have to install the MT ones (and vice versa) — a missing dependency
raises a targeted `ImportError` naming the pip package.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    start: float   # seconds from media start
    end: float
    text: str


_TIMESTAMP_VTT = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)
_TIMESTAMP_VTT_SHORT = re.compile(
    r"(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
# Inline VTT styling: <00:00:01.000><c> word</c>, <b>, <i>, etc.
_VTT_INLINE_TAG = re.compile(r"<[^>]+>")
# Just the inline cue timestamps, which carry per-word timing in YouTube
# auto-captions. Kept separate from `_VTT_INLINE_TAG` so `parse_vtt_words`
# can read them while `parse_vtt` keeps stripping them.
_VTT_INLINE_TS = re.compile(
    r"<(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})>"
)


def _tc_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_vtt(path: Path) -> List[Cue]:
    """Parse WebVTT (yt-dlp's default YouTube subtitle format)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cues: List[Cue] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _TIMESTAMP_VTT.search(line)
        if m:
            start = _tc_to_seconds(*m.group(1, 2, 3, 4))
            end = _tc_to_seconds(*m.group(5, 6, 7, 8))
        else:
            m2 = _TIMESTAMP_VTT_SHORT.search(line)
            if m2:
                g = m2.groups()
                start = _tc_to_seconds("0", g[0], g[1], g[2])
                end = _tc_to_seconds("0", g[3], g[4], g[5])
            else:
                i += 1
                continue
        i += 1
        buf: List[str] = []
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i])
            i += 1
        raw = " ".join(buf).strip()
        clean = _VTT_INLINE_TAG.sub("", raw).strip()
        if clean:
            cues.append(Cue(start=start, end=end, text=clean))
    return _dedupe_rolling_cues(cues)


def parse_srt(path: Path) -> List[Cue]:
    """Parse SubRip (local sidecar files, downstream editors)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    cues: List[Cue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        ts_line = lines[1] if _TIMESTAMP_VTT.search(lines[1] if len(lines) > 1 else "") else lines[0]
        m = _TIMESTAMP_VTT.search(ts_line)
        if not m:
            continue
        start = _tc_to_seconds(*m.group(1, 2, 3, 4))
        end = _tc_to_seconds(*m.group(5, 6, 7, 8))
        body_start = lines.index(ts_line) + 1
        clean = _VTT_INLINE_TAG.sub("", " ".join(lines[body_start:])).strip()
        if clean:
            cues.append(Cue(start=start, end=end, text=clean))
    return cues


def _dedupe_rolling_cues(cues: List[Cue]) -> List[Cue]:
    """YouTube auto-captions repeat each line across adjacent cues as new
    words land. The last cue in a run contains the full sentence, so earlier
    cues whose text is a prefix of the next cue's text get dropped."""
    out: List[Cue] = []
    for c in cues:
        if out and c.text.startswith(out[-1].text) and c.text != out[-1].text:
            out[-1] = c
        elif out and out[-1].text == c.text:
            out[-1] = Cue(out[-1].start, c.end, c.text)
        else:
            out.append(c)
    return out


# ---- word-level parsing (YouTube auto-caption inline timings) -------------
#
# `_dedupe_rolling_cues` cannot actually undo rolling captions: the prefix rule
# only ever merges the first two cues of a run, so the carried-over line is
# re-emitted as a fresh cue and the flattened text comes out ~2x too long.
# The inline `<00:00:00.378><c> word</c>` stream is immune to that — every word
# appears exactly once, in the cue that first introduced it — and it carries
# per-word timing, which is the most precise alignment data on disk.

@dataclass
class TimedWord:
    text: str
    start: float   # seconds from media start
    end: float


def _iter_vtt_blocks(path: Path):
    """Yield `(start, end, body_lines)` for each cue, text untouched."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _TIMESTAMP_VTT.search(line)
        if m:
            start = _tc_to_seconds(*m.group(1, 2, 3, 4))
            end = _tc_to_seconds(*m.group(5, 6, 7, 8))
        else:
            m2 = _TIMESTAMP_VTT_SHORT.search(line)
            if not m2:
                i += 1
                continue
            g = m2.groups()
            start = _tc_to_seconds("0", g[0], g[1], g[2])
            end = _tc_to_seconds("0", g[3], g[4], g[5])
        i += 1
        buf: List[str] = []
        # A cue payload ends at a genuinely empty line. yt-dlp writes a line
        # holding a single space as real payload, and treating that as the
        # terminator (as `parse_vtt` does) drops the whole first cue.
        while i < len(lines) and lines[i].rstrip("\r") != "":
            buf.append(lines[i])
            i += 1
        yield start, end, buf


def has_inline_word_timings(path: Path) -> bool:
    """True when the file carries `<hh:mm:ss.mmm>` inline cue timestamps.

    Manual/publisher subtitles never do, so callers must check this before
    trusting `parse_vtt_words` to cover the whole file.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return _VTT_INLINE_TS.search(text) is not None


def _inline_pieces(raw: str, cue_start: float, cue_end: float):
    """Split one inline-timed caption line into `(start, end, text)` runs."""
    marks: List[tuple] = []          # (timestamp, text-slice)
    pos, cur = 0, cue_start
    for m in _VTT_INLINE_TS.finditer(raw):
        marks.append((cur, raw[pos:m.start()]))
        cur = _tc_to_seconds(*m.group(1, 2, 3, 4))
        pos = m.end()
    marks.append((cur, raw[pos:]))
    for idx, (t, chunk) in enumerate(marks):
        nxt = marks[idx + 1][0] if idx + 1 < len(marks) else cue_end
        yield t, max(nxt, t), chunk


def parse_vtt_words(path: Path) -> List[TimedWord]:
    """Read the inline `<ts><c>word</c>` stream as deduped, timed words.

    Returns `[]` for files without inline timings (manual subtitles) rather
    than silently falling back to cue text — the caller has to know which
    kind of timing it got. A word's `end` is the next word's start, clamped
    to the cue end for the last word in a cue. One timestamp occasionally
    introduces several tokens (`&gt;&gt; Like<...>`); those share the run and
    are spread evenly across it so starts stay strictly ordered.

    A cue carrying no inline timestamp at all is not automatically a repeat:
    yt-dlp omits the `<ts>` marker when the newly scrolled-in text begins
    exactly at the cue start, so such a cue can introduce real speech the
    inline stream never carries (60 words in a 36-minute English file, 810
    in its Hindi dub). Such cues are diffed against what has already been
    emitted and only the genuinely new tail is kept, timed across the cue
    span. The diff runs on the whitespace-free character stream, not on
    tokens: the inline stream splits punctuation off as its own token
    (`देंगे` + `।`) while the rolling carry-over line renders it attached
    (`देंगे।`), so token equality would miss the overlap and re-emit the
    whole line.
    """
    import html

    def _tokens(s: str) -> List[str]:
        return html.unescape(_VTT_INLINE_TAG.sub("", s)).split()

    def _emit(toks: List[str], t0: float, t1: float) -> None:
        if t1 < t0:
            t1 = t0
        step = (t1 - t0) / len(toks)
        for k, tok in enumerate(toks):
            words.append(TimedWord(
                text=tok,
                start=t0 + k * step,
                end=t0 + (k + 1) * step if k + 1 < len(toks) else t1,
            ))
            seen.append(tok)

    # Manual/publisher subtitles have no inline timings anywhere; recovering
    # their cue text here would hand the caller cue-level guesses dressed up
    # as word timings. `[]` keeps that distinction honest.
    recover_untimed = has_inline_word_timings(path)

    words: List[TimedWord] = []
    seen: List[str] = []          # words[i].text, kept in step for the diff
    for cue_start, cue_end, body in _iter_vtt_blocks(path):
        timed = [raw for raw in body if _VTT_INLINE_TS.search(raw)]
        if timed:
            for raw in timed:
                # Untimed lines in a cue that *does* carry timings are the
                # rolling carry-over of an earlier cue; already emitted.
                for t0, t1, chunk in _inline_pieces(raw, cue_start, cue_end):
                    toks = _tokens(chunk)
                    if toks:
                        _emit(toks, t0, t1)
            continue
        if not recover_untimed:
            continue

        toks = _tokens(" ".join(body))
        if not toks:
            continue
        # Longest prefix of this cue (in characters) that is already the tail
        # of the emitted stream is the carry-over; the rest is new speech.
        tail = "".join(seen)[-200:]
        cue_chars = "".join(toks)
        overlap = 0
        for n in range(min(len(cue_chars), len(tail)), 0, -1):
            if cue_chars[:n] == tail[-n:]:
                overlap = n
                break
        if overlap >= len(cue_chars):
            continue
        new: List[str] = []
        pos = 0
        for tok in toks:
            nxt = pos + len(tok)
            if nxt > overlap:
                # A token straddling the boundary keeps only its new suffix.
                new.append(tok[max(0, overlap - pos):])
            pos = nxt
        if not new:
            continue
        t0 = max(cue_start, words[-1].end) if words else cue_start
        _emit(new, t0, max(cue_end, t0))

    out: List[TimedWord] = []
    for w in words:
        if out:
            prev = out[-1]
            # A word that jumps backwards is a repeat carried over from an
            # earlier cue, not new speech.
            if w.start < prev.start - 1e-6:
                continue
            if abs(w.start - prev.start) < 1e-6 and w.text == prev.text:
                continue
            if prev.end > w.start:
                prev.end = w.start
        if w.end < w.start:
            w.end = w.start
        out.append(w)
    return out


def rolling_duplication_ratio(path: Path) -> float:
    """Share of word bigrams that are repeats — a rolling-caption detector.

    YouTube auto-captions re-emit each line as the next cue scrolls in, so a
    naive `parse_vtt` flatten repeats nearly every bigram once (ratio ~0.5).
    Manual subtitles sit near zero. Returns 0.0 for files too short to score.

    Deliberately does NOT use `normalize_text`: stripping punctuation merges
    bigrams that differ only by a comma or danda, which pushes every file's
    ratio up (manual Bengali 0.0625 -> 0.0899, rolling English 0.5742 ->
    0.6111). NFC + casefold + whitespace keeps the two populations further
    apart. (Before `_strip_punct` existed, `normalize_text` also shattered
    Indic words into bare consonants and scored that same Bengali file at
    0.5154 — indistinguishable from a rolling file.)
    """
    flat = unicodedata.normalize("NFC", transcript_text(parse_vtt(path))).casefold()
    toks = flat.split()
    total = len(toks) - 1
    if total < 1:
        return 0.0
    bigrams = [(toks[i], toks[i + 1]) for i in range(total)]
    return (total - len(set(bigrams))) / total


def load_transcript(path: Path) -> List[Cue]:
    """Parse a VTT/SRT, or wrap a plain `.txt` transcript as a single cue.

    `.txt` lets a user score against a hand-cleaned reference that has no
    timing information — handy when the only ground truth is a published
    script, not the broadcast captions.
    """
    ext = Path(path).suffix.lower()
    if ext == ".vtt":
        return parse_vtt(path)
    if ext == ".srt":
        return parse_srt(path)
    if ext in (".txt", ".text", ""):
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
        return [Cue(start=0.0, end=0.0, text=text)] if text else []
    raise ValueError(f"Unsupported subtitle format: {ext} ({path})")


def transcript_text(cues: List[Cue]) -> str:
    """Flatten cues to a single string for corpus-level metrics."""
    return " ".join(c.text for c in cues).strip()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")

# Per-codepoint verdict cache for `_strip_punct` — characters repeat heavily
# inside a transcript, so this makes the character walk effectively a lookup.
_PUNCT_CACHE: Dict[int, str] = {}


def _strip_punct(s: str) -> str:
    """Replace punctuation and symbols with a space, keeping combining marks.

    The obvious `re.sub(r"[^\\w\\s]", " ", s)` is wrong for every Indic
    script: Python's `\\w` is letters + digits + underscore, and *excludes*
    Unicode categories Mn/Mc/Me, so it deletes every matra, nukta, anusvara
    and virama. "गुणवत्ता वाली छवियां" came out as "ग णवत त व ल छव य" — eight
    real words shattered into bare consonants — which silently poisoned every
    WER / CER / BLEU / chrF / ROUGE / METEOR number this module reports for
    Hindi, Bengali, Tamil and the rest. Classify by Unicode category instead
    so marks stay attached to their base character. For Latin / ASCII input
    this is identical to the old regex.
    """
    cache = _PUNCT_CACHE
    out: List[str] = []
    for ch in s:
        o = ord(ch)
        r = cache.get(o)
        if r is None:
            # L* letters, N* numbers, M* combining marks, plus `_` and
            # whitespace are exactly what `\w`/`\s` kept, minus the M* bug.
            keep = unicodedata.category(ch)[0] in "LNM" or ch.isspace() or ch == "_"
            r = cache[o] = ch if keep else " "
        out.append(r)
    return "".join(out)


def normalize_text(s: str) -> str:
    """Casefold, strip punctuation, collapse whitespace, NFC-normalize.

    Keeps the comparison fair across ASR engines that may or may not emit
    punctuation / capitalization. Does NOT strip filler words — those are
    real tokens the ASR either heard or missed.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = _strip_punct(s)
    s = _WS.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Metric wrappers — thin shims around the canonical library implementations
# ---------------------------------------------------------------------------
#
# We delegate to the reference packages (jiwer, sacrebleu, rouge_score, nltk)
# so the numbers we print match what every other benchmark tool prints for
# the same inputs. No arithmetic lives in this module — only the normalisation
# of library return types into a flat `dict` we can serialise to JSON.

def _require(pkg: str, pip_name: Optional[str] = None):
    """Import `pkg` or raise an ImportError that names the pip package."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise ImportError(
            f"Transcript evaluation needs `{pip_name or pkg}`. "
            f"Install it with `pip install {pip_name or pkg}`."
        ) from e


def wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Word error rate via `jiwer.process_words` (returns S/D/I breakdown)."""
    jiwer = _require("jiwer")
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    r = jiwer.process_words(ref, hyp)
    return {
        "wer": r.wer,
        "n_ref_words": len(ref.split()),
        "n_hyp_words": len(hyp.split()),
        "substitutions": r.substitutions,
        "deletions": r.deletions,
        "insertions": r.insertions,
        "hits": r.hits,
    }


def cer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Character error rate via `jiwer.process_characters`."""
    jiwer = _require("jiwer")
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    r = jiwer.process_characters(ref, hyp)
    return {
        "cer": r.cer,
        "n_ref_chars": len(ref.replace(" ", "")),
        "n_hyp_chars": len(hyp.replace(" ", "")),
        "substitutions": r.substitutions,
        "deletions": r.deletions,
        "insertions": r.insertions,
        "hits": r.hits,
    }


def bleu(reference: str, hypothesis: str, *, normalize: bool = True) -> Dict[str, Any]:
    """Corpus BLEU via `sacrebleu.corpus_bleu`.

    With `normalize=True` (the historical default) the text is casefolded and
    stripped of all punctuation — including the Devanagari danda — *before*
    sacrebleu applies its own `13a` tokenizer, so the reported signature does
    not describe the input and the score is not comparable with published
    WMT BLEU. Pass `normalize=False` for the real, citable number.
    """
    sacrebleu = _require("sacrebleu")
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    s = sacrebleu.corpus_bleu([hyp], [[ref]])
    return {
        "bleu": s.score / 100.0,              # sacrebleu reports 0..100
        "precisions": [p / 100.0 for p in s.precisions],
        "bp": s.bp,
        "sys_len": s.sys_len,
        "ref_len": s.ref_len,
        "signature": str(s),                  # includes tokeniser + version
        "normalized": normalize,
    }


def chrf(reference: str, hypothesis: str, *, normalize: bool = True) -> Dict[str, Any]:
    """chrF2 via `sacrebleu.corpus_chrf` — char 6-grams, beta=2, NO word order.

    This is chrF, *not* chrF++: `word_order` stays at sacrebleu's default of 0.
    Use `chrf_pp` for chrF++. As with `bleu`, `normalize=True` mangles the text
    before sacrebleu sees it; pass `normalize=False` for a citable score.
    """
    sacrebleu = _require("sacrebleu")
    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    s = sacrebleu.corpus_chrf([hyp], [[ref]])
    return {
        "chrf": s.score / 100.0,
        "signature": str(s),
        "normalized": normalize,
    }


def chrf_pp(reference: str, hypothesis: str) -> Dict[str, Any]:
    """chrF++ on RAW text: char 6-grams + word 2-grams, beta=2.

    `word_order=2` is what makes it chrF++ rather than chrF2, and the text is
    passed through untouched — chrF++ is defined over the surface string, so
    pre-stripping punctuation (the danda, in Hindi) changes the metric.
    """
    sacrebleu = _require("sacrebleu")
    s = sacrebleu.corpus_chrf(
        [hypothesis], [[reference]],
        word_order=2, char_order=6, beta=2,
    )
    return {
        "chrf_pp": s.score / 100.0,
        "signature": str(s),
        "normalized": False,
    }


def chrf_pp_segments(refs: List[str], hyps: List[str]) -> Dict[str, Any]:
    """Per-segment chrF++ plus the corpus score over the same pairs.

    The corpus score is recomputed by sacrebleu over all segments at once —
    it is not the mean of the per-segment scores, and the two differ.
    """
    if len(refs) != len(hyps):
        raise ValueError(
            f"chrf_pp_segments: {len(refs)} references vs {len(hyps)} hypotheses"
        )
    sacrebleu = _require("sacrebleu")
    metric = sacrebleu.CHRF(word_order=2, char_order=6, beta=2)
    segments = [
        metric.sentence_score(h, [r]).score / 100.0
        for r, h in zip(refs, hyps)
    ]
    corpus = metric.corpus_score(list(hyps), [list(refs)])
    return {
        "segments": segments,
        "corpus": corpus.score / 100.0,
        "n": len(refs),
        "signature": str(corpus),
        "normalized": False,
    }


def _rouge_scorer():
    """Lazily build (and cache) the google rouge-score scorer.

    rouge-score's built-in tokenizer is `re.sub(r"[^a-z0-9]+", " ", text)`,
    i.e. ASCII-only: it deletes every Devanagari, Bengali and Tamil
    character outright, so ROUGE over an Indic pair was scored on nothing
    but the embedded English loanwords and digits (and returned a flat 0.0
    for text with none). We hand it a Unicode-aware tokenizer built on
    `normalize_text` instead. On ASCII input the two agree exactly — the
    default also folds case, splits on `_` and drops punctuation.
    """
    global _ROUGE
    try:
        return _ROUGE
    except NameError:
        pass
    rs = _require("rouge_score", pip_name="rouge-score")
    from rouge_score import rouge_scorer, tokenizers   # noqa: WPS433

    class _UnicodeTokenizer(tokenizers.Tokenizer):
        def tokenize(self, text):
            return normalize_text(text).replace("_", " ").split()

    globals()["_ROUGE"] = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False,
        tokenizer=_UnicodeTokenizer(),
    )
    return globals()["_ROUGE"]


def _rouge_pack(score) -> Dict[str, float]:
    return {"f1": score.fmeasure, "precision": score.precision, "recall": score.recall}


def rouge_n(reference: str, hypothesis: str, n: int = 1) -> Dict[str, Any]:
    """ROUGE-{1,2} F1/precision/recall via Google's `rouge-score`."""
    key = f"rouge{n}"
    if key not in ("rouge1", "rouge2"):
        raise ValueError(f"rouge_n: only n=1,2 supported, got n={n}")
    scores = _rouge_scorer().score(normalize_text(reference),
                                   normalize_text(hypothesis))
    return _rouge_pack(scores[key])


def rouge_l(reference: str, hypothesis: str) -> Dict[str, Any]:
    """ROUGE-L (LCS-based F1) via Google's `rouge-score`."""
    scores = _rouge_scorer().score(normalize_text(reference),
                                   normalize_text(hypothesis))
    return _rouge_pack(scores["rougeL"])


def _ensure_nltk_data() -> None:
    """Make sure NLTK's meteor-time resources (wordnet, omw) are present."""
    nltk = _require("nltk")
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def meteor(reference: str, hypothesis: str) -> Dict[str, Any]:
    """METEOR via `nltk.translate.meteor_score.single_meteor_score`.

    Uses NLTK's default alignment stages (exact → stem → synonym via WordNet),
    which is the reference implementation. First call auto-downloads the
    WordNet / OMW corpora NLTK needs — a few MB, cached under NLTK_DATA.
    """
    _ensure_nltk_data()
    from nltk.translate.meteor_score import single_meteor_score  # noqa: WPS433
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return {"meteor": 0.0, "n_ref": len(ref_tokens), "n_hyp": len(hyp_tokens)}
    score = single_meteor_score(ref_tokens, hyp_tokens)
    return {"meteor": float(score),
            "n_ref": len(ref_tokens), "n_hyp": len(hyp_tokens)}


# ---------------------------------------------------------------------------
# ASR / MT report builders
# ---------------------------------------------------------------------------

def _read_asr_text(asr_json: Path) -> str:
    """Flatten `asr.json` segments to a single string for metric scoring."""
    payload = json.loads(Path(asr_json).read_text(encoding="utf-8"))
    segs = payload.get("segments") or []
    return " ".join((s.get("text") or "").strip() for s in segs).strip()


def _read_mt_text(mt_json: Path) -> str:
    """Flatten the MT segments in a merged ASR+MT JSON file."""
    payload = json.loads(Path(mt_json).read_text(encoding="utf-8"))
    segs = payload.get("segments") or []
    out: List[str] = []
    for s in segs:
        # Mirror what tts.py / duration.py actually speak: duration-aware
        # translation rewrites only `final_text`, so scoring `mt_text` would
        # measure a string that never reached the dub.
        t = s.get("final_text") or s.get("mt_text") or ""
        if t:
            out.append(t.strip())
    return " ".join(out).strip()


def _all_text_metrics(ref: str, hyp: str) -> Dict[str, Any]:
    """Run every metric we have on one (ref, hyp) text pair."""
    return {
        "bleu": bleu(ref, hyp),
        "chrf": chrf(ref, hyp),
        "chrf_pp": chrf_pp(ref, hyp),         # raw text, word_order=2
        "rouge1": rouge_n(ref, hyp, 1),
        "rouge2": rouge_n(ref, hyp, 2),
        "rougeL": rouge_l(ref, hyp),
        "meteor": meteor(ref, hyp),
    }


def evaluate_asr(
    asr_json: Path,
    transcript_path: Path,
    *,
    language: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Score an ASR JSON against a ground-truth transcript (VTT/SRT/TXT).

    Reports WER + CER (canonical ASR metrics) plus BLEU / ROUGE / METEOR so
    the same number set is directly comparable with the MT report. Those
    overlap scores penalise swapped word order that WER already catches, but
    they stay useful when the ASR output is paraphrastic (e.g. Whisper's
    punctuation-rewritten long-form transcripts).
    """
    cues = load_transcript(transcript_path)
    ref = transcript_text(cues)
    hyp = _read_asr_text(asr_json)
    report: Dict[str, Any] = {
        "transcript": str(Path(transcript_path).resolve()),
        "transcript_source": source,
        "transcript_cues": len(cues),
        "language": language,
        "asr_json": str(Path(asr_json).resolve()),
        "wer": wer(ref, hyp),
        "cer": cer(ref, hyp),
    }
    report.update(_all_text_metrics(ref, hyp))
    return report


def evaluate_mt(
    mt_json: Path,
    transcript_path: Path,
    *,
    target_language: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Score a merged MT JSON against a dubbed-language transcript.

    Reports BLEU / chrF / ROUGE-{1,2,L} / METEOR. BLEU alone is a weak signal
    for Indic targets (sparse n-gram overlap under morphology); chrF and
    METEOR give a more honest picture, and ROUGE-L exposes long-range order.
    """
    cues = load_transcript(transcript_path)
    ref = transcript_text(cues)
    hyp = _read_mt_text(mt_json)
    report: Dict[str, Any] = {
        "transcript": str(Path(transcript_path).resolve()),
        "transcript_source": source,
        "transcript_cues": len(cues),
        "target_language": target_language,
        "mt_json": str(Path(mt_json).resolve()),
    }
    report.update(_all_text_metrics(ref, hyp))
    return report


def write_report(report: Dict[str, Any], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return out_path


# ---------------------------------------------------------------------------
# CLI — `python -m backend.transcripts test ...`
# ---------------------------------------------------------------------------

def _print_overlap_metrics(r: Dict[str, Any]) -> None:
    """Print the shared metric block (BLEU, chrF, ROUGE, METEOR)."""
    b = r["bleu"]; ch = r["chrf"]
    r1 = r["rouge1"]; r2 = r["rouge2"]; rl = r["rougeL"]; me = r["meteor"]
    print(f"BLEU       : {b['bleu']:.4f}  "
          f"(precisions={[round(p,3) for p in b['precisions']]}, "
          f"BP={b['bp']:.3f}, sys_len={b['sys_len']}, ref_len={b['ref_len']})")
    print(f"chrF2      : {ch['chrf']:.4f}  (normalized input)")
    cpp = r.get("chrf_pp")
    if cpp:
        print(f"chrF++     : {cpp['chrf_pp']:.4f}  (raw text, word_order=2)")
    print(f"ROUGE-1 F1 : {r1['f1']:.4f}  "
          f"(P={r1['precision']:.3f}, R={r1['recall']:.3f})")
    print(f"ROUGE-2 F1 : {r2['f1']:.4f}  "
          f"(P={r2['precision']:.3f}, R={r2['recall']:.3f})")
    print(f"ROUGE-L F1 : {rl['f1']:.4f}  "
          f"(P={rl['precision']:.3f}, R={rl['recall']:.3f})")
    print(f"METEOR     : {me['meteor']:.4f}  "
          f"(ref_tokens={me['n_ref']}, hyp_tokens={me['n_hyp']})")


def _print_asr_report(r: Dict[str, Any]) -> None:
    w, c = r["wer"], r["cer"]
    print(f"transcript : {r['transcript']}")
    print(f"source     : {r.get('transcript_source')} ({r.get('language') or 'n/a'})")
    print(f"cues       : {r['transcript_cues']}")
    print(f"WER        : {w['wer']:.4f}  "
          f"(ref={w['n_ref_words']}, hyp={w['n_hyp_words']}, "
          f"S={w['substitutions']}, D={w['deletions']}, I={w['insertions']})")
    print(f"CER        : {c['cer']:.4f}  "
          f"(ref={c['n_ref_chars']}, hyp={c['n_hyp_chars']})")
    _print_overlap_metrics(r)


def _print_mt_report(r: Dict[str, Any]) -> None:
    print(f"transcript : {r['transcript']}")
    print(f"source     : {r.get('transcript_source')} ({r.get('target_language') or 'n/a'})")
    print(f"cues       : {r['transcript_cues']}")
    _print_overlap_metrics(r)


def main(argv: Optional[List[str]] = None) -> int:
    """Score existing pipeline outputs against a transcript — no re-run needed.

    Examples:
        # ASR WER/CER against publisher captions
        python -m backend.transcripts test \
            --asr-json data/asr/VID.bodhan.asr.json \
            --transcript data/videos/VID.en.vtt \
            --language en

        # MT BLEU/chrF against a dubbed-language caption file
        python -m backend.transcripts test \
            --mt-json data/mt/VID.bodhan.hi.mt.json \
            --transcript data/videos/VID.hi.vtt \
            --target-language hi

        # Both at once; writes JSON reports next to --out-dir (default: cwd)
        python -m backend.transcripts test \
            --asr-json ... --mt-json ... --transcript ... --out-dir data/eval
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m backend.transcripts",
        description="Score ASR/MT outputs against an existing transcript "
                    "(VTT, SRT, or plain .txt).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("test", help="Score ASR and/or MT output against a transcript.")
    t.add_argument("--asr-json", type=Path, default=None,
                   help="ASR JSON produced by the pipeline (data/asr/...). "
                        "Scored with WER + CER.")
    t.add_argument("--mt-json", type=Path, default=None,
                   help="Merged MT JSON produced by the pipeline (data/mt/...). "
                        "Scored with BLEU + chrF.")
    t.add_argument("--transcript", type=Path, default=None,
                   help="Reference transcript — .vtt / .srt / .txt. Shared "
                        "default for ASR and MT; override per-mode with "
                        "--asr-transcript / --mt-transcript when the source "
                        "and dub are in different files.")
    t.add_argument("--asr-transcript", type=Path, default=None,
                   help="Override reference for ASR only (source-language sub).")
    t.add_argument("--mt-transcript", type=Path, default=None,
                   help="Override reference for MT only (dubbed-language sub).")
    t.add_argument("--language", default=None,
                   help="Reported ASR language label (e.g. en).")
    t.add_argument("--target-language", default=None,
                   help="Reported MT target-language label (e.g. hi).")
    t.add_argument("--source", default=None,
                   help="Reported transcript provenance "
                        "(manual | auto | local). Metadata only.")
    t.add_argument("--out-dir", type=Path, default=None,
                   help="Write <stem>.asr_eval.json / .mt_eval.json here. "
                        "Omit to only print to stdout.")
    t.add_argument("--json", action="store_true",
                   help="Print the full report as JSON instead of a summary.")

    args = ap.parse_args(argv)
    if not args.asr_json and not args.mt_json:
        ap.error("pass --asr-json and/or --mt-json")

    reports: Dict[str, Dict[str, Any]] = {}

    if args.asr_json:
        ref = args.asr_transcript or args.transcript
        if not ref or not Path(ref).exists():
            ap.error(f"ASR transcript not found: {ref}")
        print("=== ASR vs transcript ===")
        rep = evaluate_asr(
            args.asr_json, ref,
            language=args.language, source=args.source,
        )
        reports["asr"] = rep
        if args.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            _print_asr_report(rep)
        if args.out_dir:
            out = Path(args.out_dir) / f"{Path(args.asr_json).stem}.asr_eval.json"
            write_report(rep, out)
            print(f"wrote      : {out}")

    if args.mt_json:
        ref = args.mt_transcript or args.transcript
        if not ref or not Path(ref).exists():
            ap.error(f"MT transcript not found: {ref}")
        print("=== MT vs dubbed transcript ===")
        rep = evaluate_mt(
            args.mt_json, ref,
            target_language=args.target_language, source=args.source,
        )
        reports["mt"] = rep
        if args.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            _print_mt_report(rep)
        if args.out_dir:
            out = Path(args.out_dir) / f"{Path(args.mt_json).stem}.mt_eval.json"
            write_report(rep, out)
            print(f"wrote      : {out}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
