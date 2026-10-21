"""Ground-truth layer for the timing-alignment study.

Everything downstream — duration error, pause placement, lip-sync offset — is
scored against what this module writes. So it does three jobs and nothing else:

  1. `vad`           where speech actually is, from a *named* backend.
  2. `align_words`   word timings, from whisperx forced-align (NOT transcribe).
  3. `reference_qc`  a gate that refuses degenerate references before they are
                     used to "prove" anything.

and `build_reference` cuts one golden clip and writes all of it to disk with
full provenance.

ALL TIMES IN EVERY ARTIFACT THIS MODULE WRITES ARE CLIP-RELATIVE: 0.0 is the
clip's `t0` in the source media, never the source media's own zero. The only
absolute times live in `meta.json` under `source.t0` / `source.t1`. Mixing the
two up is the likeliest bug in anything that consumes these files, so the
schema string inside every JSON repeats it.

House rule, enforced throughout: a metric that could not be computed is
recorded in `errors[]` / `not_measured[]` with a reason. Never omitted, never
guessed.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import statistics
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.media_io import _ffmpeg_bin, probe_duration
from backend.transcripts import (
    Cue,
    _require,
    has_inline_word_timings,
    parse_vtt,
    parse_vtt_words,
    rolling_duplication_ratio,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# pyannote ships no standalone VAD checkpoint on this host; its segmentation
# model is a file inside the cached diarization bundle. Resolved through the
# HF cache (never the network) so a sweep works offline.
_PYANNOTE_BUNDLE = "pyannote/speaker-diarization-community-1"
_PYANNOTE_SEGMENTATION = "segmentation/pytorch_model.bin"

# Does the Hindi subtitle track carry *dub* timing, or is it a translation laid
# on the English timeline? Measured in Phase 1; see
# data/golden_reference_provenance.md (cue-onset-vs-VAD-onset test).
# "source_timeline" means: use the text, NEVER the times.
HI_SUBTITLE_TIMELINE: Dict[str, str] = {
    "6-NRiom8K9Y": "source_timeline",   # 64 cues match EN audio vs 39 HI
    "ZPUtA3W-7_I": "dub",               # 105 vs 88, median offset 97 ms
    "qjPH9njnaVU": "dub",               # 124 vs 115, median offset 74 ms
}

SR = 16_000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float          # clip-relative seconds
    end: float
    src: str              # "whisperx" | "vtt_inline"
    score: float = 0.0


@dataclass
class VadResult:
    speech: List[Tuple[float, float]]
    pauses: List[Tuple[float, float]]
    min_pause_s: float
    # `backend` is NOT cosmetic: pyannote and the energy VAD have different
    # operating points, so a silent fallback on one clip would make that clip
    # incomparable to the rest of the sweep without anyone noticing.
    backend: str = "unknown"
    min_speech_s: float = 0.0
    duration_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        total_speech = sum(e - s for s, e in self.speech)
        return {
            "schema": "golden.vad/1",
            "times": "clip-relative seconds; 0.0 == clip t0",
            "backend": self.backend,
            "min_pause_s": self.min_pause_s,
            "min_speech_s": self.min_speech_s,
            "duration_s": round(self.duration_s, 4),
            "n_speech": len(self.speech),
            "n_pauses": len(self.pauses),
            "speech_s": round(total_speech, 4),
            "speech_frac": round(total_speech / self.duration_s, 4) if self.duration_s else None,
            # Pauses are INTERNAL gaps only (between two speech regions).
            # Leading/trailing silence is reported separately because a clip
            # boundary is not a pause the dub has to reproduce.
            "lead_silence_s": round(self.speech[0][0], 4) if self.speech else None,
            "tail_silence_s": (round(self.duration_s - self.speech[-1][1], 4)
                               if self.speech and self.duration_s else None),
            "speech": [[round(s, 4), round(e, 4)] for s, e in self.speech],
            "pauses": [[round(s, 4), round(e, 4)] for s, e in self.pauses],
            "errors": [self.error] if self.error else [],
        }


@dataclass
class RefQC:
    verdict: str                              # "pass" | "pass_with_warnings" | "reject"
    rules: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    not_measured: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schema"] = "golden.reference_qc/1"
        return d


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def _read_wav(wav: Path) -> Tuple["Any", int]:
    sf = _require("soundfile")
    np = _require("numpy")
    y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y), sr


@lru_cache(maxsize=2)
def _pyannote_vad_pipeline(min_speech_s: float, min_pause_s: float):
    """Build the pyannote VAD pipeline from the cached diarization bundle.

    Raises on any failure; `vad()` turns that into the energy fallback and
    records the reason. Cached because loading the checkpoint costs ~2 s and a
    sweep calls this once per track per clip.
    """
    from huggingface_hub import try_to_load_from_cache

    ckpt = try_to_load_from_cache(_PYANNOTE_BUNDLE, _PYANNOTE_SEGMENTATION)
    if not isinstance(ckpt, str):
        raise RuntimeError(
            f"{_PYANNOTE_BUNDLE}/{_PYANNOTE_SEGMENTATION} is not in the HF cache "
            "(offline, and we never fetch during a scoring run)"
        )
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    model = Model.from_pretrained(ckpt)
    pipe = VoiceActivityDetection(segmentation=model)
    # pyannote's own knobs line up 1:1 with ours: `min_duration_off` fills
    # gaps shorter than it, `min_duration_on` drops speech shorter than it.
    pipe.instantiate({"min_duration_on": min_speech_s, "min_duration_off": min_pause_s})
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception as e:                      # CPU is correct, just slower
        log.warning("vad: could not move pyannote to cuda (%s); staying on cpu", e)
    return pipe


def _vad_pyannote(wav: Path, *, min_pause_s: float, min_speech_s: float
                  ) -> List[Tuple[float, float]]:
    import torch
    y, sr = _read_wav(wav)
    pipe = _pyannote_vad_pipeline(min_speech_s, min_pause_s)
    ann = pipe({"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": sr})
    return [(float(s.start), float(s.end)) for s in ann.get_timeline().support()]


def _vad_energy(wav: Path, *, min_pause_s: float, min_speech_s: float,
                thresh_db: float = -38.0) -> List[Tuple[float, float]]:
    """10 ms frame-energy VAD — the same operating point `scripts/select_clips.py`
    used to pick the clips, so a fallback stays consistent with selection."""
    np = _require("numpy")
    librosa = _require("librosa")
    hop = 160                                   # 10 ms at 16 kHz
    y, sr = _read_wav(wav)
    hop = max(1, int(round(sr * 0.010)))
    rms = librosa.feature.rms(y=y, frame_length=hop * 2 + 80, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    mask = db > thresh_db

    runs: List[List[float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append([i * hop / sr, j * hop / sr])
            i = j
        else:
            i += 1
    merged: List[List[float]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_pause_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_speech_s]


def vad(wav: Path, *, min_pause_s: float = 0.25, min_speech_s: float = 0.20,
        backend: str = "auto") -> VadResult:
    """Speech / internal-pause segmentation of `wav`, clip-relative.

    `backend`: "auto" (pyannote, else energy), "pyannote", or "energy".
    The backend that actually produced the answer is always in the result.
    """
    wav = Path(wav)
    try:
        duration = float(len(_read_wav(wav)[0])) / SR
    except Exception:
        duration = 0.0
    try:
        duration = probe_duration(wav)
    except Exception as e:
        log.warning("vad: probe_duration(%s) failed (%s); using sample count", wav.name, e)

    used, err = backend, None
    speech: List[Tuple[float, float]] = []
    if backend in ("auto", "pyannote"):
        try:
            speech = _vad_pyannote(wav, min_pause_s=min_pause_s, min_speech_s=min_speech_s)
            used = "pyannote"
        except Exception as e:
            if backend == "pyannote":
                return VadResult(speech=[], pauses=[], min_pause_s=min_pause_s,
                                 backend="pyannote", min_speech_s=min_speech_s,
                                 duration_s=duration, error=f"pyannote VAD failed: {e!r}")
            err = f"pyannote unavailable, fell back to energy VAD: {e!r}"
            log.warning("vad: %s", err)
            used = "energy"
    if used == "energy" or backend == "energy":
        try:
            speech = _vad_energy(wav, min_pause_s=min_pause_s, min_speech_s=min_speech_s)
            used = "energy"
        except Exception as e:
            return VadResult(speech=[], pauses=[], min_pause_s=min_pause_s,
                             backend="energy", min_speech_s=min_speech_s,
                             duration_s=duration, error=f"energy VAD failed: {e!r}")

    speech = sorted((float(a), float(b)) for a, b in speech)
    pauses = [(speech[i][1], speech[i + 1][0]) for i in range(len(speech) - 1)
              if speech[i + 1][0] - speech[i][1] >= min_pause_s]
    return VadResult(speech=speech, pauses=pauses, min_pause_s=min_pause_s,
                     backend=used, min_speech_s=min_speech_s, duration_s=duration,
                     error=err)


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _align_model(language: str, device: str):
    whisperx = _require("whisperx")
    return whisperx.load_align_model(language_code=language, device=device)


def align_words(wav: Path, text: str, *, language: str = "hi",
                device: Optional[str] = None) -> List[Word]:
    """whisperx forced-align `text` against `wav`. Never transcribes.

    This is the ONLY route to Hindi word timings in this corpus: no `hi-orig`
    auto-caption exists on any of the four videos, and every manual subtitle
    carries zero inline timestamps.

    The whole clip is passed as a single alignment segment. Per-cue segments
    would align faster and tighter, but only when the cue times are already
    trustworthy — and for `6-NRiom8K9Y` the Hindi cues sit on the *English*
    timeline, so seeding whisperx with them would bake that error straight
    into the ground truth.

    Returns `[]` with a logged reason on any failure. One bad clip must not
    kill a sweep.
    """
    wav = Path(wav)
    text = " ".join((text or "").split())
    if not text:
        log.warning("align_words: empty text for %s; returning []", wav.name)
        return []
    try:
        torch = _require("torch")
        whisperx = _require("whisperx")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        y, sr = _read_wav(wav)
        if sr != SR:
            raise ValueError(f"{wav} is {sr} Hz; whisperx align needs {SR} Hz")
        dur = float(len(y)) / sr
        model, meta = _align_model(language, dev)
        res = whisperx.align(
            [{"text": text, "start": 0.0, "end": dur}],
            model, meta, y, dev, return_char_alignments=False,
        )
    except Exception as e:
        log.warning("align_words: whisperx align failed on %s (%s): %r",
                    wav.name, language, e)
        return []

    out: List[Word] = []
    dropped = 0
    for w in res.get("word_segments", []):
        s, e = w.get("start"), w.get("end")
        if s is None or e is None:
            # whisperx emits untimed entries for tokens the acoustic model has
            # no character for (digits, emoji, foreign script). Dropping them
            # silently would misstate the word count, so it is counted.
            dropped += 1
            continue
        out.append(Word(text=str(w.get("word", "")), start=float(s), end=float(max(e, s)),
                        src="whisperx", score=round(float(w.get("score") or 0.0), 4)))
    if dropped:
        log.info("align_words: %s — %d/%d tokens had no timing and were dropped",
                 wav.name, dropped, dropped + len(out))
    return out


# ---------------------------------------------------------------------------
# Reference QC — the degenerate-reference gate
# ---------------------------------------------------------------------------

# Thresholds live here, named, so a report can cite them.
QC_MIN_CUES_PER_MIN = 4.0
QC_MAX_CUE_SPAN_S = 30.0
QC_COVERAGE_RANGE = (0.50, 1.05)
QC_MIN_TAIL_RATIO = 0.80
QC_WPS_RANGE = (1.5, 6.0)
QC_MAX_ROLLING_RATIO = 0.25
# Fraction of VAD-detected speech that the aligned word stream must span before
# that stream may be used for WORD-LEVEL timing metrics. This is a per-clip
# rule and the file-level cue-density rule does not subsume it: a subtitle can
# pass `cues/minute >= 4` comfortably while being far too sparse over the
# particular window a clip was cut from. Measured on this corpus, healthy clips
# land at 0.66-0.80; the four `6-NRiom8K9Y` clips cut from the under-subtitled
# first 5.5 minutes land at 0.22-0.52, because that track carries 7-11 cues per
# window there against 41-44 later. whisperx aligned the text it was given
# faithfully -- the text was simply incomplete, which is invisible to any
# check that looks only at the subtitle file as a whole.
QC_MIN_WORD_VAD_COVERAGE = 0.55
# The actual gate. Span-coverage turned out to be a poor discriminator: word
# spans exclude the inter-word micro-gaps that VAD counts, so healthy ENGLISH
# source tracks sit at 0.53-0.58, overlapping the broken range and producing
# false rejects. Word RATE separates the two populations cleanly. Measured over
# all 18 clips of this corpus:
#     healthy dub (Modi/Durov/dense NPTEL)  3.13 - 5.15 words/s
#     healthy source (English)              2.51 - 2.70 words/s
#     under-subtitled NPTEL windows         0.96 - 1.45 words/s
# 2.0 sits in the empty gap between 1.45 and 2.51. Coverage is still recorded
# as a diagnostic, because it is what makes the failure legible once the rate
# rule has flagged it.
QC_MIN_WORDS_PER_SPEECH_S = 2.0


def _media_duration(media: Path) -> Tuple[Optional[float], str, Optional[str]]:
    """(seconds, how, error). soundfile first, ffprobe second.

    soundfile cannot open an mp4 at all, so "soundfile raised" is recorded as a
    measured property of the file rather than treated as unreadable media — a
    reference is only unreadable when *no* route can read it.
    """
    sf_err: Optional[str] = None
    try:
        sf = _require("soundfile")
        return float(sf.info(str(media)).duration), "soundfile", None
    except Exception as e:
        sf_err = f"{type(e).__name__}: {e}"
    try:
        return float(probe_duration(media)), "ffprobe", sf_err
    except Exception as e:
        return None, "none", f"soundfile: {sf_err}; ffprobe: {type(e).__name__}: {e}"


def _count_words(vtt: Path, cues: Sequence[Cue]) -> Tuple[int, str]:
    """Word count, from the inline stream when the file has one.

    `parse_vtt` on a rolling caption inflates the count ~2x, which would push
    words/s out of range for a perfectly good reference — hence the branch.
    """
    if has_inline_word_timings(vtt):
        n = len(parse_vtt_words(vtt))
        if n:
            return n, "parse_vtt_words (inline stream)"
    return sum(len(c.text.split()) for c in cues), "parse_vtt cue text"


def reference_qc(vtt: Path, media: Path) -> RefQC:
    """Gate one reference subtitle against its media.

    Every rule's measured value lands in `rules` whether or not it fired, so a
    rejection can be argued with and a pass can be audited.
    """
    vtt, media = Path(vtt), Path(media)
    rules: Dict[str, Any] = {"vtt": str(vtt), "media": str(media)}
    failures: List[str] = []
    warnings: List[str] = []
    not_measured: List[str] = []

    if not vtt.exists():
        return RefQC(verdict="reject", rules=rules,
                     failures=[f"vtt_missing: {vtt}"],
                     not_measured=["every rule (no subtitle file)"])
    if not media.exists():
        return RefQC(verdict="reject", rules=rules,
                     failures=[f"media_missing: {media}"],
                     not_measured=["every rule (no media file)"])

    dur, dur_src, read_err = _media_duration(media)
    rules["media_duration_s"] = round(dur, 3) if dur else None
    rules["media_duration_src"] = dur_src
    rules["soundfile_readable"] = dur_src == "soundfile"
    rules["media_read_error"] = read_err
    if dur is None or dur <= 0:
        failures.append(f"media_unreadable: no backend could read a duration ({read_err})")
        return RefQC(verdict="reject", rules=rules, failures=failures,
                     not_measured=["cue_density_per_min", "coverage", "tail_ratio",
                                   "words_per_s (all need media duration)"])

    try:
        cues = parse_vtt(vtt) if vtt.suffix.lower() == ".vtt" else []
    except Exception as e:
        return RefQC(verdict="reject", rules=rules,
                     failures=[f"vtt_unparseable: {type(e).__name__}: {e}"],
                     not_measured=["every cue-derived rule"])
    rules["n_cues"] = len(cues)
    if not cues:
        failures.append("no_cues: the subtitle parsed to zero cues")
        return RefQC(verdict="reject", rules=rules, failures=failures,
                     not_measured=["cue_density_per_min", "max_cue_span_s",
                                   "coverage", "tail_ratio", "words_per_s"])

    cued_s = sum(c.end - c.start for c in cues)

    density = len(cues) / (dur / 60.0)
    rules["cue_density_per_min"] = round(density, 3)
    rules["cue_density_min"] = QC_MIN_CUES_PER_MIN
    if density < QC_MIN_CUES_PER_MIN:
        failures.append(f"cue_density_per_min={density:.2f} < {QC_MIN_CUES_PER_MIN}")

    span = max(c.end - c.start for c in cues)
    rules["max_cue_span_s"] = round(span, 3)
    rules["max_cue_span_max"] = QC_MAX_CUE_SPAN_S
    if span > QC_MAX_CUE_SPAN_S:
        # A warning, not a reject: one long cue over a music bed does not ruin
        # an otherwise dense reference, and cue_density catches the file that
        # is nothing *but* long cues.
        warnings.append(f"max_cue_span_s={span:.1f} > {QC_MAX_CUE_SPAN_S}")

    cov = cued_s / dur
    rules["coverage"] = round(cov, 4)
    rules["coverage_range"] = list(QC_COVERAGE_RANGE)
    rules["cued_s"] = round(cued_s, 3)
    if not (QC_COVERAGE_RANGE[0] <= cov <= QC_COVERAGE_RANGE[1]):
        failures.append(f"coverage={cov:.3f} outside {QC_COVERAGE_RANGE}")

    tail = cues[-1].end / dur
    rules["last_cue_end_s"] = round(cues[-1].end, 3)
    rules["tail_ratio"] = round(tail, 4)
    rules["tail_ratio_min"] = QC_MIN_TAIL_RATIO
    if tail < QC_MIN_TAIL_RATIO:
        failures.append(f"tail_ratio={tail:.3f} < {QC_MIN_TAIL_RATIO} (truncated reference)")

    n_words, word_src = _count_words(vtt, cues)
    wps = n_words / cued_s if cued_s > 0 else 0.0
    rules["n_words"] = n_words
    rules["word_count_src"] = word_src
    # Denominator is CAPTIONED time, not media time: a lecture with a silent
    # intro is not speaking slowly, it is just not speaking.
    rules["words_per_s"] = round(wps, 3)
    rules["words_per_s_denominator"] = "sum of cue durations"
    rules["words_per_s_range"] = list(QC_WPS_RANGE)
    if not (QC_WPS_RANGE[0] <= wps <= QC_WPS_RANGE[1]):
        failures.append(f"words_per_s={wps:.2f} outside {QC_WPS_RANGE}")

    try:
        roll = rolling_duplication_ratio(vtt)
    except Exception as e:
        roll = None
        not_measured.append(f"rolling_duplication_ratio: {type(e).__name__}: {e}")
    inline = has_inline_word_timings(vtt)
    rules["rolling_duplication_ratio"] = round(roll, 4) if roll is not None else None
    rules["rolling_duplication_max"] = QC_MAX_ROLLING_RATIO
    rules["has_inline_word_timings"] = inline
    # MEASURED on this corpus: conversational *manual* subtitles reach 0.262
    # (ZPUtA3W-7_I.en), 0.336 (ZPUtA3W-7_I.hi), 0.368 (qjPH9njnaVU.hi) purely
    # from genuine repetition in speech, while the one real rolling caption
    # (ZPUtA3W-7_I.en-orig) is 0.550. Ratio alone therefore cannot separate the
    # two populations. The rule's stated consequence — "must use
    # parse_vtt_words, not parse_vtt" — only applies when an inline stream
    # exists at all, so the warning fires on ratio AND inline. High repetition
    # without inline timings is recorded as a value, not a failure.
    rules["rolling_caption"] = bool(inline and roll is not None and roll > QC_MAX_ROLLING_RATIO)
    if rules["rolling_caption"]:
        warnings.append(
            f"rolling_duplication_ratio={roll:.3f} > {QC_MAX_ROLLING_RATIO} with inline "
            "timings: rolling caption, read it with parse_vtt_words not parse_vtt")
    elif roll is not None and roll > QC_MAX_ROLLING_RATIO:
        rules["high_repetition_no_inline"] = True

    verdict = "reject" if failures else ("pass_with_warnings" if warnings else "pass")
    return RefQC(verdict=verdict, rules=rules, failures=failures,
                 warnings=warnings, not_measured=not_measured)


# ---------------------------------------------------------------------------
# Track resolution
# ---------------------------------------------------------------------------

def _latin_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "LATIN" in unicodedata.name(c, "")) / len(letters)


@dataclass
class Tracks:
    video_id: str
    source_dir: Path
    source_wav: Optional[Path] = None
    ref_dub_wav: Optional[Path] = None
    source_vtt: Optional[Path] = None
    source_words_vtt: Optional[Path] = None    # *-orig.vtt with inline timings
    ref_vtt: Optional[Path] = None
    video: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


def resolve_tracks(video_id: str, source_dir: Path) -> Tracks:
    """Pick the right file per role, empirically — never by naming convention.

    `6-NRiom8K9Y` is why: its `*.en.vtt` is a **Bengali** translation and
    `*.en-LUU0EuDKgKo.vtt` is **Gujarati**, despite both declaring
    `Language: en`. The English one is `en-6UJrWS5jR_I`. So the English
    subtitle is chosen by measuring the Latin-script ratio of its text, and the
    chosen path is written into meta.json for the record.
    """
    d = Path(source_dir)
    t = Tracks(video_id=video_id, source_dir=d)

    for cand in (f"{video_id}.en.16k.wav", f"{video_id}.en-US.16k.wav"):
        if (d / cand).exists():
            t.source_wav = d / cand
            break
    else:
        hits = sorted(d.glob(f"{video_id}.en*.16k.wav"))
        if hits:
            t.source_wav = hits[0]

    if (d / f"{video_id}.hi.16k.wav").exists():
        t.ref_dub_wav = d / f"{video_id}.hi.16k.wav"
    else:
        t.notes.append(f"no Hindi 16k wav in {d.name}: this video has no dub track")

    best, best_r = None, -1.0
    for p in sorted(d.glob(f"{video_id}.en*.vtt")):
        if "-orig" in p.name:
            continue
        try:
            r = _latin_ratio(" ".join(c.text for c in parse_vtt(p)[:40]))
        except Exception as e:
            t.notes.append(f"could not parse {p.name}: {e!r}")
            continue
        if r > best_r:
            best, best_r = p, r
    if best is not None:
        t.source_vtt = best
        t.notes.append(f"english subtitle = {best.name} (latin-script ratio {best_r:.3f})")

    for p in sorted(d.glob(f"{video_id}.en*-orig.vtt")) + sorted(d.glob(f"{video_id}.en-orig.vtt")):
        if has_inline_word_timings(p):
            t.source_words_vtt = p
            break
    if t.source_words_vtt is None:
        t.notes.append("no *-orig.vtt with inline word timings: English words come from whisperx")

    if (d / f"{video_id}.hi.vtt").exists():
        t.ref_vtt = d / f"{video_id}.hi.vtt"
    if (d / f"{video_id}.mp4").exists():
        t.video = d / f"{video_id}.mp4"
    else:
        t.notes.append(f"no {video_id}.mp4 in {d.name}: clip.mp4 cannot be cut")
    return t


# ---------------------------------------------------------------------------
# Clip cutting + provenance
# ---------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
                           f"{(proc.stderr or '')[-1500:]}")


def _cut_wav(src: Path, dst: Path, t0: float, t1: float) -> List[str]:
    cmd = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}",
           "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)]
    _run(cmd)
    return cmd


def _cut_video(src: Path, dst: Path, t0: float, t1: float) -> List[str]:
    # Re-encoded, not stream-copied: a copy snaps to the nearest keyframe, which
    # would shift the clip by up to a GOP and silently decorrelate video from
    # the audio cut beside it.
    cmd = [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(dst)]
    _run(cmd)
    return cmd


def _clip_cues(vtt: Path, t0: float, t1: float) -> List[Dict[str, Any]]:
    """Cues overlapping [t0, t1), re-based to clip-relative time and clamped."""
    out = []
    for c in parse_vtt(vtt):
        if c.end <= t0 or c.start >= t1:
            continue
        out.append({
            "start": round(max(c.start, t0) - t0, 3),
            "end": round(min(c.end, t1) - t0, 3),
            "text": " ".join(c.text.split()),
            "clipped_head": c.start < t0,
            "clipped_tail": c.end > t1,
        })
    return out


def _clip_caption_words(vtt: Path, t0: float, t1: float) -> Tuple[List[Word], int]:
    """Caption-stream words overlapping [t0, t1), clip-relative.

    A word straddling the boundary is CLAMPED into the clip rather than
    dropped, so the text stays complete; clamping shortens its duration, so
    the count of clamped words is returned and recorded. Without this a word
    could carry start=-0.12, and "clip-relative" would stop being true.
    """
    dur = t1 - t0
    out: List[Word] = []
    clamped = 0
    for w in parse_vtt_words(vtt):
        if w.end <= t0 or w.start >= t1:
            continue
        s, e = w.start - t0, w.end - t0
        if s < 0.0 or e > dur:
            clamped += 1
        out.append(Word(text=w.text, start=round(max(s, 0.0), 4),
                        end=round(min(max(e, max(s, 0.0)), dur), 4),
                        src="vtt_inline", score=0.0))
    return out, clamped


def _norm_tok(s: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        ch for ch in s if ch.isalnum())).casefold()


def word_vad_coverage(words: Sequence[Word],
                      speech: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """How much of the VAD speech the aligned word stream actually accounts for.

    Text-free instrument against text-derived one: VAD says where speech is,
    the word stream says where TRANSCRIBED speech is, and a large shortfall
    means the transcript was incomplete rather than that the audio was silent.
    Returns None when either side is missing, so the caller records
    `not_measured` instead of a zero.
    """
    def _se(x: Any) -> Tuple[float, float]:
        if isinstance(x, dict):
            return (float(x.get("start", x.get("t_start", 0.0))),
                    float(x.get("end", x.get("t_end", 0.0))))
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            return float(x[0]), float(x[1])
        return float(getattr(x, "start", 0.0)), float(getattr(x, "end", 0.0))

    if not words or not speech:
        return None
    vad_s = sum(max(0.0, b - a) for a, b in (_se(x) for x in speech))
    word_s = sum(max(0.0, b - a) for a, b in (_se(w) for w in words))
    if vad_s <= 0:
        return None
    cov = word_s / vad_s
    rate = len(words) / vad_s
    return {
        "vad_speech_s": round(vad_s, 3),
        "word_span_s": round(word_s, 3),
        "coverage": round(cov, 4),
        "coverage_min": QC_MIN_WORD_VAD_COVERAGE,
        "n_words": len(words),
        "words_per_speech_s": round(rate, 3),
        "words_per_speech_s_min": QC_MIN_WORDS_PER_SPEECH_S,
        "usable_for_word_metrics": bool(rate >= QC_MIN_WORDS_PER_SPEECH_S),
    }


def agreement_median_ms(a: Sequence[Word], b: Sequence[Word]) -> Optional[Dict[str, Any]]:
    """Median |Δstart| between two word streams for the SAME track.

    This number is the ground truth's own noise floor. The study may not claim
    a timing win smaller than it: below this, the two equally-valid references
    disagree with each other by more than the effect.

    Tokens are matched with `difflib` on an alphanumeric-only normalisation,
    because the inline caption stream splits punctuation into its own token
    while whisperx attaches it.
    """
    if not a or not b:
        return None
    ta = [_norm_tok(w.text) for w in a]
    tb = [_norm_tok(w.text) for w in b]
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    deltas: List[float] = []
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            if ta[i + k]:
                deltas.append(abs(a[i + k].start - b[j + k].start))
    if not deltas:
        return None
    deltas.sort()
    return {
        "median_ms": round(statistics.median(deltas) * 1000.0, 1),
        "p90_ms": round(deltas[min(len(deltas) - 1, int(0.90 * len(deltas)))] * 1000.0, 1),
        "n_matched": len(deltas),
        "n_a": len(a),
        "n_b": len(b),
        "matched_frac": round(len(deltas) / min(len(a), len(b)), 4),
    }


def _words_doc(track: str, language: str, words: Sequence[Word], *,
               source: str, errors: List[str], not_measured: List[str],
               n_clamped: int = 0) -> Dict[str, Any]:
    return {
        "schema": "golden.words/1",
        "times": "clip-relative seconds; 0.0 == clip t0",
        "track": track,
        "language": language,
        "source": source,
        "n_words": len(words),
        # Words whose span crossed the clip boundary and were truncated; their
        # START is still exact, their DURATION is not.
        "n_clamped_at_boundary": n_clamped,
        "words": [{"text": w.text, "start": round(w.start, 4), "end": round(w.end, 4),
                   "src": w.src, "score": w.score} for w in words],
        "errors": errors,
        "not_measured": not_measured,
    }


def build_reference(clip: dict, *, out_root: Path, force: bool = False) -> dict:
    """Cut one golden clip and write its full ground truth.

    Writes `out_root/<clip_id>/`: meta.json, clip.mp4, source.wav, ref_dub.wav,
    source.words.json, ref_dub.words.json, source.vad.json, ref_dub.vad.json,
    reference_qc.json, segments.json. Returns the meta dict.

    Anything that could not be produced is named in `meta["errors"]` or
    `meta["not_measured"]` with the reason; the function does not raise for a
    missing track.
    """
    clip_id = clip["clip_id"]
    t0, t1 = float(clip["t0"]), float(clip["t1"])
    src_dir = Path(clip["source_dir"])
    if not src_dir.is_absolute():
        src_dir = ROOT / src_dir
    out_dir = Path(out_root) / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = resolve_tracks(clip["video_id"], src_dir)
    errors: List[str] = []
    not_measured: List[str] = []
    artifacts: Dict[str, Any] = {}
    commands: Dict[str, str] = {}

    meta: Dict[str, Any] = {
        "schema": "golden.meta/1",
        "times": ("ALL times in every artifact in this directory are CLIP-RELATIVE "
                  "(0.0 == source.t0). Only source.t0/source.t1 are absolute."),
        "clip_id": clip_id,
        "video_id": clip["video_id"],
        "tier": clip.get("tier"),
        "source": {
            "dir": str(src_dir.relative_to(ROOT)) if str(src_dir).startswith(str(ROOT)) else str(src_dir),
            "t0": t0, "t1": t1, "duration_s": round(t1 - t0, 3),
            "source_wav": str(tr.source_wav.name) if tr.source_wav else None,
            "ref_dub_wav": str(tr.ref_dub_wav.name) if tr.ref_dub_wav else None,
            "source_vtt": str(tr.source_vtt.name) if tr.source_vtt else None,
            "source_words_vtt": str(tr.source_words_vtt.name) if tr.source_words_vtt else None,
            "ref_vtt": str(tr.ref_vtt.name) if tr.ref_vtt else None,
            "video": str(tr.video.name) if tr.video else None,
        },
        "track_resolution_notes": tr.notes,
        "ref_dub_subtitle_timeline": HI_SUBTITLE_TIMELINE.get(clip["video_id"], "unknown"),
    }

    def _skip(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 0 and not force

    # --- media cuts -------------------------------------------------------
    cut_wavs: Dict[str, Path] = {}
    for role, src in (("source", tr.source_wav), ("ref_dub", tr.ref_dub_wav)):
        dst = out_dir / f"{role}.wav"
        if src is None:
            errors.append(f"{role}.wav: no source track ({role} audio absent for this video)")
            continue
        try:
            if _skip(dst):
                commands[f"{role}.wav"] = "(cached; rerun with --force to recut)"
            else:
                commands[f"{role}.wav"] = " ".join(_cut_wav(src, dst, t0, t1))
            cut_wavs[role] = dst
        except Exception as e:
            errors.append(f"{role}.wav: ffmpeg cut failed: {e!r}")

    clip_mp4 = out_dir / "clip.mp4"
    if tr.video is None:
        not_measured.append("clip.mp4: source video file absent in the corpus dir")
    else:
        try:
            if _skip(clip_mp4):
                commands["clip.mp4"] = "(cached; rerun with --force to recut)"
            else:
                commands["clip.mp4"] = " ".join(_cut_video(tr.video, clip_mp4, t0, t1))
        except Exception as e:
            errors.append(f"clip.mp4: ffmpeg cut failed: {e!r}")

    # --- segments ---------------------------------------------------------
    segments: Dict[str, Any] = {
        "schema": "golden.segments/1",
        "times": "clip-relative seconds; 0.0 == clip t0",
        "tracks": {},
    }
    for role, vtt, lang in (("source", tr.source_vtt, "en"), ("ref_dub", tr.ref_vtt, "hi")):
        if vtt is None:
            segments["tracks"][role] = {"vtt": None, "cues": [],
                                        "error": f"no {lang} subtitle for this video"}
            not_measured.append(f"segments[{role}]: no {lang} subtitle")
            continue
        trust = ("source" if role == "source"
                 else HI_SUBTITLE_TIMELINE.get(clip["video_id"], "unknown"))
        entry: Dict[str, Any] = {"vtt": vtt.name, "language": lang,
                                 "timing_trust": trust, "cues": _clip_cues(vtt, t0, t1)}
        if role == "ref_dub" and trust != "dub":
            entry["warning"] = (
                "cue TIMES are on the source-language timeline, not the dub — "
                "use the text only; dub timing comes from ref_dub.words.json")
        segments["tracks"][role] = entry
    (out_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2))

    # --- reference QC (whole-file: density/coverage/tail are file properties) --
    qc_doc: Dict[str, Any] = {
        "schema": "golden.reference_qc/1",
        "scope": ("run on the FULL source subtitle + media, not the clip: cue density, "
                  "coverage and tail truncation are properties of the whole reference"),
        "tracks": {},
    }
    for role, vtt, media in (("source", tr.source_vtt, tr.source_wav),
                             ("ref_dub", tr.ref_vtt, tr.ref_dub_wav)):
        if vtt is None or media is None:
            qc_doc["tracks"][role] = {"verdict": "not_measured",
                                      "reason": f"missing {'vtt' if vtt is None else 'media'}"}
            not_measured.append(f"reference_qc[{role}]: missing "
                                f"{'subtitle' if vtt is None else 'media'}")
            continue
        qc_doc["tracks"][role] = reference_qc(vtt, media).to_dict()
    (out_dir / "reference_qc.json").write_text(json.dumps(qc_doc, ensure_ascii=False, indent=2))
    meta["reference_qc"] = {k: v.get("verdict") for k, v in qc_doc["tracks"].items()}

    # --- VAD --------------------------------------------------------------
    meta["vad"] = {}
    vad_speech_by_role: Dict[str, List[Any]] = {}
    for role, wav in cut_wavs.items():
        v = vad(wav)
        d = v.to_dict()
        vad_speech_by_role[role] = d.get("speech") or []
        (out_dir / f"{role}.vad.json").write_text(json.dumps(d, indent=2))
        meta["vad"][role] = {"backend": d["backend"], "n_speech": d["n_speech"],
                             "n_pauses": d["n_pauses"], "speech_frac": d["speech_frac"]}
        if v.error:
            errors.append(f"{role}.vad: {v.error}")
    for role in ("source", "ref_dub"):
        if role not in cut_wavs:
            not_measured.append(f"{role}.vad.json: no {role}.wav to analyse")

    # --- words ------------------------------------------------------------
    meta["words"] = {}
    meta["agreement_median_ms"] = {}
    for role, lang, vtt in (("source", "en", tr.source_vtt), ("ref_dub", "hi", tr.ref_vtt)):
        werrs: List[str] = []
        wnm: List[str] = []
        wav = cut_wavs.get(role)
        if wav is None:
            not_measured.append(f"{role}.words.json: no {role}.wav to align against")
            continue

        # Reference text for forced alignment. For ref_dub on a
        # source-timeline subtitle the TIMES are wrong but the TEXT is right,
        # which is exactly what forced alignment needs.
        text = " ".join(c["text"] for c in
                        segments["tracks"].get(role, {}).get("cues", []))

        caption_words: List[Word] = []
        n_clamped = 0
        if role == "source" and tr.source_words_vtt is not None:
            try:
                caption_words, n_clamped = _clip_caption_words(tr.source_words_vtt, t0, t1)
            except Exception as e:
                werrs.append(f"caption word stream: {type(e).__name__}: {e}")

        wx_words: List[Word] = []
        if text:
            wx_words = align_words(wav, text, language=lang)
            if not wx_words:
                werrs.append("whisperx forced-align returned no timed words "
                             "(see log for the reason)")
        else:
            wnm.append(f"whisperx align: no {lang} reference text in this clip window")

        if caption_words:
            chosen, source_label = caption_words, f"{tr.source_words_vtt.name} inline stream"
        elif wx_words:
            chosen, source_label, n_clamped = wx_words, f"whisperx forced-align ({lang})", 0
        else:
            chosen, source_label, n_clamped = [], "none", 0
            werrs.append("no word timings from any route")

        (out_dir / f"{role}.words.json").write_text(json.dumps(
            _words_doc(role, lang, chosen, source=source_label, errors=werrs,
                       not_measured=wnm, n_clamped=n_clamped),
            ensure_ascii=False, indent=2))
        meta["words"][role] = {"n": len(chosen), "source": source_label}

        # Gate the stream against the text-free VAD for THIS clip. A sparse
        # transcript produces a confident-looking but badly incomplete word
        # stream, and nothing else in the pipeline would notice.
        cov = word_vad_coverage(chosen, vad_speech_by_role.get(role) or [])
        if cov is None:
            meta["words"][role]["vad_coverage"] = {
                "not_measured": "needs both aligned words and a VAD speech track"}
        else:
            meta["words"][role]["vad_coverage"] = cov
            if not cov["usable_for_word_metrics"]:
                msg = (f"{role}.words.json carries only {cov['words_per_speech_s']:.2f} words "
                       f"per second of VAD speech ({cov['n_words']} words over "
                       f"{cov['vad_speech_s']:.1f}s, spanning {cov['coverage']:.0%}) -- below "
                       f"{QC_MIN_WORDS_PER_SPEECH_S}; the transcript for this window is "
                       f"incomplete, so this clip must NOT be used for word-level timing "
                       f"metrics. VAD-based metrics are unaffected.")
                log.warning("refbuild: %s: %s", clip_id, msg)
                meta.setdefault("not_measured", []).append(msg)

        if caption_words and wx_words:
            # Both routes exist -> the disagreement between them is the noise
            # floor of this track's ground truth.
            (out_dir / f"{role}.words.whisperx.json").write_text(json.dumps(
                _words_doc(role, lang, wx_words, source=f"whisperx forced-align ({lang})",
                           errors=[], not_measured=[]), ensure_ascii=False, indent=2))
            agr = agreement_median_ms(caption_words, wx_words)
            meta["agreement_median_ms"][role] = agr or {
                "error": "no tokens matched between the caption stream and whisperx"}
        else:
            meta["agreement_median_ms"][role] = {
                "not_measured": ("needs BOTH a caption word stream and whisperx words; "
                                 f"caption={len(caption_words)}, whisperx={len(wx_words)}")}

    # --- provenance -------------------------------------------------------
    for p in sorted(out_dir.iterdir()):
        if p.name == "meta.json" or not p.is_file():
            continue
        artifacts[p.name] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
    meta["ffmpeg"] = commands
    meta["artifacts"] = artifacts
    meta["errors"] = errors
    meta["not_measured"] = not_measured
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log.info("build_reference: %s -> %s (%d artifacts, %d errors)",
             clip_id, out_dir, len(artifacts), len(errors))
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_qc(label: str, qc: RefQC) -> None:
    print(f"\n{label}")
    print(f"  verdict: {qc.verdict.upper()}")
    for k, v in qc.rules.items():
        print(f"    {k:<30} {v}")
    for f_ in qc.failures:
        print(f"  FAIL    {f_}")
    for w in qc.warnings:
        print(f"  WARN    {w}")
    for n in qc.not_measured:
        print(f"  N/M     {n}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("refbuild", help="cut golden clips and build their ground truth")
    b.add_argument("--clips", default="data/golden/index.json")
    b.add_argument("--out", default="data/golden")
    b.add_argument("--only", nargs="*", help="restrict to these clip_ids")
    b.add_argument("--tier", nargs="*", help="restrict to these tiers (A/B/C)")
    b.add_argument("--force", action="store_true", help="recut media even if present")

    g = sub.add_parser("goldcheck", help="QC one reference subtitle against its media")
    g.add_argument("--vtt", required=True)
    g.add_argument("--media", required=True)
    g.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if a.cmd == "goldcheck":
        qc = reference_qc(Path(a.vtt), Path(a.media))
        if a.json:
            print(json.dumps(qc.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_qc(f"goldcheck {a.vtt}", qc)
        return 0 if qc.verdict != "reject" else 1

    clips_path = Path(a.clips)
    if not clips_path.is_absolute():
        clips_path = ROOT / clips_path
    clips = json.loads(clips_path.read_text())
    if a.only:
        clips = [c for c in clips if c["clip_id"] in set(a.only)]
    if a.tier:
        want = {t.upper() for t in a.tier}
        clips = [c for c in clips if str(c.get("tier", "")).upper() in want]
    if not clips:
        print("no clips matched the filters", file=sys.stderr)
        return 2

    out_root = Path(a.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    rc = 0
    for c in clips:
        try:
            m = build_reference(c, out_root=out_root, force=a.force)
        except Exception as e:
            log.exception("refbuild: %s failed", c["clip_id"])
            print(f"{c['clip_id']}: FAILED {e!r}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n{m['clip_id']}  tier={m['tier']}  {m['source']['t0']}-{m['source']['t1']}s")
        print(f"  qc        {m['reference_qc']}")
        print(f"  vad       {m['vad']}")
        print(f"  words     {m['words']}")
        print(f"  agreement {m['agreement_median_ms']}")
        if m["errors"]:
            print(f"  errors    {m['errors']}")
        if m["not_measured"]:
            print(f"  n/m       {m['not_measured']}")
    return rc


if __name__ == "__main__":
    # `python -m backend.eval.reference refbuild ...` works even when the
    # package-level `__main__.py` (owned by the harness) changes shape.
    raise SystemExit(main())
