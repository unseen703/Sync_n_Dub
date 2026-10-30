"""Backend-facing forced-alignment entry point.

Reads an ASR JSON (produced by `backend.asr`) and runs a CTC aligner over
each segment to produce word-level timestamps. Writes back a parallel JSON
with the same segment ids and the `words[]` field filled in.

The aligner needs the audio the ASR saw — not the full mix — because the
diarized dialogue stem is what the segment timings were cut against. The
caller passes `audio` explicitly; the pipeline passes the dialogue stem (or
the raw 16 k mono when stems were skipped).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from backend.media.align import (
    AlignRequest,
    AlignResult,
    available_models,
    get_aligner,
)
from backend.asr import ASRResult, ASRSegment, Word

log = logging.getLogger(__name__)


def align_asr(
    audio: Path,
    asr_result: ASRResult,
    *,
    model: str = "mms_fa",
    device: Optional[str] = None,
    language: Optional[str] = None,
    **model_kwargs,
) -> AlignResult:
    """Align every segment of `asr_result` against `audio`.

    `language` falls back to the ASR's own detected language.
    """
    audio = Path(audio)
    if not audio.exists():
        raise FileNotFoundError(audio)
    if model not in available_models():
        raise ValueError(
            f"Unknown aligner {model!r}. Available: {', '.join(available_models())}"
        )

    requests: List[AlignRequest] = [
        AlignRequest(
            id=seg.id, start=float(seg.start), end=float(seg.end),
            text=(seg.text or "").strip(),
            language=seg.language or language or asr_result.language,
            speaker=seg.speaker,
        )
        for seg in asr_result.segments
    ]

    log.info("align: %s with %s (device=%s, segments=%d)",
             audio.name, model, device or "auto", len(requests))
    aligner = get_aligner(model, device=device, **model_kwargs)
    result = aligner.align(audio, requests,
                           language=language or asr_result.language)
    nw = sum(len(s.words) for s in result.segments)
    log.info("align: %d segments, %d words (model=%s)",
             len(result.segments), nw, result.model_name)
    return result


def apply_alignment_to_asr(asr_result: ASRResult,
                            align_result: AlignResult) -> ASRResult:
    """Return a new `ASRResult` with each segment's `words` replaced from
    the aligner's output. Order and ids must already match.
    """
    by_id = {s.id: s for s in align_result.segments}
    new_segments: List[ASRSegment] = []
    for seg in asr_result.segments:
        a = by_id.get(seg.id)
        words = []
        if a is not None:
            words = [Word(text=w.text, start=w.start, end=w.end, score=w.score)
                     for w in a.words]
        new_segments.append(ASRSegment(
            id=seg.id, start=seg.start, end=seg.end, text=seg.text,
            speaker=seg.speaker, words=words, language=seg.language,
        ))
    return ASRResult(
        audio_path=asr_result.audio_path,
        model_name=asr_result.model_name,
        language=asr_result.language,
        segments=new_segments,
        extra={**asr_result.extra, "aligner": align_result.model_name},
    )


# ===========================================================================
# M5 -- forced-align the TTS OUTPUT (not the source)
# ===========================================================================
#
# TIME BASES. The two word fields are not interchangeable, and mixing them
# gives a plan that is wrong by one segment offset and still renders cleanly:
#
#   segments[].words[]      ABSOLUTE seconds on the SOURCE timeline, from
#                           `align_asr` above against the source dialogue stem.
#   segments[].tts_words[]  CLIP-RELATIVE seconds; 0.0 is the first sample of
#                           that segment's raw TTS wav. Same frame as a
#                           Piece's `src_span`. They say nothing about where
#                           the segment lands on the output timeline.
#
# Only add `segments[].start` to a tts_word once you have decided the segment
# sits at its slot start, which pause-aware placement does not.
#
# `BaseAligner.align` takes (audio_path, requests) and does not care where the
# audio came from, so TTS output is just another wav with known text. Two
# constraints:
#   * mms_fa raises when sr != 16000 and edge-tts writes 24 kHz, so every clip
#     is resampled first, through `backend.media_io`'s ffmpeg resolver rather
#     than a bare `ffmpeg`.
#   * The aligner is built ONCE and threaded through every clip; per-segment
#     construction reloads the ~1 GB MMS_FA bundle each time.

import hashlib
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

ALIGN_SAMPLE_RATE = 16_000

# Shortest alignable clip. Below this there is not a CTC frame's worth of
# evidence; the mms_fa adapter declines <50 ms, this declines earlier.
MIN_ALIGNABLE_S = 0.06


def _resample_16k(audio: Path, *, work_dir: Optional[Path] = None) -> Path:
    """Return a 16 kHz mono wav for `audio`, converting only when needed.

    Uses `media_io._ffmpeg_bin()` rather than assuming `ffmpeg` is on PATH.
    """
    import soundfile as sf

    from backend.common.media_io import _ffmpeg_bin

    audio = Path(audio)
    info = sf.info(str(audio))
    if info.samplerate == ALIGN_SAMPLE_RATE and info.channels == 1:
        return audio

    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ttsalign_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    # Hash the resolved source path into the name: every TTS clip is called
    # SEG_NNNN.wav, so a bare stem collides as soon as two merged JSONs share
    # a work_dir and one gets aligned against the other's audio.
    tag = hashlib.sha1(str(audio.resolve()).encode()).hexdigest()[:10]
    out = work_dir / f"{audio.stem}.{tag}.16k.mono.wav"
    cmd = [
        _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(audio),
        "-ac", "1", "-ar", str(ALIGN_SAMPLE_RATE), "-acodec", "pcm_s16le",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"resample to 16k failed (rc={proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\nSTDERR: {proc.stderr.strip()}"
        )
    return out


def _warm_aligner(model: str, device: Optional[str], aligner, **model_kwargs):
    """One aligner, reused. `aligner` wins so callers can share an instance."""
    if aligner is not None:
        return aligner
    return get_aligner(model, device=device, **model_kwargs)


def align_audio_to_text(
    audio: Path,
    text: str,
    *,
    model: str = "mms_fa",
    device: Optional[str] = None,
    language: Optional[str] = None,
    segment_id: str = "SEG_0000",
    aligner=None,
    work_dir: Optional[Path] = None,
    **model_kwargs,
) -> List[Word]:
    """Word timings for one standalone clip, CLIP-RELATIVE to its first sample.

    Never raises on an unalignable clip; returns `[]`.
    """
    import soundfile as sf

    audio = Path(audio)
    text = (text or "").strip()
    if not audio.exists():
        log.warning("align_audio_to_text: %s missing (%s)", audio, segment_id)
        return []
    if not text:
        return []

    try:
        wav16 = _resample_16k(audio, work_dir=work_dir)
        dur = float(sf.info(str(wav16)).duration)
    except Exception as e:  # noqa: BLE001
        log.warning("align_audio_to_text: %s unreadable (%s)", segment_id, e)
        return []
    if dur < MIN_ALIGNABLE_S:
        log.warning("align_audio_to_text: %s is %.3fs, too short to align",
                    segment_id, dur)
        return []

    al = _warm_aligner(model, device, aligner, **model_kwargs)
    # start=0.0 makes the returned times clip-relative: the mms_fa adapter
    # offsets every word by `req.start`.
    req = AlignRequest(id=segment_id, start=0.0, end=dur, text=text,
                       language=language)
    try:
        res = al.align(wav16, [req], language=language)
    except Exception as e:  # noqa: BLE001
        log.warning("align_audio_to_text: %s failed (%s)", segment_id, e)
        return []
    return list(res.segments[0].words) if res.segments else []


def align_tts_segments(
    merged_json: Path,
    *,
    model: str = "mms_fa",
    language: str,
    device: Optional[str] = None,
    audio_field: str = "tts_raw_audio",
    work_dir: Optional[Path] = None,
    out_json: Optional[Path] = None,
    text_field: Optional[str] = None,
    **model_kwargs,
) -> dict:
    """Fill `segments[].tts_words[]` for every segment that has a TTS clip.

    `tts_words[]` entries are `{text,start,end,score}` with **clip-relative**
    times (see the schema warning at the top of this section).

    Defaults to `tts_raw_audio`: `tts_audio` may already have been stretched
    by the TTS stage, and times measured on a stretched clip cannot be used to
    cut the raw clip, which is what `Piece.src_span` does.

    Degrades per segment: a clip that will not align gets `tts_words = []`
    plus `qc["tts_align"]` naming the reason, and the run continues.
    """
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segments = data.get("segments", []) or []

    work_dir = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="tts_align_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    aligner = None
    n_done = n_words = 0
    failures: List[dict] = []

    for seg in segments:
        sid = seg.get("id", "SEG_????")
        clip = seg.get(audio_field) or seg.get("tts_audio")
        txt = (seg.get(text_field) if text_field
               else (seg.get("final_text") or seg.get("mt_text") or seg.get("text")))
        reason = None
        if not clip:
            reason = "no_audio"
        elif not Path(clip).exists():
            reason = "audio_missing"
        elif not (txt or "").strip():
            reason = "no_text"

        if reason:
            seg["tts_words"] = []
            seg.setdefault("qc", {})["tts_align"] = reason
            failures.append({"id": sid, "reason": reason})
            continue

        if aligner is None:          # built lazily so a no-clip run costs nothing
            aligner = _warm_aligner(model, device, None, **model_kwargs)

        words = align_audio_to_text(
            Path(clip), txt, model=model, device=device, language=language,
            segment_id=sid, aligner=aligner, work_dir=work_dir,
        )
        seg["tts_words"] = [
            {"text": w.text, "start": round(float(w.start), 4),
             "end": round(float(w.end), 4),
             "score": (round(float(w.score), 4) if w.score is not None else None)}
            for w in words
        ]
        if not words:
            seg.setdefault("qc", {})["tts_align"] = "empty"
            failures.append({"id": sid, "reason": "empty"})
        else:
            seg.get("qc", {}).pop("tts_align", None)
            n_done += 1
            n_words += len(words)

    data.setdefault("extra", {})["tts_align"] = {
        "model": model,
        "language": language,
        "audio_field": audio_field,
        "segments": n_done,
        "words": n_words,
        "failures": failures,
    }
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("tts_align: %d/%d segments, %d words, %d failures",
             n_done, len(segments), n_words, len(failures))
    return data


# ===========================================================================
# M6 -- viseme events
# ===========================================================================
#
# A viseme event here is a time-stamped, coarse MOUTH SHAPE class, not a
# phoneme: the eye resolves about nine shapes, of which "lips closed"
# (BILABIAL) is the most salient.
#
# The `mms_char` backend gets them free. torchaudio's MMS_FA aligner emits one
# `TokenSpan` per CHARACTER of uroman-romanized text, and `mms_fa_adapter`
# computes exactly that before discarding everything but each word's
# first/last span. `_mms_char_spans` re-runs the same call and keeps the
# characters. uroman maps English source and Devanagari target into one Latin
# inventory, so a Devanagari म and an English "m" both land in BILABIAL -- the
# claim is not that the phoneme sequences are equal, but that the visually
# salient events happen at the same TIMES, which is what DTW over
# (onset, class) measures.
#
# uroman is orthographic, not phonetic, so the class sequences carry noise:
# Devanagari's inherent schwa romanizes a vowel Hindi speech may delete, and
# English "know" keeps its silent k. The `allosaurus` backend reads phones
# straight off the audio with no text and no G2P, and is the better signal
# where it is installed.

VISEME_CLASSES = (
    "BILABIAL",     # p b m   -- lips fully closed; the loudest visual event
    "LABIODENTAL",  # f v     -- lower lip to teeth
    "ROUNDED",      # o u w   -- lips protruded
    "OPEN_V",       # a       -- jaw open
    "CLOSE_V",      # i e y   -- jaw near-closed, lips spread
    "ALVEOLAR",     # t d n l r
    "VELAR",        # k g q h -- almost no visible lip change
    "SIBILANT",     # s z c j x
    "OTHER",
)

# Latin char -> class, over the uroman/MMS_FA inventory (lowercase a-z + ').
_CHAR_VISEME: Dict[str, str] = {
    **{c: "BILABIAL" for c in "pbm"},
    **{c: "LABIODENTAL" for c in "fv"},
    **{c: "ROUNDED" for c in "ouw"},
    **{c: "OPEN_V" for c in "a"},
    **{c: "CLOSE_V" for c in "iey"},
    **{c: "ALVEOLAR" for c in "tdnlr"},
    **{c: "VELAR" for c in "kgqh"},
    **{c: "SIBILANT" for c in "szcjx"},
}

# IPA -> class, for the allosaurus backend. Only the lip-relevant distinctions
# matter, so this is deliberately coarse and falls through to OTHER.
_IPA_VISEME: Dict[str, str] = {
    **{c: "BILABIAL" for c in "pbmɱʙ"},
    **{c: "LABIODENTAL" for c in "fvʋⱱ"},
    **{c: "ROUNDED" for c in "ouwʊɔɵʉøyɥœɒ"},
    **{c: "OPEN_V" for c in "aɑæɐʌɛɜ"},
    **{c: "CLOSE_V" for c in "ieɪɨɘəɯ"},
    **{c: "ALVEOLAR" for c in "tdnlrɾɽɳɖʈɻɭɬ"},
    **{c: "VELAR" for c in "kgŋxɣqɢχʁhɦʔɰ"},
    **{c: "SIBILANT" for c in "szʃʒʂʐɕʑtʃdʒ"},
}

# Mismatch and skip costs are scaled by class; BILABIAL dominates because a
# missing lip closure is the most visible error.
DEFAULT_CLASS_WEIGHTS: Dict[str, float] = {
    "BILABIAL": 2.0,
    "LABIODENTAL": 1.3,
    "ROUNDED": 1.0,
    "OPEN_V": 0.8,
    "CLOSE_V": 0.8,
    "SIBILANT": 0.8,
    "ALVEOLAR": 0.7,
    "VELAR": 0.6,
    "OTHER": 0.4,
}

VISEME_BACKENDS = ("mms_char", "allosaurus")


@dataclass
class VisemeEvent:
    t: float        # onset, seconds (clip-relative + `clip_offset_s`)
    dur: float
    cls: str
    char: str       # the romanized char / IPA phone it came from, for debugging
    score: float

    def to_jsonable(self) -> dict:
        return {"t": round(self.t, 4), "dur": round(self.dur, 4),
                "cls": self.cls, "char": self.char, "score": round(self.score, 4)}

    @classmethod
    def from_jsonable(cls, d: dict) -> "VisemeEvent":
        return cls(t=float(d["t"]), dur=float(d.get("dur", 0.0)),
                   cls=d.get("cls", "OTHER"), char=d.get("char", ""),
                   score=float(d.get("score", 0.0)))


def _merge_runs(events: List[VisemeEvent], *, min_class_dur_s: float
                ) -> List[VisemeEvent]:
    """Collapse consecutive same-class chars, then drop what is too brief to see.

    "mm" is one mouth shape held, not two events, and a shape shorter than
    ~30 ms does not survive a single 25 fps frame.

    THE THRESHOLDED DURATION IS THE HOLD -- this onset to the next -- not
    `e.dur`. For `mms_char`, `e.dur` is the width of the CTC emission frames
    for one character and MMS_FA's frame period is 20 ms, so most spans are
    exactly one frame wide; thresholding `e.dur` at 30 ms deletes almost every
    event, BILABIAL worst of all, and leaves `viseme_cost` nothing to match.
    A CTC span is an onset marker, not an articulation. The last event has no
    successor, so it falls back to its own span.
    """
    if not events:
        return []
    merged: List[VisemeEvent] = []
    cur = VisemeEvent(**vars(events[0]))
    for e in events[1:]:
        if e.cls == cur.cls and e.t - (cur.t + cur.dur) < 0.02:
            cur.dur = max(cur.dur, e.t + e.dur - cur.t)
            cur.char += e.char
            cur.score = (cur.score + e.score) / 2.0
        else:
            merged.append(cur)
            cur = VisemeEvent(**vars(e))
    merged.append(cur)
    if min_class_dur_s <= 0.0:
        return merged
    kept: List[VisemeEvent] = []
    for i, e in enumerate(merged):
        hold = (merged[i + 1].t - e.t) if i + 1 < len(merged) else e.dur
        if hold >= min_class_dur_s:
            kept.append(e)
    return kept


def _mms_char_spans(aligner, clip, sr: int, text: str) -> List[Tuple[str, float, float, float]]:
    """(char, t0, t1, score) per romanized character, clip-relative seconds.

    Calls `mms_fa_adapter._align_clip` directly -- the same call `align()`
    makes -- because the public `align()` discards the per-character spans and
    returns only word boundaries.
    """
    import torch

    from backend.media.align.mms_fa_adapter import _romanize_pairs, _split_words

    aligner._ensure_loaded()
    paired = _romanize_pairs(_split_words(text), aligner._get_uroman())
    if not paired:
        return []
    words_latin = [latin for _, latin in paired]
    token_spans, num_frames = aligner._align_clip(clip, words_latin, torch)
    if not token_spans or not num_frames:
        return []

    inv = {v: k for k, v in (aligner._dictionary or {}).items()}
    sec_per_frame = (clip.shape[0] / sr) / num_frames
    out: List[Tuple[str, float, float, float]] = []
    for latin, spans in zip(words_latin, token_spans):
        for i, sp in enumerate(spans):
            ch = inv.get(int(sp.token))
            if ch is None:                 # fall back to source char order
                ch = latin[i] if i < len(latin) else "?"
            out.append((ch, sp.start * sec_per_frame, sp.end * sec_per_frame,
                        float(sp.score)))
    return out


def _allosaurus_events(wav16: Path, *, language: Optional[str],
                       min_class_dur_s: float, clip_offset_s: float
                       ) -> List[VisemeEvent]:
    """Timestamped phones straight from audio -- no text, no G2P, no uroman."""
    from allosaurus.app import read_recognizer

    global _ALLOSAURUS
    try:
        _ALLOSAURUS
    except NameError:
        _ALLOSAURUS = None
    if _ALLOSAURUS is None:
        _ALLOSAURUS = read_recognizer()
    lang = {"hi": "hin", "en": "eng"}.get((language or "").lower(), "ipa")
    raw = _ALLOSAURUS.recognize(str(wav16), lang, timestamp=True)

    events: List[VisemeEvent] = []
    for line in (raw or "").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t0, dur = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        phone = parts[2]
        cls = _IPA_VISEME.get(phone[0], "OTHER") if phone else "OTHER"
        events.append(VisemeEvent(t=t0 + clip_offset_s, dur=dur, cls=cls,
                                  char=phone, score=1.0))
    return _merge_runs(events, min_class_dur_s=min_class_dur_s)


def extract_viseme_events(
    audio: Path,
    text: str,
    *,
    model: str = "mms_fa",
    language: Optional[str] = None,
    device: Optional[str] = None,
    aligner=None,
    clip_offset_s: float = 0.0,
    min_class_dur_s: float = 0.03,
    backend: str = "mms_char",
    span: Optional[Tuple[float, float]] = None,
    work_dir: Optional[Path] = None,
) -> List[VisemeEvent]:
    """Coarse mouth-shape events for one clip.

    `clip_offset_s` is added to every onset, so the caller decides whether the
    result is clip-relative (0.0, the default) or absolute on some timeline.
    `span` optionally restricts the analysis to (start, end) seconds of `audio`,
    which is how the source side cuts one segment out of the full dialogue stem.

    Returns `[]` rather than raising on anything unalignable.
    """
    import numpy as np
    import soundfile as sf

    if backend not in VISEME_BACKENDS:
        raise ValueError(f"unknown viseme backend {backend!r}; "
                         f"choose one of {VISEME_BACKENDS}")
    audio = Path(audio)
    if not audio.exists():
        log.warning("viseme: %s missing", audio)
        return []

    try:
        wav16 = _resample_16k(audio, work_dir=work_dir)
        data, sr = sf.read(str(wav16), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        log.warning("viseme: %s unreadable (%s)", audio, e)
        return []
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1)
    if span is not None:
        s0 = max(0, int(round(span[0] * sr)))
        s1 = min(len(data), int(round(span[1] * sr)))
        data = data[s0:s1]
    if data.size < int(MIN_ALIGNABLE_S * sr):
        log.warning("viseme: clip %s too short (%d samples)", audio.name, data.size)
        return []

    if backend == "allosaurus":
        try:
            if span is None and wav16 == audio:
                tmp = wav16
            else:                       # allosaurus reads a path, not an array
                # Same collision as in `_resample_16k`: side="source" cuts
                # every segment out of one dialogue stem, so a bare stem names
                # every cut the same file.
                tag = hashlib.sha1(
                    f"{audio.resolve()}|{span}".encode()).hexdigest()[:10]
                tmp = Path(work_dir or tempfile.mkdtemp(prefix="viseme_")) / \
                    f"{audio.stem}.{tag}.span.wav"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(tmp), data, sr)
            return _allosaurus_events(tmp, language=language,
                                      min_class_dur_s=min_class_dur_s,
                                      clip_offset_s=clip_offset_s)
        except Exception as e:  # noqa: BLE001
            log.warning("viseme[allosaurus]: failed on %s (%s); returning []",
                        audio.name, e)
            return []

    text = (text or "").strip()
    if not text:
        return []
    al = _warm_aligner(model, device, aligner)
    try:
        spans = _mms_char_spans(al, np.ascontiguousarray(data), sr, text)
    except Exception as e:  # noqa: BLE001
        log.warning("viseme[mms_char]: failed on %s (%s); returning []",
                    audio.name, e)
        return []

    events = [
        VisemeEvent(t=t0 + clip_offset_s, dur=max(0.0, t1 - t0),
                    cls=_CHAR_VISEME.get(ch, "OTHER"), char=ch, score=sc)
        for ch, t0, t1, sc in spans
    ]
    return _merge_runs(events, min_class_dur_s=min_class_dur_s)


def viseme_events_for_merged(
    merged_json: Path,
    *,
    side: str,
    audio: Optional[Path] = None,
    model: str = "mms_fa",
    language: str,
    out_json: Optional[Path] = None,
    backend: str = "mms_char",
    device: Optional[str] = None,
    audio_field: str = "tts_raw_audio",
    min_class_dur_s: float = 0.03,
    work_dir: Optional[Path] = None,
) -> dict:
    """Viseme events for every segment, for one side of the comparison.

    side="source": one clip cut from the source dialogue stem per segment,
      analysed with the segment's SOURCE text. Onsets are ABSOLUTE (offset by
      the segment's slot start), because that is the timeline they live on.
    side="tts":    the segment's raw TTS clip with its translated text. Onsets
      are CLIP-RELATIVE, matching `tts_words[]`.

    Writes `segments[].viseme_<side>[]` and `extra.visemes_<side>`.
    """
    if side not in ("source", "tts"):
        raise ValueError(f"side must be 'source' or 'tts', got {side!r}")
    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segments = data.get("segments", []) or []
    work_dir = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix=f"viseme_{side}_"))

    src_audio = Path(audio) if audio else Path(data.get("audio_path", ""))
    aligner = None
    n_ev = n_seg = 0
    failures: List[dict] = []
    key = f"viseme_{side}"

    for seg in segments:
        sid = seg.get("id", "SEG_????")
        if side == "source":
            clip, span = src_audio, (float(seg.get("start", 0.0)),
                                     float(seg.get("end", 0.0)))
            txt = seg.get("text") or ""
            offset = float(seg.get("start", 0.0))
            lang = seg.get("language") or language
        else:
            c = seg.get(audio_field) or seg.get("tts_audio")
            clip, span = (Path(c) if c else None), None
            txt = seg.get("final_text") or seg.get("mt_text") or ""
            offset, lang = 0.0, language

        if clip is None or not Path(clip).exists():
            seg[key] = []
            seg.setdefault("qc", {})[f"viseme_{side}"] = "no_audio"
            failures.append({"id": sid, "reason": "no_audio"})
            continue

        if aligner is None and backend == "mms_char":
            aligner = _warm_aligner(model, device, None)

        evs = extract_viseme_events(
            clip, txt, model=model, language=lang, device=device,
            aligner=aligner, clip_offset_s=offset,
            min_class_dur_s=min_class_dur_s, backend=backend,
            span=span, work_dir=work_dir,
        )
        seg[key] = [e.to_jsonable() for e in evs]
        if not evs:
            seg.setdefault("qc", {})[f"viseme_{side}"] = "empty"
            failures.append({"id": sid, "reason": "empty"})
        else:
            n_seg += 1
            n_ev += len(evs)

    data.setdefault("extra", {})[f"visemes_{side}"] = {
        "backend": backend, "model": model, "language": language,
        "time_base": "absolute" if side == "source" else "clip_relative",
        "segments": n_seg, "events": n_ev, "failures": failures,
    }
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("visemes[%s]: %d/%d segments, %d events, %d failures",
             side, n_seg, len(segments), n_ev, len(failures))
    return data


# ---------------------------------------------------------------------------
# M6 call site: write the keys the planner actually reads
# ---------------------------------------------------------------------------
#
# Two consumers want two shapes. `viseme_cost` wants ONSET EVENTS, which
# `viseme_events_for_merged` writes as `viseme_source[]` / `viseme_tts[]`.
# `backend/timing_advanced.py::_d_phoneme` wants HALF-OPEN INTERVALS
# `{start,end,viseme}` under `segments[].src_visemes` / `tts_visemes`, which it
# samples at 20 ms frames -- without those keys it returns
# `(0.0, "no_visemes")` and `plan_dp`'s `lambda_phoneme` term is inert.
#
# `attach_visemes` extracts once per segment per side and writes both shapes.

VISEME_MAX_HOLD_S = 0.30


def _events_to_intervals(events: Sequence[VisemeEvent], *,
                         max_hold_s: float = VISEME_MAX_HOLD_S) -> List[dict]:
    """`[{start,end,viseme}]` from onset events, in the SAME time base.

    The interval is the HOLD -- this onset to the next -- not `e.dur`, for the
    same reason `_merge_runs` thresholds the hold: an `mms_char` event's `dur`
    is usually one 20 ms CTC frame, and emitting those slivers leaves
    `_viseme_at` returning None at nearly every sampled frame, so `_d_phoneme`
    scores ~0.0 while appearing to have run.

    A hold is capped at `max_hold_s` so the last shape before a pause is not
    painted across a second of silence; the final event falls back to its own
    span under the same cap.
    """
    out: List[dict] = []
    n = len(events)
    for i, e in enumerate(events):
        if i + 1 < n:
            hold = events[i + 1].t - e.t
        else:
            hold = e.dur
        hold = min(max(hold, 0.0), max_hold_s)
        if hold <= 0.0:
            continue
        out.append({"start": round(e.t, 4), "end": round(e.t + hold, 4),
                    "viseme": e.cls})
    return out


def attach_visemes(
    merged_json: Path,
    *,
    language: str,
    audio: Optional[Path] = None,
    backend: str = "mms_char",
    fallback_backend: Optional[str] = "allosaurus",
    model: str = "mms_fa",
    device: Optional[str] = None,
    audio_field: str = "tts_raw_audio",
    min_class_dur_s: float = 0.03,
    max_hold_s: float = VISEME_MAX_HOLD_S,
    sides: Sequence[str] = ("source", "tts"),
    out_json: Optional[Path] = None,
    work_dir: Optional[Path] = None,
) -> dict:
    """Write `src_visemes[]` and `tts_visemes[]` onto a merged JSON.

    TIME BASES, which are not the same and must not be mixed:

      `src_visemes[]`  ABSOLUTE seconds on the source timeline, offset by the
                       segment's `start`. Matches `words[]`, `slot`,
                       `src_speech`, `src_pauses`, and `Piece.dst_start` --
                       which is the base `_d_phoneme` samples `src` at.
      `tts_visemes[]`  CLIP-RELATIVE seconds, zero at the first sample of the
                       segment's raw TTS clip. Matches `tts_words[]` and
                       `Piece.src_span` -- which is the base `_d_phoneme`
                       samples `tts` at, after walking `dst_t` back through the
                       piece's own tempo mapping.

    Also writes `viseme_source[]` / `viseme_tts[]`, the same extraction as
    ONSET EVENTS (`{t,dur,cls,char,score}`). Both shapes come from one pass
    because the extraction costs ~5 s a segment on CPU.

    The source side is cut out of the full dialogue stem by `span`; the target
    side is the segment's own raw clip. `audio_field` defaults to
    `tts_raw_audio` for the reason given in `align_tts_segments`.

    Degrades per segment, never aborts the stage: a segment that yields
    nothing gets `[]` plus `qc["visemes_<side>"]` naming the reason. Reasons
    are `no_audio`, `audio_missing`, `no_text`, `too_short`
    (< `MIN_ALIGNABLE_S`) and `empty`. With `fallback_backend` set, the other
    backend is tried and the reason records the swap, e.g.
    `mms_char_empty->allosaurus`.

    A segment whose two sides came from different backends is comparing two
    inventories -- uroman'd orthography against IPA read off the audio -- so
    those ids are listed in `extra.visemes.mixed_backend_segments` and marked
    per segment in `qc["visemes_backend_mismatch"]`.

    One `mms_fa` aligner is built lazily and reused across every segment and
    both sides; `_allosaurus_events` keeps its recognizer in a module global.
    """
    for s in sides:
        if s not in ("source", "tts"):
            raise ValueError(f"side must be 'source' or 'tts', got {s!r}")
    if backend not in VISEME_BACKENDS:
        raise ValueError(f"unknown viseme backend {backend!r}; "
                         f"choose one of {VISEME_BACKENDS}")
    if fallback_backend is not None and fallback_backend not in VISEME_BACKENDS:
        raise ValueError(f"unknown fallback backend {fallback_backend!r}; "
                         f"choose one of {VISEME_BACKENDS} or None")

    import soundfile as sf

    merged_json = Path(merged_json)
    data = json.loads(merged_json.read_text())
    segments = data.get("segments", []) or []
    work_dir = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="visemes_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    src_audio = Path(audio) if audio else Path(data.get("audio_path", ""))
    aligner = None                      # one mms_fa instance, reused
    INTERVAL_KEY = {"source": "src_visemes", "tts": "tts_visemes"}
    EVENT_KEY = {"source": "viseme_source", "tts": "viseme_tts"}
    stats = {s: {"segments": 0, "events": 0, "failures": [],
                 "backends": {}} for s in sides}
    used_backend: Dict[str, Dict[str, str]] = {s: {} for s in sides}

    for seg in segments:
        sid = seg.get("id", "SEG_????")
        for side in sides:
            ikey, ekey = INTERVAL_KEY[side], EVENT_KEY[side]
            seg[ikey] = []
            seg[ekey] = []

            if side == "source":
                clip = src_audio
                t0, t1 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
                span: Optional[Tuple[float, float]] = (t0, t1)
                txt = seg.get("text") or ""
                offset = t0
                lang = seg.get("language") or language
                dur = t1 - t0
            else:
                c = seg.get(audio_field) or seg.get("tts_audio")
                clip = Path(c) if c else None
                span, txt, offset, lang = None, (
                    seg.get("final_text") or seg.get("mt_text") or ""), 0.0, language
                dur = None

            reason = None
            if clip is None:
                reason = "no_audio"
            elif not Path(clip).exists():
                reason = "audio_missing"
            elif not (txt or "").strip() and backend == "mms_char" \
                    and fallback_backend is None:
                # allosaurus needs no text, so missing text is only fatal
                # with no text-free backend to fall back to.
                reason = "no_text"
            else:
                if dur is None:
                    try:
                        dur = float(sf.info(str(clip)).duration)
                    except Exception as e:      # noqa: BLE001
                        log.warning("visemes[%s]: %s unreadable (%s)", side, sid, e)
                        reason = "audio_missing"
                if reason is None and dur is not None and dur < MIN_ALIGNABLE_S:
                    reason = "too_short"

            if reason:
                seg.setdefault("qc", {})[f"visemes_{side}"] = reason
                stats[side]["failures"].append({"id": sid, "reason": reason})
                continue

            if aligner is None and (backend == "mms_char"
                                    or fallback_backend == "mms_char"):
                aligner = _warm_aligner(model, device, None)

            order = [backend] + ([fallback_backend]
                                 if fallback_backend and fallback_backend != backend
                                 else [])
            evs: List[VisemeEvent] = []
            chosen = None
            for b in order:
                evs = extract_viseme_events(
                    clip, txt, model=model, language=lang, device=device,
                    aligner=aligner, clip_offset_s=offset,
                    min_class_dur_s=min_class_dur_s, backend=b,
                    span=span, work_dir=work_dir,
                )
                if evs:
                    chosen = b
                    break

            if not evs:
                why = "empty" if len(order) == 1 else \
                    f"{backend}_empty->{order[-1]}_empty"
                seg.setdefault("qc", {})[f"visemes_{side}"] = why
                stats[side]["failures"].append({"id": sid, "reason": why})
                continue

            seg[ekey] = [e.to_jsonable() for e in evs]
            seg[ikey] = _events_to_intervals(evs, max_hold_s=max_hold_s)
            used_backend[side][sid] = chosen
            stats[side]["segments"] += 1
            stats[side]["events"] += len(evs)
            stats[side]["backends"][chosen] = \
                stats[side]["backends"].get(chosen, 0) + 1
            if chosen != backend:
                why = f"{backend}_empty->{chosen}"
                seg.setdefault("qc", {})[f"visemes_{side}"] = why
                stats[side]["failures"].append({"id": sid, "reason": why})
            else:
                seg.get("qc", {}).pop(f"visemes_{side}", None)

    mixed = []
    if "source" in sides and "tts" in sides:
        for seg in segments:
            sid = seg.get("id", "SEG_????")
            a, b = used_backend["source"].get(sid), used_backend["tts"].get(sid)
            if a and b and a != b:
                mixed.append({"id": sid, "source": a, "tts": b})
                seg.setdefault("qc", {})["visemes_backend_mismatch"] = f"{a}|{b}"

    data.setdefault("extra", {})["visemes"] = {
        "backend": backend,
        "fallback_backend": fallback_backend,
        "model": model,
        "language": language,
        "audio_field": audio_field,
        "min_class_dur_s": min_class_dur_s,
        "max_hold_s": max_hold_s,
        "total_segments": len(segments),
        "time_base": {"src_visemes": "absolute",
                      "tts_visemes": "clip_relative"},
        "sides": {s: stats[s] for s in sides},
        "mixed_backend_segments": mixed,
    }
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    for s in sides:
        log.info("visemes[%s]: %d/%d segments, %d events, %d failures (%s)",
                 s, stats[s]["segments"], len(segments), stats[s]["events"],
                 len(stats[s]["failures"]),
                 stats[s]["backends"] or "none")
    if mixed:
        log.warning("visemes: %d segment(s) compared across two backends: %s",
                    len(mixed), ", ".join(m["id"] for m in mixed))
    return data


def _as_events(seq) -> List[VisemeEvent]:
    out = []
    for e in seq or []:
        out.append(e if isinstance(e, VisemeEvent) else VisemeEvent.from_jsonable(e))
    return out


def viseme_cost(
    src,
    tgt,
    *,
    class_weights: Optional[Dict[str, float]] = None,
    match_window_s: float = 0.20,
    skip_cost: float = 0.5,
    time_weight: float = 1.0,
    class_weight: float = 1.0,
    max_time_cost: float = 2.0,
    band_s: Optional[float] = 0.25,
    skip_disp_cap_s: Optional[float] = 3.0,
    norm_by: str = "max",
) -> dict:
    """DTW over (onset, class) between two viseme sequences.

    The sequences need not be the same length -- Hindi and English say the
    same thing with different numbers of mouth shapes. DTW lets one side spend
    more events in one place and charges only the resulting misalignment.

    Costs: a matched pair costs `|dt|/match_window_s`, capped at
    `max_time_cost` and forbidden beyond `band_s`, plus a class penalty scaled
    by the heavier of the two classes. A skipped event costs
    `skip_cost * weight[class] * (1 + min(d_near, skip_disp_cap_s)/win)`,
    where `d_near` is the distance to the nearest event in the other sequence.
    Both are class-weighted, so dropping a lip closure costs far more than a
    velar nobody can see. `viseme_cost(X, X) == 0`: the diagonal is free.

    Two settings keep the metric monotone in desync, and changing either
    brings back a score that goes DOWN as the desync grows:

      `norm_by="max"` divides by `max(n_src, n_tgt)`, a property of the
        inputs. Dividing by DTW path length lets a worse alignment grow the
        denominator as fast as the numerator and cancel its own cost.
      `skip_disp_cap_s` makes the skip penalty displacement-aware. With a
        fixed skip cost, skipping is cheaper than any pairing once `|dt|`
        passes about `2 * skip_cost * win`, so the total saturates and 2 s of
        desync scores the same as 1 s. An event with a counterpart nearby
        still skips cheaply, so different-length sequences remain allowed.

    It is NOT length-blind: dropping half the target's events while leaving
    every surviving onset untouched costs about what a 70 ms global shift
    does, so print `n_src`/`n_tgt` beside any number taken from here. Beyond
    `skip_disp_cap_s` the saturation returns by construction.

    `band_s=None, skip_disp_cap_s=None, norm_by="path"` reproduces the older
    metric exactly, for re-reading old numbers.
    """
    if norm_by not in ("max", "path"):
        raise ValueError(f"norm_by must be 'max' or 'path', got {norm_by!r}")
    w = dict(DEFAULT_CLASS_WEIGHTS)
    if class_weights:
        w.update(class_weights)
    a, b = _as_events(src), _as_events(tgt)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return {"dtw_cost": float("inf") if (n or m) else 0.0,
                "dtw_cost_norm": float("inf") if (n or m) else 0.0,
                "onset_mae_s": None, "bilabial_onset_mae_s": None,
                "matched": 0, "unmatched_src": n, "unmatched_tgt": m,
                "n_src": n, "n_tgt": m,
                "bilabial_matched": 0, "match_rate": 0.0,
                "band_s": band_s, "norm_by": norm_by}

    win = max(1e-6, match_window_s)

    def _skips(xs, ys) -> List[float]:
        if skip_disp_cap_s is None:
            return [skip_cost * w.get(e.cls, 1.0) for e in xs]
        out = []
        for e in xs:
            near = min((abs(e.t - o.t) for o in ys), default=skip_disp_cap_s)
            out.append(skip_cost * w.get(e.cls, 1.0)
                       * (1.0 + min(near, skip_disp_cap_s) / win))
        return out

    sa, sb = _skips(a, b), _skips(b, a)

    def pair_cost(i: int, j: int) -> float:
        dt = abs(a[i].t - b[j].t)
        if band_s is not None and dt > band_s:
            return float("inf")
        c = time_weight * min(dt / win, max_time_cost)
        if a[i].cls != b[j].cls:
            c += class_weight * max(w.get(a[i].cls, 1.0), w.get(b[j].cls, 1.0))
        return c

    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    # back[i][j]: 0 diag, 1 skip src, 2 skip tgt
    back = [[0] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + sa[i - 1]
        back[i][0] = 1
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + sb[j - 1]
        back[0][j] = 2
    for i in range(1, n + 1):
        wa = sa[i - 1]
        for j in range(1, m + 1):
            pc = pair_cost(i - 1, j - 1)
            diag = (D[i - 1][j - 1] + pc) if pc != INF else INF
            up = D[i - 1][j] + wa
            left = D[i][j - 1] + sb[j - 1]
            best = min(diag, up, left)
            D[i][j] = best
            back[i][j] = 0 if best == diag else (1 if best == up else 2)

    i, j = n, m
    deltas: List[float] = []
    bilabial: List[float] = []
    skipped_src = skipped_tgt = 0
    while i > 0 or j > 0:
        mv = back[i][j]
        if i > 0 and j > 0 and mv == 0:
            d = a[i - 1].t - b[j - 1].t
            deltas.append(abs(d))
            if a[i - 1].cls == "BILABIAL" and b[j - 1].cls == "BILABIAL":
                bilabial.append(abs(d))
            i, j = i - 1, j - 1
        elif i > 0 and (mv == 1 or j == 0):
            skipped_src += 1
            i -= 1
        else:
            skipped_tgt += 1
            j -= 1

    total = D[n][m]
    matched = len(deltas)
    den = (max(n, m) if norm_by == "max"
           else matched + skipped_src + skipped_tgt)
    return {
        "dtw_cost": round(total, 6),
        "dtw_cost_norm": round(total / max(1, den), 6),
        "onset_mae_s": round(sum(deltas) / matched, 4) if matched else None,
        "bilabial_onset_mae_s": (round(sum(bilabial) / len(bilabial), 4)
                                 if bilabial else None),
        "bilabial_matched": len(bilabial),
        "matched": matched,
        "unmatched_src": skipped_src,
        "unmatched_tgt": skipped_tgt,
        "n_src": n,
        "n_tgt": m,
        # Share of the richer sequence that found an in-band partner; says
        # whether `onset_mae_s` was computed over anything.
        "match_rate": round(matched / max(1, max(n, m)), 4),
        "band_s": band_s,
        "norm_by": norm_by,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
#
# A runnable call site for the viseme stage. To wire it into the pipeline:
# one `attach_visemes(mt_json, language=..., out_json=mt_json)` after stage
# 9f1 (the TTS clips and `tts_words[]` must exist) and before 9f2 (the timing
# plan reads the keys).

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m backend.align",
        description="Write src_visemes[]/tts_visemes[] onto a merged JSON.")
    ap.add_argument("merged_json", type=Path)
    ap.add_argument("--language", required=True,
                    help="target language of the TTS clips, e.g. hi")
    ap.add_argument("--audio", type=Path, default=None,
                    help="source dialogue stem; defaults to extra audio_path")
    ap.add_argument("--backend", default="mms_char", choices=VISEME_BACKENDS)
    ap.add_argument("--fallback-backend", default="allosaurus",
                    choices=(*VISEME_BACKENDS, "none"))
    ap.add_argument("--audio-field", default="tts_raw_audio")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-class-dur-s", type=float, default=0.03)
    ap.add_argument("--max-hold-s", type=float, default=VISEME_MAX_HOLD_S)
    ap.add_argument("--sides", default="source,tts")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="where to write; omit to rewrite the input in place")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    fb = None if args.fallback_backend == "none" else args.fallback_backend
    data = attach_visemes(
        args.merged_json,
        language=args.language,
        audio=args.audio,
        backend=args.backend,
        fallback_backend=fb,
        device=args.device,
        audio_field=args.audio_field,
        min_class_dur_s=args.min_class_dur_s,
        max_hold_s=args.max_hold_s,
        sides=tuple(s.strip() for s in args.sides.split(",") if s.strip()),
        work_dir=args.work_dir,
        out_json=args.out_json or args.merged_json,
    )
    print(json.dumps(data["extra"]["visemes"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
