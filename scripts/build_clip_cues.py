#!/usr/bin/env python
"""Build clean, cue-timed source text for one clip window.

Two sources, same output shape:

  * ``vtt-words`` -- a YouTube auto-caption ``*-orig.vtt``, read through
    ``parse_vtt_words``. This matters: YouTube serves its auto-captions in a
    *rolling* format where each cue repeats the previous cue's tail, so the
    plain ``parse_vtt`` text of this very file measures a 0.62 duplication
    ratio. Feeding that to TTS speaks every phrase twice. The inline word
    timestamps carry each word exactly once, so re-chunking from words is what
    de-rolls it.
  * ``asr`` -- run an ASR adapter over the window's audio. Used when no
    original-language caption exists on disk.

Words are regrouped into cues of at most ``--max-cue-s``, split at the largest
inter-word gaps so a cue boundary lands in a pause rather than mid-phrase. Cue
boundaries are what the lip-sync alignment ultimately rides on.

Output is a JSON file of cues rebased to the window (t=0 at ``--start``).

Usage:
    python -I scripts/build_clip_cues.py --source vtt-words \
        --vtt data/videos/.../dIYDPtHWBNA.en-orig.vtt \
        --start 240 --duration 30 --out data/clips/NAG_240to270.cues.en.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

log = logging.getLogger("build_cues")


def words_from_vtt(vtt: Path, start: float, end: float) -> list[tuple[str, float, float]]:
    from backend.transcripts import parse_vtt_words
    return [(w.text, w.start, w.end) for w in parse_vtt_words(vtt)
            if w.end > start and w.start < end]


def words_from_asr(audio: Path, model: str, language: str,
                   start: float) -> list[tuple[str, float, float]]:
    """Transcribe `audio` (already cut to the window) and return word timings.

    Times come back relative to the clip, which is already the window base, so
    nothing is subtracted here.
    """
    from models.asr import get_asr
    res = get_asr(model).transcribe(audio, language=language)
    words: list[tuple[str, float, float]] = []
    for seg in res.segments:
        if seg.words:
            words += [(w.text, w.start, w.end) for w in seg.words]
        else:
            # No word timings from this adapter: fall back to the segment as a
            # single unit rather than inventing per-word times.
            words.append((seg.text.strip(), seg.start, seg.end))
    return words


def chunk_words(words: list[tuple[str, float, float]], *,
                max_cue_s: float, base: float) -> list[dict]:
    """Group words into cues of at most `max_cue_s`, breaking at the longest gap."""
    cues: list[dict] = []
    cur: list[tuple[str, float, float]] = []

    def flush() -> None:
        if not cur:
            return
        s = cur[0][1] - base
        e = cur[-1][2] - base
        text = " ".join(w[0].strip() for w in cur if w[0].strip())
        if text and e > s:
            cues.append({"start": round(max(0.0, s), 3), "end": round(e, 3),
                         "slot_s": round(e - max(0.0, s), 3), "text": text})

    for w in words:
        if cur and (w[2] - cur[0][1]) > max_cue_s:
            # Break at the biggest pause inside the span we have accumulated,
            # so the cut lands between phrases instead of inside one.
            gaps = [(cur[i + 1][1] - cur[i][2], i) for i in range(len(cur) - 1)]
            if gaps:
                best_gap, idx = max(gaps)
                if best_gap > 0.12 and idx >= 1:
                    tail = cur[idx + 1:]
                    del cur[idx + 1:]
                    flush()
                    cur = tail
                else:
                    flush(); cur = []
            else:
                flush(); cur = []
        cur.append(w)
    flush()
    return cues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=("vtt-words", "asr"))
    ap.add_argument("--vtt", type=Path)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--ref-audio", type=Path, default=None,
                    help="Source-language audio for this window, recorded into "
                         "the cue JSON so the dub stage can predict the "
                         "speaker's gender from it. Defaults to --audio.")
    ap.add_argument("--asr-model", default="bodhan")
    ap.add_argument("--language", default="en")
    ap.add_argument("--start", required=True, type=float)
    ap.add_argument("--duration", required=True, type=float)
    ap.add_argument("--max-cue-s", type=float, default=4.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    end = args.start + args.duration

    if args.source == "vtt-words":
        if not args.vtt:
            ap.error("--source vtt-words needs --vtt")
        from backend.transcripts import has_inline_word_timings, rolling_duplication_ratio
        if not has_inline_word_timings(args.vtt):
            log.error("%s has no inline word timings -- its cue text is rolling "
                      "and would be spoken twice. Use --source asr instead.",
                      args.vtt)
            return 2
        log.info("rolling duplication ratio of plain cue text: %.3f "
                 "(de-rolled by reading word timings)",
                 rolling_duplication_ratio(args.vtt))
        words = words_from_vtt(args.vtt, args.start, end)
        base = args.start
    else:
        if not args.audio:
            ap.error("--source asr needs --audio")
        words = words_from_asr(args.audio, args.asr_model, args.language,
                               args.start)
        base = 0.0

    if not words:
        log.error("no words in window %.1f-%.1fs", args.start, end)
        return 2

    cues = chunk_words(words, max_cue_s=args.max_cue_s, base=base)
    # Clamp to the window: a word straddling the edge must not push a cue past
    # the clip, or the assembled track runs longer than the video.
    for c in cues:
        c["end"] = round(min(c["end"], args.duration), 3)
        c["slot_s"] = round(max(0.0, c["end"] - c["start"]), 3)
    cues = [c for c in cues if c["slot_s"] >= 0.05]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": args.source,
        "source_file": str(args.vtt or args.audio),
        # Carried so `clip_dub_audio.py` can measure the speaker's gender
        # instead of defaulting it. A defaulted gender is how a male voice
        # was once dubbed over a female speaker in all 15 languages.
        "ref_audio": str(args.ref_audio or args.audio) if (args.ref_audio or args.audio) else None,
        "asr_model": args.asr_model if args.source == "asr" else None,
        "language": args.language,
        "window_abs_s": [args.start, end],
        "duration_s": args.duration,
        "max_cue_s": args.max_cue_s,
        "n_words": len(words), "n_cues": len(cues),
        "cues": cues,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    log.info("%d words -> %d cues (%.1f-%.1fs) -> %s",
             len(words), len(cues), args.start, end, args.out)
    for c in cues:
        log.info("  %6.2f-%6.2f (%4.1fs) %s", c["start"], c["end"],
                 c["slot_s"], c["text"][:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
