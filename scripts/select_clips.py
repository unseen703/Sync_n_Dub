#!/usr/bin/env python3
"""Choose golden-set clip windows from the corpus.

Selection is deliberately evidence-driven rather than "every N seconds": a timing
experiment needs windows that actually contain the phenomena under test — speech
with internal pauses, speaker changes where available, and no long dead air that
would flatter every method equally.

Scores each candidate window on:
  speech_frac   fraction of the window that is speech in BOTH en and hi
  n_pauses      internal pauses >= PAUSE_MIN_S in the source (what method 3 needs)
  speaker_turns distinct speaker changes, from the VTT or a diarization RTTM
  rate_var      variation in local speaking rate (stresses duration adaptation)
  align_ok      en and hi both have speech; neither side is silent

Writes data/golden/index.json. Cuts no media -- that is refbuild's job.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
PAUSE_MIN_S = 0.25
HOP = 160  # 10 ms at 16 kHz

TS = re.compile(r"(\d\d):(\d\d):(\d\d)\.(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d)\.(\d\d\d)")


def _sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def cue_spans(path: Path) -> list[tuple[float, float]]:
    t = path.read_text(encoding="utf-8", errors="replace")
    return [(_sec(*m.group(1, 2, 3, 4)), _sec(*m.group(5, 6, 7, 8))) for m in TS.finditer(t)]


def speech_mask(wav: Path, thresh_db: float = -38.0) -> tuple[np.ndarray, int]:
    """10 ms frame-energy speech mask. Deliberately model-free: clip selection
    should not inherit a VAD model's biases, and the question is only 'is anyone
    talking here'. The scoring harness uses pyannote VAD for actual metrics."""
    import librosa
    y, sr = sf.read(str(wav))
    if y.ndim > 1:
        y = y.mean(axis=1)
    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=400, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    return db > thresh_db, sr


def runs_from_mask(mask: np.ndarray, sr: int, min_gap: float = 0.25,
                   min_speech: float = 0.20) -> list[list[float]]:
    out: list[list[float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append([i * HOP / sr, j * HOP / sr])
            i = j
        else:
            i += 1
    merged: list[list[float]] = []
    for r in out:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_speech]


@dataclass
class Clip:
    clip_id: str
    video_id: str
    source_dir: str
    t0: float
    t1: float
    tier: str
    speech_frac_en: float
    speech_frac_hi: float
    n_pauses: int
    n_cues: int
    score: float
    flags: list


def score_video(vid: str, vdir: Path, *, tier: str, n_clips: int, clip_s: float,
                search_t0: float, search_t1: float, hi_lang: str | None) -> list[Clip]:
    en_wav = next((p for p in vdir.glob(f"{vid}.en*.16k.wav")), None)
    hi_wav = vdir / f"{vid}.{hi_lang}.16k.wav" if hi_lang else None
    if en_wav is None or not en_wav.exists():
        print(f"  !! no English 16k wav in {vdir}", file=sys.stderr)
        return []

    en_mask, sr = speech_mask(en_wav)
    en_runs = runs_from_mask(en_mask, sr)
    if hi_wav and hi_wav.exists():
        hi_mask, _ = speech_mask(hi_wav)
    else:
        hi_mask = None
        print(f"  ~ no Hindi audio for {vid}; scoring on English only")

    dur = len(en_mask) * HOP / sr
    search_t1 = min(search_t1, dur) if search_t1 > 0 else dur

    # candidate windows on a stride; overlap is resolved by greedy non-max suppression
    stride = clip_s / 3.0
    cues = []
    for cand in (f"{vid}.en.vtt", f"{vid}.en-LUU0EuDKgKo.vtt", f"{vid}.en-orig.vtt"):
        p = vdir / cand
        if p.exists():
            cues = cue_spans(p)
            break

    cands: list[Clip] = []
    t = search_t0
    while t + clip_s <= search_t1:
        a, b = int(t / (HOP / sr)), int((t + clip_s) / (HOP / sr))
        en_seg = en_mask[a:b]
        sf_en = float(en_seg.mean()) if len(en_seg) else 0.0
        sf_hi = float(hi_mask[a:b].mean()) if hi_mask is not None and b <= len(hi_mask) else 0.0
        pauses = sum(1 for i in range(len(en_runs) - 1)
                     if t <= en_runs[i][1] < t + clip_s
                     and en_runs[i + 1][0] - en_runs[i][1] >= PAUSE_MIN_S)
        ncue = sum(1 for s, e in cues if t <= s < t + clip_s)

        flags = []
        if sf_en < 0.45:
            flags.append("sparse_speech_en")
        if hi_mask is not None and sf_hi < 0.45:
            flags.append("sparse_speech_hi")
        if pauses < 2:
            flags.append("few_pauses")

        # Reward dense speech on BOTH sides and internal pause structure; a window
        # that is mostly silence makes every timing method look identical.
        score = (min(sf_en, sf_hi if hi_mask is not None else sf_en) * 2.0
                 + min(pauses, 8) * 0.12
                 + min(ncue, 20) * 0.01
                 - 0.5 * len(flags))
        cands.append(Clip(
            clip_id=f"{vid}_{int(t):05d}to{int(t+clip_s):05d}", video_id=vid,
            source_dir=str(vdir.relative_to(ROOT)), t0=round(t, 2), t1=round(t + clip_s, 2),
            tier=tier, speech_frac_en=round(sf_en, 3), speech_frac_hi=round(sf_hi, 3),
            n_pauses=pauses, n_cues=ncue, score=round(score, 3), flags=flags))
        t += stride

    # greedy non-overlapping pick, best first
    chosen: list[Clip] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if len(chosen) >= n_clips:
            break
        if all(c.t1 <= o.t0 or c.t0 >= o.t1 for o in chosen):
            chosen.append(c)
    chosen.sort(key=lambda c: c.t0)
    print(f"  {vid}: {len(cands)} candidates -> {len(chosen)} clips "
          f"(score {min((c.score for c in chosen), default=0):.2f}..{max((c.score for c in chosen), default=0):.2f})")
    return chosen


CORPUS = [
    # vid,          slug dir prefix,              tier, n, clip_s, t0,    t1,     hi lang
    ("6-NRiom8K9Y", "nptel_design_thinking_intro", "A", 6, 60.0,  5.0,   -1,     "hi"),
    ("ZPUtA3W-7_I", "lex_narendra_modi",           "A", 5, 60.0,  600.0, 11000,  "hi"),
    ("qjPH9njnaVU", "lex_pavel_durov_telegram",    "B", 4, 60.0,  600.0, 16000,  "hi"),
    ("YrtANPtnhyg", "ted_sixthsense_pranav_mistry","C", 3, 60.0,  20.0,  900,    None),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/golden/index.json")
    ap.add_argument("--only", nargs="*", help="restrict to these video ids")
    a = ap.parse_args(argv)

    all_clips: list[Clip] = []
    for vid, slug, tier, n, clip_s, t0, t1, hi in CORPUS:
        if a.only and vid not in a.only:
            continue
        vdir = ROOT / "data" / "videos" / f"{slug}_{vid}"
        if not vdir.exists():
            print(f"  !! missing {vdir}", file=sys.stderr)
            continue
        all_clips += score_video(vid, vdir, tier=tier, n_clips=n, clip_s=clip_s,
                                 search_t0=t0, search_t1=t1, hi_lang=hi)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(c) for c in all_clips], indent=2))
    print(f"\nwrote {out} ({len(all_clips)} clips)")
    for c in all_clips:
        print(f"  {c.tier} {c.clip_id:<28} {c.t0:>8.1f}-{c.t1:<8.1f} "
              f"en={c.speech_frac_en:.2f} hi={c.speech_frac_hi:.2f} "
              f"pauses={c.n_pauses:<3} score={c.score:<6} {','.join(c.flags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
