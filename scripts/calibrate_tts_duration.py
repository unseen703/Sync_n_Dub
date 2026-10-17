#!/usr/bin/env python3
"""Generate duration-calibration data and fit the per-(language, voice) model.

Why this exists: `backend/duration.fit_duration_model` has never had enough data
to run. Only 21 (text, measured-duration) pairs survive across all 285 runs on
disk, split across two engines -- so every manifest shows `n_fit: 0` and the
pipeline has always used the hand-guessed DEFAULT_MODEL. plan.md asks for ~200
utterances for a stable fit.

Rather than wait for more pipeline runs, synthesize a calibration set directly
from real Hindi text (the corpus's manual Hindi subtitles) and measure it. The
text is in-domain -- lecture and interview speech, the same register the
pipeline dubs -- which matters because the fit is a speaking-rate estimate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TS = re.compile(r"\d\d:\d\d:\d\d\.\d\d\d\s*-->")

#: The engine the pipeline speaks with, and so the only one worth
#: calibrating against. Needs BODHAN_TTS_KEY.
ENGINE = "bodhan"


def _cue_runs(vtts: list[Path], *, max_run: int = 400) -> list[list[str]]:
    """Consecutive cue texts per file, as runs of contiguous speech."""
    runs: list[list[str]] = []
    for p in vtts:
        cur: list[str] = []
        for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or "-->" in line:
                continue
            if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
                continue
            t = re.sub(r"<[^>]+>", "", line).strip()
            if t:
                cur.append(t)
            if len(cur) >= max_run:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
    return runs


def harvest_texts(vtts: list[Path], *, n: int, min_chars: int = 15,
                  max_chars: int = 650) -> list[str]:
    """Build length-stratified calibration texts spanning the REAL segment range.

    Single subtitle cues are far too short to calibrate on. Measured on this
    corpus: Hindi cues run 3-78 chars (median 28), while the segments the
    pipeline actually synthesises run 1-1196 chars (median 288). Fitting on cues
    and predicting segments is a 10x extrapolation, and the bias term -- which
    dominates at short lengths -- would be estimated entirely in the wrong
    regime.

    So join CONSECUTIVE cues into pseudo-segments. Consecutive cues are
    contiguous speech from one speaker, which is what a diarization turn is, so
    the joined text stays in-distribution rather than synthetic. Targets are
    drawn from log-spaced length bins, otherwise the corpus's natural short-cue
    mode dominates the fit.
    """
    runs = _cue_runs(vtts)
    if not runs:
        return []

    n_bins = 10
    edges = [min_chars * (max_chars / min_chars) ** (i / n_bins)
             for i in range(n_bins + 1)]
    per_bin = max(1, n // n_bins)
    out: list[str] = []
    seen: set[str] = set()

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        target = (lo + hi) / 2.0
        got = 0
        for r_i, run in enumerate(runs):
            if got >= per_bin:
                break
            # stride through each run so samples are spread over the source,
            # not all taken from its opening minute
            step = max(1, len(run) // max(1, per_bin))
            for start in range(r_i % step, len(run), step):
                if got >= per_bin:
                    break
                buf: list[str] = []
                total = 0
                for cue in run[start:]:
                    buf.append(cue)
                    total += len(cue) + 1
                    if total >= target:
                        break
                text = " ".join(buf).strip()
                if not (lo <= len(text) <= hi * 1.15) or len(text) > max_chars:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                out.append(text)
                got += 1
    return out


def synth(texts: list[str], voice: str, language: str,
          out_dir: Path) -> list[tuple[str, float]]:
    """Speak each text once and measure it, through the TTS registry.

    This used to call `edge_tts.Communicate` directly. It goes through
    `get_tts` now, for two reasons: Edge was removed on 2026-10-17, and the
    model being fitted is per-(language, VOICE) -- so calibrating with a
    different engine than the one the pipeline speaks with produces a
    confident number for the wrong speaking rate, which is worse than the
    hand-guessed DEFAULT_MODEL it is meant to replace.

    Resumable: a text whose WAV is already on disk is measured, not
    re-synthesised. Bodhan is rate-limited and a 300-text run will meet
    429s, so a re-run must fill gaps rather than start over.
    """
    import soundfile as sf

    from models.tts import get_tts
    from models.tts.base import TTSRequest

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, float]] = []
    todo: list[TTSRequest] = []
    for i, t in enumerate(texts):
        wav = out_dir / f"cal_{i:04d}.wav"
        if wav.exists():
            pairs.append((t, float(sf.info(str(wav)).duration)))
            continue
        todo.append(TTSRequest(id=f"cal_{i:04d}", text=t, voice=voice,
                               language=language))
    print(f"  {len(pairs)} already on disk, {len(todo)} to synthesise")
    if todo:
        tts = get_tts(ENGINE)
        res = tts.synthesize(todo, out_dir=out_dir, language=language,
                             sample_rate=24000)
        by_id = {r.id: r for r in res.segments}
        for req in todo:
            rec = by_id.get(req.id)
            if rec is None or rec.error:
                print(f"  !! {req.id}: {rec.error if rec else 'no result'}",
                      file=sys.stderr)
                continue
            pairs.append((req.text, float(rec.duration_s)))
    return pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", default="hi")
    # An indic-speak catalog name. `Kavya` is the lead female Hindi voice;
    # `backend/voices.py::DEFAULT_INDIC_SPEAK` has the other 44.
    ap.add_argument("--voice", default="Kavya")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--cache-dir", default="data/duration_model")
    ap.add_argument("--work-dir", default="data/duration_model/_calibration")
    a = ap.parse_args(argv)

    vtts = sorted((ROOT / "data" / "videos").glob(f"*/*.{a.language}.vtt"))
    if not vtts:
        print(f"no *.{a.language}.vtt in the corpus", file=sys.stderr)
        return 2
    texts = harvest_texts(vtts, n=a.n)
    print(f"harvested {len(texts)} texts from {len(vtts)} subtitle files "
          f"({min(map(len,texts))}-{max(map(len,texts))} chars)")

    work = ROOT / a.work_dir / f"{a.language}.{a.voice}"
    pairs = synth(texts, a.voice, a.language, work)
    print(f"synthesised {len(pairs)}/{len(texts)}")
    if len(pairs) < 30:
        print("too few samples to fit", file=sys.stderr)
        return 1

    from backend.duration import DEFAULT_MODEL, fit_duration_model
    model = fit_duration_model(pairs, language=a.language, voice=a.voice)

    # The fit is only worth shipping if it beats the hand-guessed default.
    import statistics
    base_mae = statistics.mean(abs(DEFAULT_MODEL.predict(t) - d) for t, d in pairs)
    print(f"\n  fitted : a_chars={model.a_chars:.5f} b_words={model.b_words:.5f} "
          f"c_bias={model.c_bias:.4f}")
    print(f"  n_fit  : {model.n_fit}")
    print(f"  MAE    : {model.mae_s:.3f}s   (DEFAULT_MODEL: {base_mae:.3f}s)")
    rel = (base_mae - model.mae_s) / base_mae * 100 if base_mae else 0.0
    mean_d = statistics.mean(d for _, d in pairs)
    print(f"  improvement over default: {rel:.1f}%")
    print(f"  MAE as % of mean duration ({mean_d:.2f}s): {model.mae_s/mean_d*100:.1f}%")

    cache = ROOT / a.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{a.language}.{a.voice}.json"
    dest.write_text(json.dumps(model.to_jsonable(), indent=2))
    # also key it as the pipeline's default voice slot, so a run that does not
    # name a voice still gets a fitted model instead of DEFAULT_MODEL
    (cache / f"{a.language}.default.json").write_text(json.dumps(model.to_jsonable(), indent=2))
    print(f"\nwrote {dest} (+ {a.language}.default.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
