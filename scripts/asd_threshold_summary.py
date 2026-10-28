#!/usr/bin/env python
"""How many frames pass the speaking threshold, per job/model, at 0.3/0.5/0.7.

Reads the per-frame scores the bbox driver saved
(`review/asd_scores.<model>.npz`) and counts against them. Thresholding is a
post-process: the ASD run does not depend on the threshold, so the same saved
scores answer all three cutoffs -- there is nothing to re-run per threshold.

Two counts, because "frames that pass" is ambiguous with several faces:

  track-frames >= t   every (face, frame) scored at/above t, summed over faces.
                      A frame with two speaking faces counts twice. This is the
                      raw volume a gate at t would fire on.
  active frames >= t   distinct VIDEO frames with at least one face at/above t --
                      "a speaker is on screen here". The denominator is the
                      frames any tracked face appears in (covered frames).

    $PY scripts/asd_threshold_summary.py --jobs <id> <id> ... \\
        --out logs/asd_threshold_summary.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: One model since 2026-10-28; the list shape stays so a second one can be
#: compared against it again without reworking the script.
MODELS = ["loconet"]
THRESHOLDS = [0.3, 0.5, 0.7]


def summarise(npz_path: Path, thresholds):
    import numpy as np
    from backend.asd import load_scores_npz

    res = load_scores_npz(npz_path)
    track_total = 0
    tf_ge = {t: 0 for t in thresholds}
    by_frame: dict[int, float] = {}
    for sc in res.per_track.values():
        s = np.asarray(sc.scores, dtype=float)
        f = np.asarray(sc.frames, dtype=int)
        track_total += s.size
        for t in thresholds:
            tf_ge[t] += int((s >= t).sum())
        for fr, v in zip(f.tolist(), s.tolist()):
            if v > by_frame.get(fr, -1.0):
                by_frame[fr] = v
    covered = len(by_frame)
    active_ge = {t: sum(1 for v in by_frame.values() if v >= t)
                 for t in thresholds}
    return {
        "model": res.model_name, "fps": res.fps,
        "n_tracks": len(res.per_track),
        "track_total": track_total, "tf_ge": tf_ge,
        "covered": covered, "active_ge": active_ge,
    }


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", nargs="+", required=True, help="job ids under <data>/jobs/")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS)
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args(argv)

    thr = args.thresholds
    lines: list[str] = []

    def emit(s=""):
        print(s)
        lines.append(s)

    hdr_tf = " | ".join(f"tf>={t}" for t in thr)
    hdr_af = " | ".join(f"act>={t}" for t in thr)
    for job in args.jobs:
        job_root = Path(args.data_dir) / "jobs" / job
        short = job.split("__")[1] if "__" in job else job
        emit(f"\n### {short}  `{job}`\n")
        emit(f"| model | tracks | track-frames | {hdr_tf} | covered frames | {hdr_af} |")
        emit("|---|--:|--:|" + "--:|" * len(thr) + "--:|" + "--:|" * len(thr))
        for m in args.models:
            npz = job_root / "review" / f"asd_scores.{m}.npz"
            if not npz.is_file():
                emit(f"| {m} | missing ({npz.name}) |" + " |" * (2 + 2 * len(thr)))
                continue
            r = summarise(npz, thr)
            tf = " | ".join(f"{r['tf_ge'][t]} ({pct(r['tf_ge'][t], r['track_total'])})"
                            for t in thr)
            af = " | ".join(f"{r['active_ge'][t]} ({pct(r['active_ge'][t], r['covered'])})"
                            for t in thr)
            emit(f"| {m} | {r['n_tracks']} | {r['track_total']} | {tf} "
                 f"| {r['covered']} | {af} |")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
