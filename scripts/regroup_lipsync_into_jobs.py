#!/usr/bin/env python
"""Move flat `data/lipsync/` + `data/mix/` renders into per-language job folders.

`scripts/run_lipsync_only.py` writes to a shared `data/lipsync/` directory
(see its line 67), which predates the convention in CLAUDE.md §2 -- *one run,
one job folder under `data/jobs/<job_id>/`, nothing written outside it*. A
15-language sweep driven through that script therefore scatters 30 renders and
45 audio artifacts across two shared directories, while the MuseTalk driver for
the same sweep produces one self-contained folder per language.

This regroups the flat outputs so both halves of a sweep look the same on disk.
Files are MOVED, and the resulting layout mirrors what
`clip_asd_musetalk_run.py` writes:

    data/jobs/<ts>__<video>_<window>__<config>_<lang>/
        final.<lang>.wav2lip.mp4             raw lip-sync render
        audio/clip.dub.48k.stereo.wav        the generated dub track
        audio/clip.dub.cues.json             per-cue text + stretch sidecar
        videos/clip.dubbed.mp4               source clip carrying the dub
        config.json
        DUB_PROVENANCE.json

Idempotent: a language whose job folder already holds its final render is
skipped. Use --dry-run first.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem-base", required=True,
                    help="e.g. NAG_dIYDPtHWBNA_240to270")
    ap.add_argument("--job-prefix", required=True,
                    help="e.g. 20261015T012752__dIYDPtHWBNA_240to270__wav2lip_codeformer")
    ap.add_argument("--model", default="wav2lip")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--config-json", type=Path, default=None,
                    help="JSON file copied into each job folder as config.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = args.data_dir
    lipsync_dir, mix_dir, jobs_dir = data / "lipsync", data / "mix", data / "jobs"
    base_cfg = (json.loads(args.config_json.read_text())
                if args.config_json and args.config_json.exists() else {})

    # Languages are whatever actually rendered, read off disk rather than
    # assumed from a list -- a language that failed must not get a job folder
    # implying it succeeded.
    langs = sorted({
        p.name[len(args.stem_base) + 1:].split(".")[0]
        for p in lipsync_dir.glob(f"{args.stem_base}_*.{args.model}.lipsync.mp4")
    })
    if not langs:
        print(f"no renders matching {args.stem_base}_*.{args.model}.lipsync.mp4")
        return 1
    print(f"{len(langs)} language(s): {' '.join(langs)}\n")

    moved = skipped = 0
    for lg in langs:
        stem = f"{args.stem_base}_{lg}"
        job = jobs_dir / f"{args.job_prefix}_{lg}"
        final_raw = job / f"final.{lg}.{args.model}.mp4"
        if final_raw.exists():
            print(f"  {lg}: already regrouped, skipping")
            skipped += 1
            continue

        plan: list[tuple[Path, Path]] = [
            (lipsync_dir / f"{stem}.{args.model}.lipsync.mp4", final_raw),
            (mix_dir / f"{stem}.dubbed.wav", job / "audio" / "clip.dub.48k.stereo.wav"),
            (mix_dir / f"{stem}.dubbed.cues.json", job / "audio" / "clip.dub.cues.json"),
            (mix_dir / f"{stem}.dubbed.mp4", job / "videos" / "clip.dubbed.mp4"),
        ]
        present = [(s, d) for s, d in plan if s.exists()]
        missing = [s.name for s, _ in plan if not s.exists()]
        print(f"  {lg}: {len(present)} file(s) -> {job.name}"
              + (f"   MISSING: {missing}" if missing else ""))
        if args.dry_run:
            continue

        for src, dst in present:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

        cfg = dict(base_cfg)
        cfg.update({"language": lg, "job_id": job.name,
                    "lipsync_model": args.model})
        (job / "config.json").write_text(json.dumps(cfg, indent=1,
                                                    ensure_ascii=False))
        (job / "DUB_PROVENANCE.json").write_text(json.dumps({
            "audio_path": "generated",
            "pipeline": "YouTube en-orig auto-caption (de-rolled via inline word "
                        "timings) -> Bodhan indic-translate (target_script=codemix) "
                        "-> Bodhan indic-speak TTS -> per-cue rubberband fit",
            "not_used": "YouTube downloaded dub track",
            "language": lg,
            "regrouped_from": [f"data/lipsync/{stem}.*", f"data/mix/{stem}.*"],
            "note": "run_lipsync_only.py writes flat into data/lipsync/; this "
                    "folder was assembled afterwards to satisfy CLAUDE.md §2.",
        }, indent=1, ensure_ascii=False))
        moved += 1

    print(f"\nregrouped={moved} skipped={skipped}")
    if not args.dry_run:
        for d in (lipsync_dir, mix_dir):
            left = list(d.glob(f"{args.stem_base}_*"))
            if left:
                print(f"still in {d}: {[p.name for p in left]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
