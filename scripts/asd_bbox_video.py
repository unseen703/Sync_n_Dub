#!/usr/bin/env python
"""Re-score one job's ASD with a chosen model and render the boxed video.

The pipeline runs ASD once (loconet) and keeps only the per-segment
assignment, so `ui/bbox_video.py` can only ever re-draw that one decision.
To COMPARE models on the same footage you have to re-score, which is what
this does: it reuses the job's existing face tracks and source clip, runs
`backend.asd.score_asd` with `--asd-model`, and draws each face box coloured
by that model's per-frame speaking score (green >= threshold, red below) --
the exact `render_boxes` the ASD-gated driver uses, so the picture matches.

Nothing in the pipeline reads the output; it is a human comparison artifact.
Face detection is NOT re-run -- every model sees identical boxes, so any
difference in the render is the ASD model's, not the tracker's.

    $PY scripts/asd_bbox_video.py --job <job_id> --asd-model loconet

Inputs are discovered the way `ui/bbox_video.py` discovers them:
  video  videos/*.mp4   tracks  faces/*.npz   audio  audio/*.16k.mono.wav

ASD is driven by the ORIGINAL 16 kHz source audio (English here), never the
dub: it correlates speech with lip motion, and the dub is not lip-synchronous
with this footage.

Needs the libstdc++ preload like any faces/ASD run:
  export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("asd_bbox")


def _one(job_root: Path, pattern: str, what: str) -> Path:
    hits = sorted(job_root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"{what}: no match for {pattern} under {job_root}")
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="job id under <data>/jobs/")
    ap.add_argument("--asd-model", required=True,
                    help="loconet (the only registered ASD model)")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="Do NOT re-score. Load the job's existing "
                         "asd/<stem>.<model>.scores.npz -- the EXACT scores the "
                         "lip-sync run consumed (picked the same way "
                         "pipeline._asd_coord_list picks them) -- and only draw "
                         "them. Use this to see what actually drove the render; "
                         "omit it to re-score fresh for a model comparison.")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="score at/above which a box is drawn green")
    ap.add_argument("--video", default=None, help="override videos/*.mp4")
    ap.add_argument("--audio", default=None, help="override audio/*.16k.mono.wav")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(name)-10s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    job_root = Path(args.data_dir) / "jobs" / args.job
    if not job_root.is_dir():
        print(f"no such job: {job_root}", file=sys.stderr)
        return 2

    from backend.asd import (find_scores_npz, load_scores_npz,
                             save_scores_npz, score_asd)
    from models.faces import FaceTracks
    # The canonical per-frame-coloured renderer. Importing it (not copying)
    # keeps "which face is talking" drawn one way across this repo.
    from scripts.clip_asd_musetalk_run import render_boxes

    video = Path(args.video) if args.video else _one(job_root, "videos/*.mp4", "source clip")
    audio = Path(args.audio) if args.audio else _one(
        job_root, "audio/*.16k.mono.wav", "16 kHz source audio")
    npz = _one(job_root, "faces/*.npz", "face tracks")
    tracks = FaceTracks.load_npz(npz)
    log.info("job=%s model=%s video=%s audio=%s tracks=%d",
             args.job, args.asd_model, video.name, audio.name, len(tracks.tracks))

    out_dir = job_root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.reuse_scores:
        # The SAME file the lip-sync picked (find_scores_npz prefers loconet
        # then falls back; `prefer` pins the model asked for here).
        pick = find_scores_npz(job_root / "asd", video.stem, prefer=args.asd_model)
        if pick is None:
            print(f"--reuse-scores: no asd/{video.stem}.{args.asd_model}.scores.npz "
                  f"under {job_root/'asd'}; run the job's ASD first, or drop the "
                  f"flag to re-score.", file=sys.stderr)
            return 2
        asd = load_scores_npz(pick)
        scores_npz = pick
        log.info("reusing the run's scores: %s (model=%s, %d track(s))",
                 pick, asd.model_name, len(asd.per_track))
    else:
        device = f"cuda:{args.gpu}"
        asd = score_asd(video, audio, tracks, model=args.asd_model, device=device)
        scores_npz = out_dir / f"asd_scores.{args.asd_model}.npz"
        save_scores_npz(asd, scores_npz)
    wall = time.time() - t0

    summary = {
        "model": asd.model_name, "fps": asd.fps,
        "driving_audio": str(audio), "driving_audio_note": "16 kHz source, not the dub",
        "num_tracks": len(asd.per_track), "score_wall_s": round(wall, 1),
        "per_track": {
            tid: {"num_frames": int(sc.frames.size),
                  "mean": round(float(sc.scores.mean()), 4) if sc.scores.size else None,
                  "max": round(float(sc.scores.max()), 4) if sc.scores.size else None,
                  "frames_speaking_ge_thr": int((sc.scores >= args.threshold).sum())}
            for tid, sc in sorted(asd.per_track.items())},
    }
    (out_dir / f"asd_scores.{args.asd_model}.json").write_text(json.dumps(summary, indent=2))

    out_video = out_dir / f"asd_bbox.{args.asd_model}.mp4"
    stills = out_dir / f"asd_bbox_stills.{args.asd_model}"
    render_boxes(video, tracks, out_video, stills, asd=asd,
                 still_times=None, fps=int(round(asd.fps or 25)))

    size = out_video.stat().st_size if out_video.is_file() else 0
    print(f"\n{out_video}")
    print(f"  scores: {scores_npz}")
    print(f"  json:   {out_dir / f'asd_scores.{args.asd_model}.json'}")
    print(f"  {size / 1e6:.1f} MB · ASD scored in {wall:.1f}s · total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
