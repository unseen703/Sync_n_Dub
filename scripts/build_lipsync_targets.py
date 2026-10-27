"""Build the per-frame face targets LatentSync needs, from a job's stage-6 work.

Stage 6 (`faces/*.npz`) already says where every face is and which track it
belongs to; stage 7 (`asd/*.scores.npz`) already says who is speaking. What
neither carries is the 106 landmarks LatentSync's warp is built from -- its
own detector produces those, and in doing so picks whichever face is largest,
which on a two-person shot is routinely the listener.

This fills that gap: it re-runs ONLY the landmark head, on the box the ASD
gating chose, and writes an artifact the renderer can consume.

    $PY scripts/build_lipsync_targets.py -j <job_id>
    $PY scripts/build_lipsync_targets.py -j <job_id> --window 20 80
    $PY scripts/build_lipsync_targets.py -j <job_id> --out targets/clip.npz

**Build against the native-resolution source, not the capped clip.** The
landmark head crops to a fixed 192x192 whatever it is handed, so a face that
is 160x200 px in 4K arrives near-native while the same face at 1080p arrives
2x interpolated. The artifact records the geometry it measured in and the
renderer scales on read, so detecting big and rendering small is both correct
and better than the reverse.

`--window` cuts the result to a frame range rather than re-detecting, so a
clip rendered from a time slice of the job video gets targets that line up
with it. Give it the SAME seconds the clip was cut at.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))


def _load_asd(path: Path):
    """Rebuild the per-track score object `build_targets` expects.

    `backend.asd` writes a flat (frames, scores, offsets) layout and ships no
    reader, so the shape is reconstructed here rather than re-scoring -- which
    would cost a TalkNet pass to recover numbers already on disk.
    """
    import numpy as np

    d = np.load(path, allow_pickle=True)
    tids = [str(x) for x in d["track_ids"]]
    offs, frames, scores = d["offsets"], d["frames"], d["scores"]

    class _Score:
        def __init__(self, f, s):
            self.frames, self.scores = f, s

    class _ASD:
        pass

    asd = _ASD()
    asd.per_track = {
        t: _Score(frames[offs[i]:offs[i + 1]], scores[offs[i]:offs[i + 1]])
        for i, t in enumerate(tids)
    }
    return asd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--job-id", required=True)
    ap.add_argument("--video", default=None,
                    help="source video; defaults to the job's videos/*.mp4. "
                         "Use the NATIVE-resolution file, not a capped clip.")
    ap.add_argument("--faces", default=None, help="stage-6 tracks .npz")
    ap.add_argument("--asd", default=None, help="stage-7 scores .npz")
    ap.add_argument("--asd-model", default="loconet",
                    help="when --asd is not given, prefer this model's "
                         "scores from the job's asd/ dir (default loconet)")
    ap.add_argument("--out", default=None,
                    help="output .npz; defaults to targets/<stem>.targets.npz")
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_S", "END_S"),
                    help="cut the result to this time range instead of "
                         "re-detecting, for a clip cut from the same source")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ASD speaking threshold (default 0.5)")
    ap.add_argument("--min-run", type=int, default=None,
                    help="drop speaking runs shorter than this (default 5)")
    ap.add_argument("--max-gap", type=int, default=None,
                    help="fill gaps up to this long inside a run (default 12)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-22s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(ROOT / f"logs/{a.job_id}.log", mode="a"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("build_targets")

    from backend.lipsync_targets import (build_targets, save_targets,
                                         slice_targets,
                                         ASD_SPEAK_THRESHOLD, MIN_SPEAK_RUN,
                                         MAX_GAP_FILL)

    job = ROOT / "data/jobs" / a.job_id
    if not job.is_dir():
        log.error("no such job: %s", job)
        return 2

    def _pick(explicit, folder, pattern, what):
        if explicit:
            p = Path(explicit)
            return p if p.is_file() else None
        hits = sorted((job / folder).glob(pattern))
        if len(hits) > 1:
            log.info("%s: %d candidates, using %s", what, len(hits), hits[0].name)
        return hits[0] if hits else None

    video = _pick(a.video, "videos", "*.mp4", "video")
    faces = _pick(a.faces, "faces", "*.npz", "faces")
    # Select the scores by MODEL (loconet first), not alphabetical glob, so the
    # LatentSync targets are built from the same model the pipeline defaults to.
    if a.asd:
        asd_p = Path(a.asd) if Path(a.asd).is_file() else None
    else:
        from backend.asd import find_scores_npz
        asd_p = find_scores_npz(job / "asd",
                                video.stem if video else None,
                                prefer=a.asd_model)
        if asd_p is not None:
            log.info("asd: preferring %s -> %s", a.asd_model, asd_p.name)
    missing = [n for n, p in (("video", video), ("faces", faces), ("asd", asd_p))
               if p is None]
    if missing:
        log.error("missing %s under %s -- stage 6 and 7 must have run",
                  ", ".join(missing), job)
        return 2

    log.info("video %s", video)
    log.info("faces %s", faces.name)
    log.info("asd   %s", asd_p.name)

    from models.faces.base import FaceTracks
    tracks = FaceTracks.load_npz(faces)
    asd = _load_asd(asd_p)
    log.info("tracks %d, scored %d, source %dx%d",
             len(tracks.tracks), len(asd.per_track), tracks.width, tracks.height)

    t0 = time.time()
    payload, stats = build_targets(
        video, tracks, asd, device=a.device,
        threshold=a.threshold if a.threshold is not None else ASD_SPEAK_THRESHOLD,
        min_run=a.min_run if a.min_run is not None else MIN_SPEAK_RUN,
        max_gap=a.max_gap if a.max_gap is not None else MAX_GAP_FILL,
    )
    log.info("built in %.0fs: %s", time.time() - t0,
             json.dumps({k: v for k, v in stats.items() if k != "track_ids"},
                        default=str))

    stem = video.stem
    if a.window:
        fps = float(payload["fps"]) or tracks.fps or 25.0
        f0, f1 = int(round(a.window[0] * fps)), int(round(a.window[1] * fps))
        payload = slice_targets(payload, f0, f1)
        stem = f"{stem}_{int(a.window[0])}to{int(a.window[1])}"
        import numpy as np
        ft = payload["frame_track"]
        log.info("window %.1f-%.1fs at %.3f fps -> frames %d-%d: "
                 "%d frame(s), %d targeted, %d run(s)",
                 a.window[0], a.window[1], fps, f0, f1, ft.shape[0],
                 int((ft >= 0).sum()), int(payload["run_start"].sum()))

    out = Path(a.out) if a.out else (job / "targets" / f"{stem}.targets.npz")
    save_targets(payload, out)
    log.info("wrote %s (%d bytes)", out, out.stat().st_size)
    log.info("render with:  --targets %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
