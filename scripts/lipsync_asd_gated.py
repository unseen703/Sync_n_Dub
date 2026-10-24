#!/usr/bin/env python
"""Lip-sync one finished dub with ASD-gated per-frame targets (Route A).

The A/B partner of `scripts/lipsync_from_dub.py`, which renders the same job
with no targeting at all. Run both and compare the mp4s: same source, same
audio, same model, the only difference being WHICH FACE gets animated.

What it is for now
------------------
The pipeline does this itself since 2026-10-22: stage 6 writes
`asd/<stem>.<model>.scores.npz` (`backend/asd.py::save_scores_npz`), and both
lip-sync call sites — stage 9h in `run_post_mt_stages` and
`edit_agent._rerun_lipsync_only` — turn it into a `coord_list` via
`pipeline._asd_coord_list`. Without one, MuseTalk falls back to its own
detection: the first S3FD box (`d = d[0]`) with DWPose landmarks from a
whole-frame pose estimate that is never matched to that box, so on a
multi-face shot it may animate someone who is not talking.

This script is for the jobs that predate the npz. Their
`asd/<stem>.<model>.assignments.json` is the HUNGARIAN output — one row per
diarized turn with a single mean score — while the coord list needs per
track, per FRAME, so TalkNet is re-scored here. On a 37-second clip that is
seconds; on a long one it is not. For a job that HAS the npz, the pipeline's
own path is the one to use.

It also stays useful as the A/B against `scripts/lipsync_from_dub.py`, which
renders the same job with no targeting at all.

Only MuseTalk reads `coord_list`. Wav2Lip and LatentSync ignore it rather
than half-honouring it, so running this with `--model wav2lip` would
silently produce the same output as the un-gated script.
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

from dotenv import load_dotenv                                  # noqa: E402
load_dotenv(ROOT / ".env")

from backend.asd import score_asd                               # noqa: E402
from backend.lipsync import run_lipsync                         # noqa: E402
from backend.lipsync_targets import build_coord_list            # noqa: E402
from models.faces.base import FaceTracks                        # noqa: E402

log = logging.getLogger("lipsync_asd")


def _one(job: Path, sub: str, *suffixes: str) -> Path:
    """The single file in `job/sub` matching any suffix, or exit saying so."""
    for suf in suffixes:
        hits = sorted(p for p in (job / sub).glob(f"*{suf}") if p.is_file())
        if hits:
            return hits[0]
    raise SystemExit(f"no *{suffixes[0]} under {job / sub}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--model", default="musetalk",
                    help="musetalk only; the others ignore coord_list")
    ap.add_argument("--asd-model", default="talknet")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--bbox-shift", type=int, default=0)
    # `run_lipsync` names the output itself, so the two halves of the A/B
    # are kept apart by DIRECTORY. Writing both into `lipsync/` would mean
    # the second run silently overwrote the first — and overwrote the job's
    # existing render along with it.
    ap.add_argument("--out-dir", default=None,
                    help="where to write; defaults to <job>/lipsync_asd")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-22s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")

    job = Path(a.data_dir) / "jobs" / a.job_id
    if not job.is_dir():
        raise SystemExit(f"no such job: {job}")

    src = _one(job, "videos", ".mp4")
    dub = _one(job, "mix", ".dubbed.mp4")
    npz = _one(job, "faces", ".npz")

    # TWO different audio tracks, and using one for both is a silent wrong
    # answer rather than an error:
    #
    #   asd_wav   the SOURCE speech, 16 kHz mono. TalkNet decides who is
    #             talking by correlating lip motion with the audio, and the
    #             lips in this video are speaking the ORIGINAL language. Fed
    #             the dub instead, it correlates Hindi TTS against English
    #             mouth shapes and returns scores that mean nothing.
    #   drive_wav the DUB, which is what the mouth is being re-animated to.
    #
    # Stage 6 scores `dia_input` for exactly this reason, preferring the
    # dialogue stem (music and effects are not speech and drag the
    # correlation down), so this picks the same file in the same order.
    asd_wav = _one(job, "audio", ".dialogue.16k.mono.wav", ".16k.mono.wav")
    drive_wav = _one(job, "mix", ".dubbed.wav")

    log.info("source       %s", src.name)
    log.info("asd audio    %s  (source speech)", asd_wav.name)
    log.info("driving audio %s  (the dub)", drive_wav.name)
    log.info("face tracks  %s", npz.name)

    tracks = FaceTracks.load_npz(npz)
    log.info("tracks      %d", len(tracks.tracks))

    # Re-scored, not reloaded — see the module docstring.
    log.info("scoring ASD (%s) ...", a.asd_model)
    t0 = time.time()
    asd = score_asd(src, asd_wav, tracks, model=a.asd_model)
    log.info("asd done in %.0fs", time.time() - t0)

    coords, stats = build_coord_list(src, tracks, asd,
                                     bbox_shift=a.bbox_shift)
    # "It ran" and "it animated the right face" are different claims, and
    # only the second matters. `stats` is what separates them.
    log.info("targets: %s", json.dumps(stats, sort_keys=True))
    animated = sum(1 for b in coords if tuple(b) != (0.0, 0.0, 0.0, 0.0))
    log.info("coord list: %d frame(s), %d animated, %d passed through",
             len(coords), animated, len(coords) - animated)
    if not animated:
        raise SystemExit(
            "no frame has a confirmed speaking face, so every frame would be "
            "passed through unchanged. Nothing to render — check the ASD "
            "scores before blaming the lip-sync.")

    merged = job / "mt" / "lipsync_only.json"
    merged.parent.mkdir(parents=True, exist_ok=True)
    if not merged.exists():
        merged.write_text(json.dumps({"segments": [], "extra": {}}))

    out_dir = Path(a.out_dir) if a.out_dir else (job / "lipsync_asd")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    r = run_lipsync(merged, model=a.model, source_video=src,
                    driving_audio=drive_wav, pre_muxed_video=dub,
                    out_dir=out_dir, num_gpus=a.gpus,
                    coord_list=coords)
    dt = time.time() - t0

    out = Path(r.out_video) if r.out_video else None
    # The FILE is the arbiter, not the error field: an adapter can exit 0
    # having written nothing (CLAUDE.md §4).
    if out and out.is_file() and out.stat().st_size > 0:
        log.info("DONE in %.0fs -> %s", dt, out)
        return 0
    log.error("FAILED in %.0fs: %s", dt, r.error or "no output written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
