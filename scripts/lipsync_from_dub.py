"""Lip-sync a clip against a dub that ALREADY EXISTS. No ASR, MT or TTS.

The pipeline's stage 9h is the only part this runs. Everything upstream --
diarization, ASR, translation, voice casting, TTS, mix -- is read off disk
from a finished dub, because re-deriving it costs a full round of hosted
Bodhan calls to produce audio that is already correct.

    $PY scripts/lipsync_from_dub.py \\
        --source data/clips/iit_1080p_4min.mp4 \\
        --dub    data/clips/seg30_2026-10-20/joined/iit_dub_joined.mp4 \\
        --job-id iit4min_lipsync --gpus 2

`--dub` is a video whose audio track is the dubbed speech. It serves as both
the driving audio (extracted here) and the pre-muxed video, which is the
pair `run_lipsync` wants: the adapter prefers a video that already carries
the right soundtrack, so the output keeps it even if the adapter re-strips
and re-attaches audio.

The source and the dub must agree on frame count. They will when the dub was
built from that exact source -- but a dub joined from re-encoded segments can
come back at a different nominal rate, and a mismatch here is the difference
between a lip-sync and a slow desync. It is checked rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))

# -13% wall, -25% peak VRAM, face crop unchanged (34.45 vs 34.46 dB).
os.environ.setdefault("MUSETALK_FP16", "1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="original video")
    ap.add_argument("--dub", required=True,
                    help="video carrying the DUBBED audio track")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--model", default="musetalk")
    # Kept apart from the ASD-gated run by DIRECTORY: `run_lipsync` names
    # the output file itself, so two renders of one job into `lipsync/`
    # would collide and the second would silently win.
    ap.add_argument("--out-dir", default=None,
                    help="where to write; defaults to <job>/lipsync")
    ap.add_argument("--gpus", type=int, default=2,
                    help="time-shard across this many cards. Each shard does "
                         "its own face-detection smoothing, so the join can "
                         "show as a one-frame mouth jump; 1 is cleaner for a "
                         "final render.")
    a = ap.parse_args()

    job = ROOT / "data/jobs" / a.job_id
    (job / "lipsync").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-24s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(ROOT / f"logs/{a.job_id}.log", mode="a"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("lipsync_from_dub")

    from backend.lipsync import run_lipsync, _frame_count
    from backend.media_io import _ffmpeg_bin
    from backend.gpu import free_devices, visible_cuda_devices

    src, dub = Path(a.source).resolve(), Path(a.dub).resolve()
    for p in (src, dub):
        if not p.is_file():
            log.error("missing: %s", p)
            return 2

    s_n, d_n = _frame_count(src), _frame_count(dub)
    log.info("source %s: %d frames", src.name, s_n)
    log.info("dub    %s: %d frames", dub.name, d_n)
    if s_n and d_n and s_n != d_n:
        # Not fatal -- run_lipsync conforms a mismatch -- but it is the one
        # thing that silently turns a lip-sync into a drifting one, so it is
        # said out loud before an hour of GPU time rather than found after.
        log.warning("frame counts DIFFER by %d; the render will be conformed "
                    "to the source, but check the dub was built from this "
                    "exact file", d_n - s_n)

    log.info("gpus: visible=%s free=%s requested=%d",
             visible_cuda_devices(), free_devices(), a.gpus)

    # The dubbed speech on its own, as the driving audio.
    wav = job / "lipsync" / f"{src.stem}.dub.wav"
    if not wav.is_file():
        ff = _ffmpeg_bin() or "ffmpeg"
        subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(dub), "-vn", "-ac", "2", "-ar", "48000",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        log.info("extracted driving audio -> %s", wav.name)

    merged = job / "lipsync_only.json"
    if not merged.exists():
        merged.write_text(json.dumps({"segments": [], "extra": {}}))

    t0 = time.time()
    out_dir = Path(a.out_dir) if a.out_dir else (job / "lipsync")
    out_dir.mkdir(parents=True, exist_ok=True)
    r = run_lipsync(merged, model=a.model, source_video=src,
                    driving_audio=wav, pre_muxed_video=dub,
                    out_dir=out_dir, num_gpus=a.gpus)
    dt = time.time() - t0

    out = Path(r.out_video) if r.out_video else None
    # The FILE is the arbiter, not the error field: an adapter can exit 0
    # having written nothing (CLAUDE.md section 4).
    if out and out.is_file() and out.stat().st_size > 0:
        log.info("DONE in %.0fs -> %s (%d frames, source %d)",
                 dt, out, _frame_count(out), s_n)
        return 0
    log.error("FAILED in %.0fs: %s", dt, r.error or "no output written")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
