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
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="diffusion denoising steps. LatentSync only -- the "
                         "registry drops it for models that do not take it. "
                         "The adapter's own default is 20; 10 halves the UNet "
                         "passes and so roughly halves wall time, because "
                         "LatentSync is UNet-bound rather than startup-bound.")
    ap.add_argument("--deepcache", action="store_true",
                    help="LatentSync only. Reuse the UNet's deep-block output "
                         "between denoising steps, recomputing it every 3rd. "
                         "Costs ~20 MiB of VRAM and nothing in host RAM. It "
                         "attacks the same per-step cost as --inference-steps, "
                         "so the two MULTIPLY: at 10 steps only 4 full passes "
                         "survive. Compare against the plain run before "
                         "trusting it on a final render.")
    ap.add_argument("--memory-efficient", action="store_true",
                    help="LatentSync only. Composite the rendered faces "
                         "straight into the encoder instead of building the "
                         "whole output clip in RAM first. Peak host memory "
                         "per shard drops roughly 18 GB -> 9 GB at 1080p. "
                         "The pixels are IDENTICAL -- unlike the speed knobs "
                         "this trades nothing away.")
    ap.add_argument("--max-segment-s", type=float, default=None,
                    help="Cap how much video is resident at once. Longer clips "
                         "are cut into pieces of seg/num_gpus each and rendered "
                         "one warm worker per GPU -- the model loads once per "
                         "card, and the pieces are sized so the cards running "
                         "together hold ~seg's worth of frames, not more. "
                         "run_lipsync's own default is 60, measured at 1080p. A "
                         "4K frame is 23.7 MiB against 1080p's 5.9, so the same "
                         "60s costs 4x the RAM -- use 30 or less on a 2160p "
                         "source.")
    ap.add_argument("--max-height", type=int, default=1080,
                    help="Downscale source and dub to this many scanlines "
                         "before rendering. A CEILING, never an upscale, so "
                         "1080p and below pass through untouched. 0 disables. "
                         "The lip-sync models rebuild the mouth at 256x256 "
                         "regardless, so a 4K source spends 4x the decode, "
                         "face-detection, compositing and encode cost to "
                         "carry pixels the UNet never sees. Matches the "
                         "ceiling wav2lip already applies.")
    ap.add_argument("--targets", default=None,
                    help="Per-frame face targets .npz (see "
                         "backend.lipsync_targets.build_targets). LatentSync "
                         "only. Decides WHICH face is animated: without it "
                         "the stock detector takes the largest face per "
                         "frame, so on a two-person shot it can animate the "
                         "listener and flip between them. Sliced per shard "
                         "and per segment automatically. Coordinates may be "
                         "in a different resolution than the render -- the "
                         "geometry is recorded in the file and scaled on "
                         "read, so targets built on a 4K source drive a "
                         "1080p render correctly.")
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

    # Resolution ceiling, applied ONCE to the whole clip rather than per
    # shard: the shards are cut from these files, so capping here means one
    # ffmpeg pass instead of one per shard, and every later stage -- decode,
    # face detection, the compositing warp, the final encode -- sees the
    # smaller frame. `cap_dims` is wav2lip's; importing it keeps the two
    # models' ceilings from drifting apart.
    if a.max_height and a.max_height > 0:
        from models.lipsync.wav2lip_adapter import cap_dims
        from backend.lipsync import _video_dims

        dims = _video_dims(src)
        if dims and dims[1] > a.max_height:
            capped = cap_dims(dims, a.max_height)
            log.info("capping %dx%d -> %dx%d (%.1fx fewer pixels per frame)",
                     dims[0], dims[1], capped[0], capped[1],
                     (dims[0] * dims[1]) / float(capped[0] * capped[1]))
            cap_dir = job / "clips"
            cap_dir.mkdir(parents=True, exist_ok=True)
            ff = _ffmpeg_bin() or "ffmpeg"
            scaled = []
            for p in (src, dub):
                out_p = cap_dir / f"{p.stem}.{capped[1]}p{p.suffix}"
                if not out_p.is_file():
                    subprocess.run(
                        [ff, "-y", "-hide_banner", "-loglevel", "error",
                         "-i", str(p),
                         "-vf", f"scale={capped[0]}:{capped[1]}:flags=lanczos",
                         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                         "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-ar", "48000", "-ac", "2",
                         "-b:a", "192k", str(out_p)], check=True)
                    log.info("  wrote %s", out_p.name)
                else:
                    log.info("  reusing %s", out_p.name)
                scaled.append(out_p)
            src, dub = scaled
        elif dims:
            log.info("source is %dx%d, at or under the %dp ceiling -- "
                     "no downscale", dims[0], dims[1], a.max_height)

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
    ls_kwargs = {}
    if a.inference_steps is not None:
        ls_kwargs["inference_steps"] = a.inference_steps
        log.info("inference_steps=%d (adapter default is 20)",
                 a.inference_steps)
    if a.deepcache:
        ls_kwargs["enable_deepcache"] = True
        log.info("deepcache=ON (full UNet pass every 3rd step)")
    if a.memory_efficient:
        ls_kwargs["memory_efficient"] = True
        log.info("memory_efficient=ON (compositing streams into the encoder)")

    seg_kwargs = {}
    if a.max_segment_s is not None:
        seg_kwargs["max_segment_s"] = a.max_segment_s
        log.info("max_segment_s=%.0f (run_lipsync default is 60)",
                 a.max_segment_s)
    if a.targets:
        tp = Path(a.targets).resolve()
        if not tp.is_file():
            log.error("targets file not found: %s", tp)
            return 2
        # Report what it covers BEFORE the render: a targets file that does
        # not line up with this clip falls back to stock detection silently
        # by design, and "it ran" would then look identical to "it animated
        # the right face".
        from backend.lipsync_targets import load_targets_payload
        tpay = load_targets_payload(tp)
        if tpay is None:
            log.error("targets file unreadable: %s", tp)
            return 2
        import numpy as _np
        _ft = _np.asarray(tpay["frame_track"])
        log.info("targets: %s -- %d frame(s), %d targeted, %d run(s), "
                 "built at %dx%d", tp.name, _ft.shape[0],
                 int((_ft >= 0).sum()), int(_np.asarray(tpay["run_start"]).sum()),
                 int(tpay["width"]), int(tpay["height"]))
        if _ft.shape[0] != s_n:
            log.warning("targets cover %d frame(s) but the source has %d; "
                        "pieces whose slice does not line up will render "
                        "with stock detection", _ft.shape[0], s_n)
        seg_kwargs["targets_npz"] = tp

    r = run_lipsync(merged, model=a.model, source_video=src,
                    driving_audio=wav, pre_muxed_video=dub,
                    out_dir=out_dir, num_gpus=a.gpus,
                    lipsync_kwargs=ls_kwargs or None, **seg_kwargs)
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
