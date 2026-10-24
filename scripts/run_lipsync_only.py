"""Invoke a single lip-sync adapter on an already-dubbed mix output.

Usage:
    python scripts/run_lipsync_only.py --model wav2lip   --stem NAG_90to120
    python scripts/run_lipsync_only.py --model musetalk  --stem NAG_90to120
    python scripts/run_lipsync_only.py --model latentsync --stem NAG_90to120

There is no face-restoration pass: `models/restore/` was removed on
2026-10-17.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.lipsync import LipSyncRequest, get_lipsync


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=("wav2lip", "musetalk", "latentsync"))
    ap.add_argument("--stem", default="2vxQaT0Fy-c_1min",
                    help="Video stem under data/mix/<stem>.dubbed.{mp4,wav}")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=None,
                    help="Where renders land (default: <data-dir>/lipsync)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    )

    data = Path(args.data_dir)
    mix_mp4 = data / "mix" / f"{args.stem}.dubbed.mp4"
    mix_wav = data / "mix" / f"{args.stem}.dubbed.wav"
    out_dir = Path(args.out_dir) if args.out_dir else data / "lipsync"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.stem}.{args.model}.lipsync.mp4"

    if not mix_mp4.exists():
        print(f"ERROR: missing {mix_mp4}", file=sys.stderr)
        return 2
    if not mix_wav.exists():
        print(f"ERROR: missing {mix_wav}", file=sys.stderr)
        return 2

    kwargs = {}
    if args.model == "musetalk":
        kwargs["python_bin"] = "/data1/dipak_data/miniconda3/envs/musetalk/bin/python"
        # batch 20 leaves ~99 MiB headroom on a 16 GiB card; drop it past ~30 s.
        # fp16 is quality-neutral on the face crop and saves ~25% VRAM.
        kwargs["batch_size"] = int(os.environ.get("MUSETALK_BATCH", "20"))
        kwargs["use_float16"] = os.environ.get("FP16", "0") == "1"
    elif args.model == "latentsync":
        kwargs["python_bin"] = "/data1/dipak_data/miniconda3/envs/latentsync/bin/python"
    elif args.model == "wav2lip":
        # Unset by default since 2026-10-22, so the adapter's own ceiling
        # applies (native at or below 1080p, halved above). This used to
        # default the env var to "2", which passed an EXPLICIT 2 on every run
        # -- and an explicit value wins over the ceiling, so this script
        # alone would have kept halving 720p sources after the default
        # changed. Set WAV2LIP_RESIZE_FACTOR to pin it deliberately.
        if os.environ.get("WAV2LIP_RESIZE_FACTOR"):
            kwargs["resize_factor"] = int(os.environ["WAV2LIP_RESIZE_FACTOR"])
        kwargs["wav2lip_batch_size"] = int(os.environ.get("WAV2LIP_BATCH", "4"))
        kwargs["face_det_batch_size"] = int(os.environ.get("WAV2LIP_FD_BATCH", "4"))

    adapter = get_lipsync(args.model, **kwargs)
    req = LipSyncRequest(video=mix_mp4, audio=mix_wav, out_path=out)
    t0 = time.time()
    res = adapter.sync(req)
    dt = time.time() - t0
    print(f"model={res.model_name} out={res.out_video} fps={res.fps:.1f} "
          f"dur={res.duration_s:.1f}s wall={dt:.1f}s "
          f"error={res.error!r}")
    if res.error:
        return 1
    return 0
