"""Invoke a single lip-sync adapter on an already-dubbed mix output.

Usage:
    python scripts/run_lipsync_only.py --model wav2lip   --stem NAG_90to120
    python scripts/run_lipsync_only.py --model musetalk  --stem NAG_90to120
    python scripts/run_lipsync_only.py --model latentsync --stem NAG_90to120

Add `--restore codeformer` to run a face-restoration pass over the
render — the Wav2Lip-CodeFormer arrangement. Worth it for Wav2Lip, whose
96x96 mouth crop comes back soft; the restored video is written beside
the raw one as `<stem>.<model>.<restore>.mp4`, so the two stay
comparable.
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
from models.restore import RestoreRequest, get_restore
from models.restore import available_models as available_restore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=("wav2lip", "musetalk", "latentsync", "lipgan"))
    ap.add_argument("--stem", default="2vxQaT0Fy-c_1min",
                    help="Video stem under data/mix/<stem>.dubbed.{mp4,wav}")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=None,
                    help="Where renders land (default: <data-dir>/lipsync)")
    ap.add_argument("--restore", default=None,
                    choices=tuple(available_restore()),
                    help="Face-restoration model to run over the render")
    ap.add_argument("--restore-fidelity", type=float, default=0.7,
                    help="CodeFormer -w: 0 = best quality, 1 = most faithful "
                         "to the input. Upstream suggests 0.5-0.7.")
    ap.add_argument("--restore-upscale", type=int, default=1,
                    help="CodeFormer -s. Keep at 1 to restore the face only "
                         "and leave the frame's resolution alone.")
    ap.add_argument("--restore-only-center-face", action="store_true",
                    help="Restore just the centre face. Nothing in this repo "
                         "gates lip-sync on who is speaking, so on a "
                         "multi-face shot this at least pins the restoration "
                         "to one of them.")
    ap.add_argument("--restore-keep-frames", action="store_true",
                    help="Keep the extracted/restored PNG frames for "
                         "inspection instead of deleting them.")
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
        # cut peak RAM/VRAM by half.
        kwargs["resize_factor"] = int(os.environ.get("WAV2LIP_RESIZE_FACTOR", "2"))
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
    if not args.restore:
        return 0

    # Face restoration over the render. Driven from the same dubbed WAV so
    # the reassembled video carries the dubbed track at 48 kHz stereo.
    rest_out = out_dir / f"{args.stem}.{args.model}.{args.restore}.mp4"
    rkwargs = {
        "fidelity": args.restore_fidelity,
        "upscale": args.restore_upscale,
        "only_center_face": args.restore_only_center_face,
        "keep_frames": args.restore_keep_frames,
    }
    restorer = get_restore(args.restore, **rkwargs)
    t1 = time.time()
    rres = restorer.restore(RestoreRequest(
        video=res.out_video, out_path=rest_out, audio=mix_wav,
    ))
    rdt = time.time() - t1
    print(f"restore={rres.model_name} out={rres.out_video} "
          f"frames={rres.frames} frames_with_faces={rres.frames_with_faces} "
          f"fps={rres.fps:.1f} dur={rres.duration_s:.1f}s wall={rdt:.1f}s "
          f"error={rres.error!r}")
    if not rres.error and rres.frames and not rres.frames_with_faces:
        # rc=0 and no face anywhere means the pass was a no-op re-encode.
        print(f"WARNING: {args.restore} found no face in any of "
              f"{rres.frames} frames — the output only re-encodes the input",
              file=sys.stderr)
    return 1 if rres.error else 0


if __name__ == "__main__":
    sys.exit(main())
