"""Real-ESRGAN (+ optional GFPGAN face enhance) over every frame of a video.

This is stage 2 of saifhassan/Wav2Lip-HD. Upstream's `run_final.sh` does it as:

    python video2frames.py --input_video out.mp4 --frames_path frames_wav2lip/X
    cd Real-ESRGAN && python inference_realesrgan.py -n RealESRGAN_x4plus \
        -i ../frames_wav2lip/X --output ../frames_hd/X --outscale 3.5 --face_enhance
    # ffmpeg -r 20 -i .../frame_%05d_out.jpg ... (COMMENTED OUT upstream)

Three deliberate differences, each of which upstream gets away with only
because its examples are a few hundred frames of small video:

1. **No JPEG round-trip.** `video2frames.py` writes `.jpg`, so every frame is
   re-compressed twice before it reaches the encoder. We stream frames in
   memory, so the only generation loss is the final encode.

2. **No frame directory.** A 1500-frame 4K clip is ~40 GB of PNG and still
   ~6 GB of JPEG. We pipe raw frames straight into ffmpeg.

3. **`--outscale` is derived, not fixed at 3.5.** Upstream's 3.5 on a 1080p
   render would emit 6720x3780. The caller passes the target dimensions --
   the SOURCE video's -- and we resize to exactly those. `backend/lipsync.py`
   checks frame parity and dimensions against the source, so an adapter that
   silently changes resolution fails the run.

Frame count is preserved exactly: one frame read, one frame written. That is
load-bearing -- the parity check treats any difference as a failure.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def build_upsampler(weights_dir: Path, tile: int, device: str):
    """RealESRGAN_x4plus, the weight `run_final.sh` actually names."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(weights_dir / "RealESRGAN_x4plus.pth"),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=torch.device(device),
    )


def build_face_enhancer(weights_dir: Path, upsampler, upscale: int):  # noqa: D401
    """GFPGAN, which is what `--face_enhance` turns on upstream.

    This is the part that matters for a lip-sync render: Wav2Lip repaints a
    96x96 mouth crop, so the mouth is the degraded region and a face-aware
    restorer is aimed exactly at it. Plain Real-ESRGAN treats it as texture.
    """
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=str(weights_dir / "GFPGANv1.4.pth"),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--target-width", type=int, required=True)
    ap.add_argument("--target-height", type=int, required=True)
    ap.add_argument("--face-enhance", action="store_true")
    ap.add_argument("--bg-upsample", action="store_true",
                    help="Also run Real-ESRGAN x4 over the WHOLE frame, as "
                         "upstream's --face_enhance does. Measured on this "
                         "host at 1080p: 10.9 s/frame, i.e. ~10.5 h for the "
                         "3460-frame batch, because a 1920x1080 frame is 12 "
                         "tiles of x4 RRDBNet -- and the result is then "
                         "resized straight back down to the source size. "
                         "Wav2Lip only repaints the mouth, so the background "
                         "going through a super-resolution GAN is work with "
                         "nothing to show for it. Off by default.")
    ap.add_argument("--tile", type=int, default=512,
                    help="0 disables tiling. Tiling is what keeps a 4K frame "
                         "inside 16 GiB; the cards here have no room for a "
                         "whole-frame x4 activation.")
    ap.add_argument("--fps", default=None,
                    help="EXACT frame rate as ffprobe reports it, e.g. "
                         "'24000/1001'. Passed straight to ffmpeg so the "
                         "rate is a rational, not a decimal. cv2's "
                         "CAP_PROP_FPS is a float and rounds 24000/1001 to "
                         "23.976023976023978; upstream Wav2Lip-HD is worse "
                         "still, truncating it to int (23) in "
                         "video2frames.py and then recombining at a "
                         "hardcoded -r 20. A rate that disagrees with its "
                         "sibling shard by any amount is what lets the "
                         "concat demuxer drop a shard without failing. "
                         "Falls back to cv2 when not given.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args()

    weights = Path(a.weights_dir)
    src = cv2.VideoCapture(a.input)
    if not src.isOpened():
        print(f"enhance: cannot open {a.input}", file=sys.stderr)
        return 2
    n = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    # Prefer the exact rational the caller probed; cv2's float is a fallback.
    fps = a.fps or f"{src.get(cv2.CAP_PROP_FPS) or 25.0}"
    tw, th = a.target_width, a.target_height

    # The background upsampler is built only when asked for. GFPGANer with
    # bg_upsampler=None still restores the face properly and lifts the rest
    # of the frame by plain interpolation -- which is all the rest of the
    # frame needs, since it is already the source pixels.
    upsampler = (build_upsampler(weights, a.tile, a.device)
                 if (a.bg_upsample or not a.face_enhance) else None)
    restorer = (build_face_enhancer(weights, upsampler, upscale=2)
                if a.face_enhance else None)
    print(f"enhance: {n} frames @ {fps}fps -> {tw}x{th} "
          f"face_enhance={bool(restorer)} bg_upsample={bool(upsampler)} "
          f"tile={a.tile}", flush=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        [a.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{tw}x{th}", "-r", str(fps),
         "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "17",
         "-pix_fmt", "yuv420p", a.output],
        stdin=subprocess.PIPE)

    written = 0
    while True:
        ok, frame = src.read()
        if not ok:
            break
        try:
            if restorer is not None:
                # GFPGANer returns (cropped_faces, restored_faces, restored_img)
                _, _, out = restorer.enhance(
                    frame, has_aligned=False, only_center_face=False,
                    paste_back=True)
            else:
                out, _ = upsampler.enhance(frame, outscale=2)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # A frame that will not fit is passed through rather than
            # dropped. Dropping it would break frame parity, which is a
            # much louder failure than one un-enhanced frame.
            print(f"enhance: frame {written} fell back to source ({e})",
                  file=sys.stderr, flush=True)
            out = frame
        if out is None:
            out = frame
        if out.shape[1] != tw or out.shape[0] != th:
            interp = (cv2.INTER_AREA if out.shape[1] > tw
                      else cv2.INTER_CUBIC)
            out = cv2.resize(out, (tw, th), interpolation=interp)
        ff.stdin.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
        written += 1
        if written % 100 == 0:
            print(f"enhance: {written}/{n}", flush=True)

    src.release()
    ff.stdin.close()
    rc = ff.wait()
    print(f"enhance: wrote {written} frame(s), ffmpeg rc={rc}", flush=True)
    if rc != 0:
        return rc
    if written != n and n > 0:
        print(f"enhance: WARNING read {n} frames but wrote {written}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
