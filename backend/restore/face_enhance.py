"""GFPGAN face restoration (+ optional Real-ESRGAN) over a rendered video.

Renamed from `wav2liphd_enhance.py` on 2026-10-23. Nothing in here was ever
Wav2Lip-specific -- it takes a video in, a video out and a target size, and
has no idea which model drew the mouths. The old name came from the upstream
repo it was ported from (saifhassan/Wav2Lip-HD), and it encoded a
post-process as a property of one lip-sync model. Both `wav2lip` and
`musetalk` renders go through this same pass.

Upstream's `run_final.sh` does it as:

    python video2frames.py --input_video out.mp4 --frames_path frames_wav2lip/X
    cd Real-ESRGAN && python inference_realesrgan.py -n RealESRGAN_x4plus \
        -i ../frames_wav2lip/X --output ../frames_hd/X --outscale 3.5 --face_enhance
    # ffmpeg -r 20 -i .../frame_%05d_out.jpg ... (COMMENTED OUT upstream)

Six deliberate differences. The first three are upstream's, which it gets
away with only because its examples are a few hundred frames of small video;
the last three are measured on this host.

1. **No JPEG round-trip.** `video2frames.py` writes `.jpg`, so every frame is
   re-compressed twice before it reaches the encoder. We stream frames in
   memory, so the only generation loss is the final encode.

2. **No frame directory.** A 1500-frame 4K clip is ~40 GB of PNG and still
   ~6 GB of JPEG. We pipe raw frames straight into ffmpeg.

3. **`--outscale` is derived, not fixed at 3.5.** Upstream's 3.5 on a 1080p
   render would emit 6720x3780. The caller passes the target dimensions and
   we resize to exactly those. `backend/lipsync.py` checks frame parity and
   dimensions against the source, so an adapter that silently changes
   resolution fails the run.

4. **RESTORATION AND SCALING ARE SEPARATE STEPS.** `GFPGANer` bundles them:
   its `upscale` argument makes `paste_faces_to_input_image` resize the whole
   background to `upscale x` and paste the face onto that. The old code
   hardcoded `upscale=2`, so a 1080p render with a 1080p target went
   1920x1080 -> LANCZOS -> 3840x2160 -> paste -> INTER_AREA -> 1920x1080, two
   8.3 Mpx resamples per frame to arrive back where it started. On a real
   frame that round trip changes 6.4% of pixels at 63.25 dB -- visually
   nothing, but it is not identity and it is not free: GPU utilisation
   measured 33.9% mean over a 1156 s run, i.e. the loop is CPU-bound and
   those resamples are on the critical path.

   `upscale` is now DERIVED from the target, and the resize to target is one
   uniform step applied to every frame. That split is what makes
   `--frame-mask` safe, so the two changes are not independent.

5. **`--frame-mask` restores only the frames the lip-sync model changed.**
   Measured 2026-10-22: 0.771 s/frame, 1156 s for 1500 frames, which is 2.6x
   the MuseTalk render being cleaned. An ASD-gated MuseTalk run animates 235
   of 1500 frames and writes the rest through untouched -- so 84% of that
   cost was spent restoring frames that are the source pixels, moving them
   AWAY from the source for no reason. `backend/restore_mask.py` builds
   the mask by diffing the render against its input, so it works for any
   lip-sync model rather than only the one that can report.

   **A masked frame must take the SAME geometric path as a restored one**,
   or the two classes arrive at the target by different resamplings and the
   difference alternates frame to frame. Hence restore-then-scale with the
   scale outside the branch: with `upscale` derived, everything outside
   GFPGAN's own (Gaussian-blurred) face-parse mask is byte-identical to the
   input, so a boundary between a restored and a passed-through frame shows
   only the face.

6. **It shards itself across the available GPUs.** From the same run: peak
   VRAM 1691 MiB of 16311 on ONE card, mean utilisation 33.9%, and the
   second card completely idle at 15 MiB. Half the hardware unused and the
   used half idle two thirds of the time. With `--gpus auto` (the default)
   the process splits the frame range, re-invokes itself once per worker,
   verifies each shard and joins them. One worker runs inline, so there is
   no subprocess on a single-GPU host.

7. **`--seek-shards` stops every shard decoding from frame 0.** OFF by
   default; see the warning below. Each worker opens the file at frame 0 and
   decodes forward to its own start, so four shards decode 2.5x the video
   between them and the last one decodes all of it. That is paid serially
   inside each worker, on the critical path of the slowest shard -- which is
   the number anyone waits for. It is also getting worse rather than better:
   `--frame-mask` cut the RESTORATION work by ~84% and did not touch decode
   at all, and more workers means more redundant decode.

   With the flag, the parent indexes the video's keyframes once and tells
   each worker the last keyframe at or before its start; the worker seeks
   there with ffmpeg input-seek and discards forward over at most one GOP.

WHY SEEKING IS OPT-IN AND NOT THE DEFAULT
-----------------------------------------
`cv2.CAP_PROP_POS_FRAMES` is unreliable on B-frames, which is why the
decode-and-discard above exists at all. An off-by-one seek does not crash and
does not change the frame COUNT -- it shifts every whole-video index in the
shard by one, so `--frame-mask` restores the wrong mouths and the run exits
0. That is the same shape as the coord-cache trap in
`models/lipsync/coords.py`, and the parent's decode-and-count verification
cannot see it: the shard has the right number of the wrong frames.

So the seek is built to be verifiable rather than trusted:

  * The index comes from ffprobe's own keyframe list, not from arithmetic on
    a guessed GOP length.
  * A keyframe whose timestamp does not land on a multiple of 1/fps means
    the source is not CFR, the index would be a rounded guess, and the whole
    optimisation is abandoned for that run rather than applied approximately.
  * `-ss` targets half a frame BEFORE the keyframe. There is no frame in
    that gap, so an accurate seek still returns the keyframe itself, and a
    timestamp that rounds the wrong way cannot land past it.
  * ffmpeg does the seek, not cv2.

Prove it on your own media before turning it on: render one shard range with
the flag and the same range without it, and compare the outputs byte for
byte. Equal means the indices line up.

LOW-RESOLUTION SOURCES
----------------------
This gets MORE out of a 512p render than a 1080p one, which is the opposite
of the intuition. `FaceRestoreHelper` is built with `face_size=512` always
and warps whatever face it finds onto a fixed 512x512 FFHQ template, so:

  1080p source   face is ~300-500 px   warp to 512 is ~1:1   detail refinement
  512p source    face is ~120-180 px   warp to 512 is 3-4x   real super-resolution

Blind restoration from a degraded low-res face IS what GFPGAN was trained
for. Two things to watch, both specific to low resolution:

  * **At `upscale > 1` only the FACE gets real detail.** With no
    `--bg-upsample`, `paste_faces_to_input_image` lifts the background by
    plain LANCZOS, so a 512 -> 1024 pass gives a GAN-sharp face on a soft
    frame. That asymmetry is invisible at 1080p (where upscale is 1) and
    obvious at 512. `--bg-upsample` is the fix and is affordable here: its
    measured 10.9 s/frame is at 1080p and scales with pixels, so a 512p
    frame is roughly 2.4 s.
  * **Small faces are skipped SILENTLY.** `get_face_landmarks_5` passes
    `eye_dist_threshold=5` and drops any face whose eyes are closer than 5
    px. On a low-res frame with a distant speaker that is reachable, and the
    frame then passes through unrestored with no error. The `no_face`
    counter exists to make that visible instead of leaving it to be found in
    the output.

THE CONCAT TRAP, WHICH IS WHY THE JOIN IS VERIFIED
--------------------------------------------------
Measured 2026-10-19 on `hd_bench1_40s`: two 480-frame shards joined to 504
frames and the run reported no error. The concat demuxer DROPS a shard whose
stream parameters disagree with its sibling rather than failing, and the
thing that disagreed was the frame rate -- upstream Wav2Lip-HD reads it
through `int(cv2.CAP_PROP_FPS)` and recombines at a hardcoded 20.

Three defences, all needed:

  1. Every shard is encoded with the SAME `--fps` rational, probed once by
     the parent with ffprobe's `r_frame_rate` and passed down verbatim.
  2. Every worker DECODES its own output and counts frames before exiting,
     which happens in parallel so it costs 1/N of a decode in wall time.
     `nb_frames` from the container is not evidence -- the corrupt render of
     2026-10-22 claimed 1500 and decoded 370.
  3. The parent decodes and counts the join.

Frame count is preserved exactly: one frame read, one frame written. That is
load-bearing -- `backend/lipsync.py` treats any difference as a failure.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SELF = Path(__file__).resolve()

#: Below this many frames per shard the split costs more in startup than it
#: saves. Model load alone is ~25-40 s: three networks (RetinaFace resnet50,
#: GFPGAN v1.4, ParseNet) per worker.
MIN_FRAMES_PER_SHARD = 120

#: A worker's own summary line, which the parent reads back out of the part
#: logs. A shard's stdout goes to its log and never reaches whoever called the
#: parent, so without this the totals of a sharded run are simply absent --
#: and the caller that parses them (`models/restore/gfpgan_adapter.py`)
#: reported 0 frames / 0 restored for every multi-GPU run, i.e. the default.
_WORKER_SUMMARY_RE = re.compile(r"restored (\d+), passed through (\d+)")


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------

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

    This is the part that matters for a lip-sync render: both models here
    rebuild the mouth at a fixed small size -- Wav2Lip repaints a 96x96 crop,
    MuseTalk inpaints at 256x256 -- so the mouth is the degraded region and a
    face-aware restorer is aimed exactly at it. Plain Real-ESRGAN treats it
    as texture.

    `upscale` is the caller's DERIVED value, never a constant. It controls
    the canvas `paste_faces_to_input_image` composites onto, so 1 means "give
    the frame back at the size it arrived".
    """
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=str(weights_dir / "GFPGANv1.4.pth"),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_mask(path: str, n_frames: int) -> set[int]:
    """Frame indices to restore, from the gfpgan adapter's diff or a bare list.

    A mask that does not match the video is refused rather than truncated:
    applying one clip's mask to another's frames restores the wrong mouths at
    exit code 0, which is the same shape as the coord-cache trap in
    `models/lipsync/coords.py`.
    """
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        frames = raw.get("frames")
        idxs = raw.get("changed")
        if idxs is None:
            raise SystemExit(f"enhance: {path} has no 'changed' list")
        if frames is not None and n_frames and int(frames) != n_frames:
            raise SystemExit(
                f"enhance: mask was built for {frames} frame(s) but this "
                f"video has {n_frames}. Refusing rather than restoring the "
                f"wrong frames.")
    elif isinstance(raw, list):
        idxs = raw
    else:
        raise SystemExit(f"enhance: {path} is not a mask")
    return {int(i) for i in idxs}


def derive_upscale(src_w: int, src_h: int, tw: int, th: int) -> int:
    """GFPGAN's paste canvas multiplier, from the target size.

    1 whenever the target is at or below the input, which is every same-size
    restoration pass -- and `cv2.resize` to identical dimensions is a lossless
    copy (verified on cv2 4.10.0), so the background then survives untouched.
    Only a genuine upscale needs more, and then the background is merely
    interpolated unless `--bg-upsample` is on.
    """
    if tw <= src_w and th <= src_h:
        return 1
    return max(1, min(4, math.ceil(max(tw / max(1, src_w),
                                       th / max(1, src_h)))))


def decoded_frames(ffprobe: str, path: str) -> int:
    """Real decode count. `nb_frames` from the container is not evidence."""
    r = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return int((r.stdout or "").strip().splitlines()[0])
    except (IndexError, ValueError):
        return -1


def probe_rate(ffprobe: str, path: str) -> str | None:
    """ffprobe's `r_frame_rate`, as the rational it is."""
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    out = (r.stdout or "").strip().splitlines()
    return out[0].strip() if out and "/" in out[0] else None


def fps_to_float(fps: str) -> float:
    """The probed rational as a float, for timestamp arithmetic only.

    Never for encoding -- the rational is passed to ffmpeg verbatim, because
    a rate that disagrees with a sibling shard is what lets concat drop one.
    """
    try:
        if "/" in str(fps):
            num, den = str(fps).split("/", 1)
            return float(num) / float(den) if float(den) else 0.0
        return float(fps)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def keyframe_index(ffprobe: str, path: str, fps_f: float) -> list[tuple[int, float]]:
    """(frame index, timestamp) for every keyframe, or [] if not derivable.

    `-skip_frame nokey` makes ffprobe report only keyframes, so this is one
    cheap pass rather than a listing of every frame in the file.

    Returns EMPTY rather than approximate whenever the index cannot be
    derived exactly. On a CFR file a keyframe sits on a multiple of 1/fps; a
    timestamp that does not means the source is variable-rate, and an index
    obtained by rounding would silently shift every mask lookup in the shard
    that used it. The caller then falls back to decoding from frame 0, which
    is slower and exactly correct.
    """
    if fps_f <= 0:
        return []
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    out: list[tuple[int, float]] = []
    for line in (r.stdout or "").splitlines():
        line = line.strip().rstrip(",")
        if not line or line == "N/A":
            continue
        try:
            t = float(line)
        except ValueError:
            continue
        exact = t * fps_f
        idx = int(round(exact))
        if abs(exact - idx) > 0.05:
            print(f"enhance: keyframe at {t:.6f}s is {exact:.3f} frames in, "
                  f"not an integer -- treating this source as variable-rate "
                  f"and decoding shards from frame 0", file=sys.stderr)
            return []
        out.append((idx, t))
    out.sort()
    return out


def open_seek_pipe(a, src_path: str, start_t: float):
    """ffmpeg decoding raw BGR from `start_t`, as a pipe.

    `-ss` BEFORE `-i` is an input seek: ffmpeg jumps to the keyframe and
    decodes from there, instead of walking the file. cv2 does not get to do
    this -- its own seek is the unreliable one.
    """
    return subprocess.Popen(
        [a.ffmpeg, "-v", "error", "-nostdin",
         "-ss", f"{max(0.0, start_t):.6f}", "-i", src_path,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)


def read_frames(src, pipe, sw: int, sh: int, base: int, f0: int, f1: int):
    """Yield (whole-video index, BGR frame) over [f0, f1).

    `base` is the index of the first frame the reader will produce: 0 for the
    cv2 reader, the seeked-to keyframe for the pipe. Everything downstream
    keys off the yielded index, including the mask -- using a write counter
    instead would shift every lookup by `f0` in any shard but the first.

    Frames before `f0` are dropped as cheaply as the reader allows. `grab()`
    decodes without the colour conversion and the array allocation `read()`
    pays for, which is most of the per-frame cost of a frame nobody wants.
    """
    nbytes = sw * sh * 3
    idx = base - 1
    while True:
        idx += 1
        if idx >= f1:
            return
        if idx < f0:
            if pipe is not None:
                if len(pipe.stdout.read(nbytes)) < nbytes:
                    return
            elif not src.grab():
                return
            continue
        if pipe is not None:
            buf = pipe.stdout.read(nbytes)
            if len(buf) < nbytes:
                return
            # Copied because frombuffer is read-only and this frame is handed
            # to OpenCV and GFPGAN, neither of which promises not to write.
            yield idx, np.frombuffer(buf, np.uint8).reshape(sh, sw, 3).copy()
        else:
            ok, frame = src.read()
            if not ok:
                return
            yield idx, frame


def plan_gpus(spec: str) -> list[str]:
    """Device ids for the workers.

    `auto` asks torch, which already honours an inherited
    `CUDA_VISIBLE_DEVICES` -- so a caller that pinned one card gets one
    worker on it rather than this quietly spreading onto a card it was told
    to leave alone.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "auto"):
        try:
            n = torch.cuda.device_count()
        except Exception:                                 # noqa: BLE001
            n = 0
        return [str(i) for i in range(n)] or ["0"]
    if spec == "cpu":
        return ["0"]
    return [g.strip() for g in spec.split(",") if g.strip()] or ["0"]


# --------------------------------------------------------------------------
# the worker: restore one contiguous frame range, inline
# --------------------------------------------------------------------------

def render_range(a, src_path: str, out_path: str, f0: int, f1: int | None,
                 fps: str, tag: str = "enhance") -> int:
    weights = Path(a.weights_dir)
    src = cv2.VideoCapture(src_path)
    if not src.isOpened():
        print(f"{tag}: cannot open {src_path}", file=sys.stderr)
        return 2
    n = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    sw = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tw = a.target_width or sw
    th = a.target_height or sh

    mask = load_mask(a.frame_mask, n) if a.frame_mask else None

    f0 = max(0, f0)
    f1 = f1 if f1 is not None else (n or (1 << 30))
    if f1 <= f0:
        print(f"{tag}: empty range [{f0}, {f1})", file=sys.stderr)
        return 2
    n_range = min(f1, n or f1) - f0

    upscale = derive_upscale(sw, sh, tw, th)
    upsampler = (build_upsampler(weights, a.tile, a.device)
                 if (a.bg_upsample or not a.face_enhance) else None)
    restorer = (build_face_enhancer(weights, upsampler, upscale=upscale)
                if a.face_enhance else None)

    print(f"{tag}: {n_range} frames {sw}x{sh} -> {tw}x{th} @ {fps}fps "
          f"face_enhance={bool(restorer)} bg_upsample={bool(upsampler)} "
          f"upscale={upscale} tile={a.tile} strength={a.strength} "
          f"only_center_face={a.only_center_face}", flush=True)
    if mask is not None:
        in_range = sum(1 for i in mask if f0 <= i < f1)
        print(f"{tag}: frame mask -> restoring {in_range} of {n_range} "
              f"({100.0 * in_range / max(1, n_range):.1f}%); the rest are "
              f"passed through and scaled identically", flush=True)
    if upscale > 1 and not a.bg_upsample:
        # Loud, because it is the low-resolution failure that looks like a
        # model problem: GFPGAN gives the face real detail while the
        # background is only interpolated up to meet it.
        print(f"{tag}: WARNING upscaling {sw}x{sh} -> {tw}x{th} with the "
              f"face restorer only. The background is plain LANCZOS, so the "
              f"face will be sharper than the frame around it. Pass "
              f"--bg-upsample for Real-ESRGAN on the background "
              f"(~2.4 s/frame at 512p, ~10.9 s/frame at 1080p).",
              file=sys.stderr, flush=True)

    # Reader. The seek path replaces cv2 entirely for this shard: it starts
    # at a keyframe the parent picked and the loop discards forward over at
    # most one GOP, instead of decoding every frame since the start of the
    # file. `base` is what keeps whole-video indices honest afterwards.
    pipe = None
    base = 0
    kf, kf_t = a.seek_from_frame, a.seek_from_time
    if kf and kf_t is not None and kf <= f0:
        # Half a frame early: there is no frame in that gap, so an accurate
        # seek still lands on the keyframe, and a timestamp that rounds the
        # wrong way cannot overshoot it.
        t = max(0.0, float(kf_t) - 0.5 / max(1e-9, fps_to_float(fps)))
        pipe = open_seek_pipe(a, src_path, t)
        base = int(kf)
        src.release()
        print(f"{tag}: seeking to keyframe {base} ({t:.3f}s) instead of "
              f"decoding {f0} frame(s) from the start", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        [a.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{tw}x{th}", "-r", str(fps),
         "-i", "pipe:0", "-an",
         "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf),
         "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)

    written = restored = skipped = no_face = 0
    t0 = time.time()
    for idx, frame in read_frames(src, pipe, sw, sh, base, f0, f1):
        # --- restore (optional, gated) --------------------------------
        # `idx` is the WHOLE-VIDEO index, which is what the mask is keyed on.
        # Using the write counter instead would silently shift every lookup
        # by `f0` in any shard but the first.
        if restorer is not None and (mask is None or idx in mask):
            try:
                # GFPGANer returns (cropped_faces, restored_faces, restored_img).
                # `weight=` is deliberately NOT passed: it is inert on
                # arch='clean' (see --strength) and passing it would imply a
                # control that does not exist. Strength is applied below.
                cropped, _, out = restorer.enhance(
                    frame, has_aligned=False,
                    only_center_face=a.only_center_face,
                    paste_back=True)
                restored += 1
                if out is not None and 0.0 <= a.strength < 1.0:
                    # Blend back toward the original. Only the face differs,
                    # so this is a face-strength control despite acting on
                    # the whole frame.
                    out = cv2.addWeighted(out, a.strength,
                                          frame, 1.0 - a.strength, 0.0)
                if not cropped:
                    # Detection found nothing -- GFPGAN returns the frame
                    # untouched and says nothing. Reachable on a low-res
                    # source via `eye_dist_threshold=5`, and on any source
                    # where the mask named a frame whose face left shot.
                    no_face += 1
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                # A frame that will not fit is passed through rather than
                # dropped. Dropping it would break frame parity, which is a
                # much louder failure than one un-enhanced frame.
                print(f"{tag}: frame {idx} fell back to source ({e})",
                      file=sys.stderr, flush=True)
                out = frame
            if out is None:
                out = frame
        elif restorer is None and upsampler is not None:
            out, _ = upsampler.enhance(frame, outscale=upscale)
        else:
            out = frame
            skipped += 1

        # --- scale (always, identically for both branches) ------------
        # Outside the branch on purpose: a passed-through frame and a
        # restored one must reach the target by the same resampling, or the
        # difference alternates frame to frame at every mask boundary.
        if out.shape[1] != tw or out.shape[0] != th:
            interp = (cv2.INTER_AREA if out.shape[1] > tw
                      else cv2.INTER_CUBIC)
            out = cv2.resize(out, (tw, th), interpolation=interp)

        ff.stdin.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
        written += 1
        if written % 100 == 0:
            print(f"{tag}: {written}/{n_range} (restored {restored}, "
                  f"passed {skipped})", flush=True)

    src.release()
    if pipe is not None:
        # Killed rather than waited on: a shard that stops at f1 leaves
        # ffmpeg with a whole file still to decode and nowhere to put it.
        pipe.stdout.close()
        pipe.kill()
        pipe.wait()
    ff.stdin.close()
    rc = ff.wait()
    dt = time.time() - t0
    print(f"{tag}: wrote {written} frame(s) in {dt:.0f}s "
          f"(restored {restored}, passed through {skipped}), ffmpeg rc={rc}",
          flush=True)
    if rc != 0:
        return rc
    if no_face:
        print(f"{tag}: WARNING {no_face} of {restored} restored frame(s) had "
              f"NO face detected and were returned unchanged. On a "
              f"low-resolution source this is usually facexlib's "
              f"eye_dist_threshold=5 dropping a small face; upscale the "
              f"input before restoring, or expect those frames unrestored.",
              file=sys.stderr, flush=True)
    if written != n_range and n_range > 0:
        print(f"{tag}: WARNING expected {n_range} frames but wrote {written}",
              file=sys.stderr)

    # Self-verify by DECODE. In a sharded run every worker does this at the
    # same time, so it costs 1/N of a full decode in wall clock.
    if a.verify:
        got = decoded_frames(a.ffprobe, out_path)
        if got != written:
            print(f"{tag}: FAILED -- wrote {written} frame(s) but "
                  f"{out_path} decodes {got}", file=sys.stderr)
            return 1
    return 0


# --------------------------------------------------------------------------
# the parent: split, spawn, verify, join
# --------------------------------------------------------------------------

def run_sharded(a, gpus: list[str], n: int, fps: str) -> int:
    n_workers = max(1, len(gpus) * max(1, a.workers_per_gpu))
    if n and n // n_workers < MIN_FRAMES_PER_SHARD:
        n_workers = max(1, n // MIN_FRAMES_PER_SHARD)
    if n_workers <= 1:
        return render_range(a, a.input, a.output, 0, None, fps)

    work = Path(a.work_dir) if a.work_dir else Path(a.output + ".shards")
    work.mkdir(parents=True, exist_ok=True)

    # Contiguous ranges, remainder spread over the first shards so no shard
    # is more than one frame longer than another.
    base, extra = divmod(n, n_workers)
    bounds, start = [], 0
    for i in range(n_workers):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size

    print(f"enhance: {n} frames @ {fps} -> {n_workers} worker(s) over gpu(s) "
          f"{','.join(gpus)}", flush=True)

    # One keyframe index for the whole run, shared by every shard. Empty
    # means either the flag is off or the source is not CFR, and every
    # worker then decodes from frame 0 as it always did.
    kfs = (keyframe_index(a.ffprobe, a.input, fps_to_float(fps))
           if a.seek_shards else [])
    kf_starts = [k for k, _t in kfs]
    if a.seek_shards:
        print(f"enhance: {len(kfs)} keyframe(s) indexed; shards will seek"
              if kfs else
              "enhance: no usable keyframe index; shards decode from frame 0",
              flush=True)

    procs = []
    t0 = time.time()
    for i, (f0, f1) in enumerate(bounds):
        out_i = work / f"part{i:02d}.mp4"
        cmd = [sys.executable, str(SELF),
               "--input", a.input, "--output", str(out_i),
               "--weights-dir", a.weights_dir,
               "--fps", fps, "--tile", str(a.tile),
               "--strength", str(a.strength),
               "--preset", a.preset, "--crf", str(a.crf),
               "--frame-start", str(f0), "--frame-end", str(f1),
               # One gpu per child is what stops it sharding again.
               "--gpus", "0",
               "--ffmpeg", a.ffmpeg, "--ffprobe", a.ffprobe]
        if a.target_width:
            cmd += ["--target-width", str(a.target_width)]
        if a.target_height:
            cmd += ["--target-height", str(a.target_height)]
        if a.face_enhance:
            cmd.append("--face-enhance")
        if a.only_center_face:
            cmd.append("--only-center-face")
        if a.bg_upsample:
            cmd.append("--bg-upsample")
        if a.frame_mask:
            cmd += ["--frame-mask", a.frame_mask]
        if not a.verify:
            cmd.append("--no-verify")

        # The last keyframe at or before this shard's first frame. Shard 0
        # starts at 0 and has nothing to skip, so it is left on the plain
        # reader rather than seeking to where it already is.
        if kfs and f0 > 0:
            j = bisect.bisect_right(kf_starts, f0) - 1
            if j >= 0 and kfs[j][0] > 0:
                cmd += ["--seek-from-frame", str(kfs[j][0]),
                        "--seek-from-time", f"{kfs[j][1]:.6f}"]

        env = dict(os.environ)
        # The child sees exactly one device and calls it 0, which is why
        # `--gpus 0` above is correct rather than a bug.
        env["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
        log = open(work / f"part{i:02d}.log", "w")
        print(f"enhance: part{i:02d} frames [{f0}, {f1}) on gpu "
              f"{env['CUDA_VISIBLE_DEVICES']}", flush=True)
        procs.append((i, f0, f1, out_i, log,
                      subprocess.Popen(cmd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)))

    failed = []
    for i, f0, f1, out_i, log, p in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            failed.append(f"part{i:02d} rc={rc}")
    if failed:
        print(f"enhance: FAILED -- {', '.join(failed)}. Logs in {work}",
              file=sys.stderr)
        return 1

    for i, f0, f1, out_i, _l, _p in procs:
        if not out_i.is_file() or out_i.stat().st_size == 0:
            print(f"enhance: FAILED -- part{i:02d} wrote nothing",
                  file=sys.stderr)
            return 1

    listing = work / "concat.txt"
    listing.write_text("".join(
        f"file '{out_i.resolve()}'\n" for _i, _a, _b, out_i, _l, _p in procs))
    join = subprocess.run(
        [a.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", a.output],
        capture_output=True, text=True)
    if join.returncode != 0 or not Path(a.output).is_file():
        print(f"enhance: FAILED -- concat rc={join.returncode}: "
              f"{join.stderr.strip()[-1000:]}", file=sys.stderr)
        return 1

    dt = time.time() - t0
    if a.verify:
        got = decoded_frames(a.ffprobe, a.output)
        if got != n:
            print(f"enhance: FAILED -- joined output decodes {got} frame(s), "
                  f"expected {n}. Shards kept in {work}", file=sys.stderr)
            return 1

    # Totals, summed from the part logs BEFORE they are removed. The shards
    # are the only place the per-worker counts exist.
    restored = passed = 0
    for i, _f0, _f1, _o, _l, _p in procs:
        try:
            m = _WORKER_SUMMARY_RE.search((work / f"part{i:02d}.log").read_text())
        except OSError:
            continue
        if m:
            restored += int(m.group(1))
            passed += int(m.group(2))

    if not a.keep_shards:
        shutil.rmtree(work, ignore_errors=True)
    print(f"enhance: DONE in {dt:.0f}s -> {a.output} "
          f"({n} frames, {dt / max(1, n):.3f}s/frame, {n_workers} workers, "
          f"restored {restored}, passed through {passed})")
    print(json.dumps({"frames": n, "workers": n_workers,
                      "restored": restored, "passed_through": passed,
                      "wall_s": round(dt, 1),
                      "s_per_frame": round(dt / max(1, n), 4)}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--target-width", type=int, default=None,
                    help="default: the input's own width, i.e. a pure "
                         "restoration pass with no scaling")
    ap.add_argument("--target-height", type=int, default=None)
    ap.add_argument("--face-enhance", action="store_true")
    ap.add_argument("--frame-mask", default=None,
                    help="JSON from backend/restore_mask.py (or a bare "
                         "list of frame indices). Only these frames are "
                         "restored; every other frame is passed through and "
                         "then scaled identically. Without it, every frame is "
                         "restored, which is the old behaviour.")
    ap.add_argument("--only-center-face", action="store_true",
                    help="Restore one face per frame instead of all of them. "
                         "Each additional detected face is another GFPGAN AND "
                         "another ParseNet forward, so a multi-face frame "
                         "costs a multiple. Off by default because 'center' "
                         "is a geometric guess, not the speaking face.")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="Blend between the restored frame and the original: "
                         "0 = untouched, 1 = full restoration (default). "
                         "Applied HERE, not by GFPGAN.\n\n"
                         "GFPGANer.enhance() takes a `weight` argument that "
                         "looks like exactly this and is a NO-OP on the "
                         "installed build: it is forwarded to "
                         "`GFPGANv1Clean.forward`, whose signature is "
                         "`(x, return_latents, return_rgb, randomize_noise, "
                         "**kwargs)` and which contains zero occurrences of "
                         "'weight'. It is swallowed by **kwargs. GFPGANv1.4 "
                         "loads arch='clean', so nothing that passes `weight` "
                         "changes anything -- including the upstream "
                         "MuseTalk-API fork, whose `gfpgan_weight` request "
                         "field goes down the same path.\n\n"
                         "A straight frame blend is equivalent and actually "
                         "works: with upscale=1 everything outside GFPGAN's "
                         "face-parse mask is already identical to the input, "
                         "so blending the frame blends only the face.")
    ap.add_argument("--bg-upsample", action="store_true",
                    help="Also run Real-ESRGAN x4 over the WHOLE frame, as "
                         "upstream's --face_enhance does. Measured on this "
                         "host at 1080p: 10.9 s/frame. Only the mouth was "
                         "repainted, so at a same-size target the background "
                         "going through a super-resolution GAN is work with "
                         "nothing to show for it. Turn it on when the TARGET "
                         "is genuinely larger than the source, where the "
                         "alternative is plain LANCZOS.")
    ap.add_argument("--tile", type=int, default=512,
                    help="0 disables tiling. Tiling is what keeps a 4K frame "
                         "inside 16 GiB; the cards here have no room for a "
                         "whole-frame x4 activation.")
    ap.add_argument("--gpus", default="auto",
                    help="'auto' (default) uses every visible CUDA device and "
                         "shards the frame range across them; '0' or '0,1' "
                         "pins explicitly. One device means one worker, run "
                         "inline with no subprocess.")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="More than one is reasonable here: a worker peaked "
                         "at 1691 MiB of 16311 and left the GPU idle 66%% of "
                         "the time, so two workers on one card mostly fill "
                         "each other's gaps. Raises peak VRAM linearly -- "
                         "check it before going above 2.")
    ap.add_argument("--frame-start", type=int, default=0,
                    help="First frame of this shard, in WHOLE-VIDEO indices. "
                         "Frames before it are decoded and discarded rather "
                         "than seeked past: cv2's CAP_PROP_POS_FRAMES is not "
                         "reliable on B-frames, and an off-by-one seek shifts "
                         "every mask index in the shard. Set by the parent; "
                         "rarely useful by hand.")
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--seek-shards", action="store_true",
                    help="Let each shard SEEK to its first frame instead of "
                         "decoding every frame before it. Four shards "
                         "otherwise decode 2.5x the video between them and "
                         "the last decodes all of it, serially, on the "
                         "critical path. OFF by default because the failure "
                         "mode is silent: an off-by-one seek keeps the frame "
                         "COUNT right and shifts every whole-video index, so "
                         "--frame-mask restores the wrong mouths at rc=0 and "
                         "the parent's count check cannot see it. The guards "
                         "are in the module docstring; verify it on your own "
                         "media before leaving it on.")
    ap.add_argument("--seek-from-frame", type=int, default=None,
                    help="Whole-video index of the keyframe this shard seeks "
                         "to. Set by the parent from ffprobe's keyframe list, "
                         "never by hand: a value that is not actually a "
                         "keyframe shifts every index in the shard.")
    ap.add_argument("--seek-from-time", type=float, default=None,
                    help="That keyframe's timestamp, as ffprobe reported it. "
                         "Paired with --seek-from-frame; both or neither.")
    ap.add_argument("--work-dir", default=None,
                    help="shard scratch; default <output>.shards")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip the decode-and-count checks. They cost about "
                         "one extra decode of the output and are what caught "
                         "a 32 MB file that claimed 1500 frames and decoded "
                         "370.")
    ap.add_argument("--fps", default=None,
                    help="EXACT frame rate as ffprobe reports it, e.g. "
                         "'24000/1001'. Passed straight to ffmpeg so the rate "
                         "is a rational, not a decimal. cv2's CAP_PROP_FPS is "
                         "a float and rounds 24000/1001 to 23.976023976023978; "
                         "upstream Wav2Lip-HD is worse still, truncating it to "
                         "int (23) and then recombining at a hardcoded -r 20. "
                         "A rate that disagrees with its sibling shard by any "
                         "amount is what lets the concat demuxer drop a shard "
                         "without failing. Probed when not given.")
    ap.add_argument("--preset", default="medium",
                    help="x264 preset. The loop is CPU-bound (33.9%% mean GPU "
                         "utilisation measured over 1156 s) and the encoder "
                         "competes for the same cores, so this is a real knob "
                         "rather than a cosmetic one.")
    ap.add_argument("--crf", default="17")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.set_defaults(verify=True)
    a = ap.parse_args()

    fps = a.fps or probe_rate(a.ffprobe, a.input)
    if not fps:
        cap = cv2.VideoCapture(a.input)
        fps = f"{cap.get(cv2.CAP_PROP_FPS) or 25.0}"
        cap.release()

    gpus = plan_gpus(a.gpus)
    # A worker is anything told where to stop, or given one device. Both
    # cases run inline; only an unbounded multi-device call forks.
    is_worker = a.frame_end is not None
    if is_worker or (len(gpus) * max(1, a.workers_per_gpu)) <= 1:
        return render_range(a, a.input, a.output, a.frame_start, a.frame_end,
                            fps)

    n = decoded_frames(a.ffprobe, a.input)
    if n <= 0:
        cap = cv2.VideoCapture(a.input)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return run_sharded(a, gpus, n, fps)


if __name__ == "__main__":
    raise SystemExit(main())
