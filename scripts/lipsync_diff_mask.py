#!/usr/bin/env python
"""Which frames did the lip-sync model actually change, and where?

The restoration pass (`scripts/face_enhance.py`) costs 0.771 s/frame at
1080p -- measured 2026-10-22, 1156 s for 1500 frames, which is 2.6x the
MuseTalk render it is cleaning up. Most of that is spent on frames the
lip-sync model never touched: an ASD-gated MuseTalk run animated 235 of
1500 and wrote the other 1265 through unchanged. Restoring those 1265 is
not merely wasted work, it moves them AWAY from the source for no reason.

This module answers "what changed" by DIFFING the render against its
pre-lip-sync source, rather than by asking the model. That is deliberate:

  * MuseTalk can say, but only sometimes. Its `coord_list` marks untouched
    frames with `(0,0,0,0)`, and `backend.lipsync_targets.build_coord_list`
    knows exactly which those are -- but nothing persists it, and the
    answer only exists when the run was ASD-gated.
  * Wav2Lip cannot say at all. It repaints a 96x96 mouth crop on every
    frame with a detected face and raises `Face not detected!` when one is
    missing (third_party/Wav2Lip/inference.py), so it has no concept of an
    untouched frame to report.
  * LatentSync and LipGAN have no reporting path either.

A diff works for all four and cannot drift from what the model did, because
it IS what the model did.

THE ENCODER-NOISE TRAP
----------------------
The two videos are separately H.264-encoded, so an UNTOUCHED frame does not
come back byte-identical. The `clip-run` skill records this in its own
context: diffing two rendered videos reported an edit covering 84% of the
frame when the real edit was a 139x131 box. So a bare `!=` marks every
frame changed and the gate silently does nothing.

The discriminator is that the two differ in SHAPE, not just in magnitude:

  encoder noise   low amplitude, spread thinly over the whole frame
  a repainted mouth   high amplitude, concentrated in one small region

So a pixel counts as changed only above `--threshold`, and a FRAME counts as
changed only when at least `--min-pixels` of them cluster. Both are reported
per frame in the output so a wrong threshold is visible as a histogram
rather than as a silently empty mask.

CALIBRATION IS NOT OPTIONAL, AND THERE IS A KNOWN ANSWER TO CALIBRATE ON
------------------------------------------------------------------------
Job 20261022-070010's ASD-gated render animated exactly 235 of 1500 frames,
and the log names the ten face tracks. Run this against that pair and
`--expect 235` checks the threshold recovers it. A mask that is wrong in
either direction is worse than no mask: too few frames leaves un-restored
mouths next to restored ones, too many gives back the cost the gate exists
to avoid.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _open(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"diff: cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, n, w, h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", required=True,
                    help="the video the lip-sync model was given "
                         "(e.g. mix/<stem>.dubbed.mp4)")
    ap.add_argument("--after", required=True,
                    help="the lip-sync render")
    ap.add_argument("--out", required=True, help="mask JSON to write")
    ap.add_argument("--threshold", type=int, default=16,
                    help="per-pixel abs difference, 0-255, above which a "
                         "pixel counts as changed. Encoder noise at CRF 17-18 "
                         "sits well below this; a repainted mouth does not.")
    ap.add_argument("--min-pixels", type=int, default=None,
                    help="changed pixels needed before the FRAME counts as "
                         "changed. Guards against a handful of noise pixels "
                         "that cleared the threshold. Default scales with "
                         "frame area (w*h/32000, floor 16) -- a fixed count "
                         "tuned at 1080p is a far larger share of a 512p "
                         "frame, so a mouth that is 1200 px at 1080p and "
                         "300 px at 512p stays detected at both.")
    ap.add_argument("--pad", type=int, default=None,
                    help="pixels added around each changed box. The box is "
                         "where pixels MOVED; a face restorer needs the "
                         "feature around it, and GFPGAN's paste mask is "
                         "itself Gaussian-blurred 101x101. Default scales "
                         "with frame height (16 at 1080p, floor 4).")
    ap.add_argument("--expect", type=int, default=None,
                    help="known changed-frame count; exits 1 if the mask "
                         "misses it by more than --expect-tol.")
    ap.add_argument("--expect-tol", type=int, default=10)
    a = ap.parse_args()

    b_cap, b_n, b_w, b_h = _open(a.before)
    f_cap, f_n, f_w, f_h = _open(a.after)

    # Both thresholds are resolution-relative. The comparison happens in the
    # RENDER's space, so they are derived from the render's dimensions.
    if a.min_pixels is None:
        a.min_pixels = max(16, round(f_w * f_h / 32000))
    if a.pad is None:
        a.pad = max(4, round(16 * f_h / 1080))
    print(f"diff: {f_w}x{f_h} -> threshold={a.threshold} "
          f"min_pixels={a.min_pixels} pad={a.pad}", flush=True)

    if b_w != f_w or b_h != f_h:
        # MuseTalk caps the render at 1080p, so a 4K source and its render
        # are legitimately different sizes. Compare in the RENDER's space:
        # downscaling the render to meet the source would blur away the very
        # edit being looked for.
        print(f"diff: before is {b_w}x{b_h}, after is {f_w}x{f_h} -- "
              f"resizing before to the render's size", flush=True)

    changed: list[int] = []
    boxes: dict[str, list[int]] = {}
    counts: list[int] = []

    idx = 0
    while True:
        ok_b, fb = b_cap.read()
        ok_f, ff = f_cap.read()
        if not (ok_b and ok_f):
            break
        if (fb.shape[1], fb.shape[0]) != (ff.shape[1], ff.shape[0]):
            fb = cv2.resize(fb, (ff.shape[1], ff.shape[0]),
                            interpolation=cv2.INTER_AREA)

        d = cv2.absdiff(cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY))
        mask = (d >= a.threshold).astype(np.uint8)
        # One open to drop isolated noise pixels, one dilate to join the
        # mouth into a single region rather than a scatter of islands.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n = int(mask.sum())
        counts.append(n)

        if n >= a.min_pixels:
            ys, xs = np.nonzero(mask)
            x1 = max(0, int(xs.min()) - a.pad)
            y1 = max(0, int(ys.min()) - a.pad)
            x2 = min(ff.shape[1], int(xs.max()) + a.pad)
            y2 = min(ff.shape[0], int(ys.max()) + a.pad)
            changed.append(idx)
            boxes[str(idx)] = [x1, y1, x2, y2]
        idx += 1
        if idx % 200 == 0:
            print(f"diff: {idx} frames, {len(changed)} changed so far",
                  flush=True)

    b_cap.release()
    f_cap.release()

    arr = np.asarray(counts) if counts else np.zeros(1)
    frac = 100.0 * len(changed) / max(1, idx)
    # The histogram is the point: a threshold that is wrong shows up here as
    # a pile of frames just under the cut, not as a plausible-looking mask.
    edges = [0, 1, 16, 64, 256, 1024, 4096, 16384, 1 << 30]
    hist = {f"{edges[i]}-{edges[i+1]}": int(((arr >= edges[i]) &
                                             (arr < edges[i + 1])).sum())
            for i in range(len(edges) - 1)}

    out = {
        "before": str(Path(a.before).resolve()),
        "after": str(Path(a.after).resolve()),
        "frames": idx,
        "threshold": a.threshold,
        "min_pixels": a.min_pixels,
        "pad": a.pad,
        "changed_count": len(changed),
        "changed_fraction_pct": round(frac, 2),
        "changed": changed,
        "boxes": boxes,
        "changed_pixels_hist": hist,
        "changed_pixels_p50": int(np.percentile(arr, 50)),
        "changed_pixels_p95": int(np.percentile(arr, 95)),
        "changed_pixels_max": int(arr.max()),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"\ndiff: {idx} frames, {len(changed)} changed ({frac:.1f}%)")
    print(f"diff: changed-pixel histogram {json.dumps(hist)}")
    print(f"diff: wrote {a.out}")

    if a.expect is not None:
        miss = abs(len(changed) - a.expect)
        if miss > a.expect_tol:
            print(f"diff: FAILED calibration -- found {len(changed)} changed "
                  f"frame(s), expected {a.expect} (+-{a.expect_tol}). "
                  f"Adjust --threshold / --min-pixels against the histogram "
                  f"above before trusting this mask.", file=sys.stderr)
            return 1
        print(f"diff: calibration ok ({len(changed)} vs expected {a.expect})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
