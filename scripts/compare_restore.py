"""Compare a lip-synced render against its face-restored version.

A restoration pass that exits 0 can still have done nothing — it re-encodes
the input either way. This answers the three questions that distinguish a
working pass from a no-op, measured rather than eyeballed:

  1. Did the pixels change *materially*, and what fraction of them?
  2. Did the change land on the *face*, or is it spread over the frame
     (which would mean an upscale or a background pass, not restoration)?
  3. Did the face actually gain detail, or just get moved around?
  4. Is the output still the same length, resolution and frame count?

**Compare frames, not videos, whenever you can.** The two videos come out
of different H.264 encoders, so ~98% of pixels differ by +/-1 from
compression alone, and on detailed content (slide text, hard edges) plenty
of those differences clear any sane noise threshold. That is enough to
smear the localisation across the whole frame and to swamp the sharpness
measurement: measured video-to-video this run reported the edit covering
84% of the frame and the face getting *less* sharp, while the same pair
measured frame-to-frame showed the edit confined to a 139x131 box (2.0% of
the frame) with +36% detail. The videos had not changed; only the ruler had.

So run the restoration with `--restore-keep-frames` and point `--frames`
at the work directory. `--noise-threshold` makes the video-to-video path
less wrong, but it cannot make it right.

PSNR/SSIM are reported over the face crop as well as the whole frame,
because on a picture-in-picture layout the face is a percent or two of the
frame and whole-frame numbers flatter any edit by ~10 dB.

Usage:
    # preferred — zero codec in the loop, needs --restore-keep-frames
    python scripts/compare_restore.py --frames <work_dir> [--out-dir ...]

    # fallback — the two renders; see the warning above
    python scripts/compare_restore.py RAW.mp4 RESTORED.mp4 \
        [--sample 12] [--out-dir data/jobs/<job>/compare]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.media_io import _ffprobe_bin  # noqa: E402


def probe(path: Path) -> dict:
    ffprobe = _ffprobe_bin()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_type,width,height,r_frame_rate,sample_rate,channels,nb_frames",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d.setdefault(k, []).append(v)
    return d


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM on greyscale, gaussian-weighted — the standard formulation."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    saa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float(np.mean(num / den))


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap detail proxy."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def compare_frames(work: Path, out_dir) -> int:
    """Compare a kept work dir's input frames against its restored ones.

    No encoder sits between the two, so every difference is the model's.
    """
    import glob

    src_dir = work / "frames"
    out_png = work / "restored" / "final_results"
    if not src_dir.is_dir() or not out_png.is_dir():
        print(f"ERROR: expected {src_dir} and {out_png}", file=sys.stderr)
        return 2
    names = [Path(f).name for f in sorted(glob.glob(str(src_dir / "*.png")))]
    if not names:
        print(f"ERROR: no frames in {src_dir}", file=sys.stderr); return 2

    accum = None
    changed = []
    for n in names:
        a = cv2.imread(str(src_dir / n)); b = cv2.imread(str(out_png / n))
        if a is None or b is None or a.shape != b.shape:
            continue
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2)
        accum = d.astype(np.float64) if accum is None else accum + d
        changed.append((d > 0).mean())
    if accum is None or accum.max() == 0:
        print("!! NO CHANGE: the pass rewrote nothing.", file=sys.stderr)
        return 1

    ys, xs = np.where(accum > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    h, w = accum.shape
    print(f"== frame-to-frame over {len(changed)} frames (no codec) ==")
    print(f"  changed pixels : {np.mean(changed) * 100:.3f}% per frame")
    print(f"  region touched : x={x0}..{x1} y={y0}..{y1} "
          f"({x1 - x0}x{y1 - y0}) = {(x1-x0)*(y1-y0)/(w*h)*100:.2f}% of {w}x{h}")

    sa, sb = [], []
    for n in names:
        a = cv2.imread(str(src_dir / n)); b = cv2.imread(str(out_png / n))
        if a is None or b is None:
            continue
        sa.append(sharpness(a[y0:y1, x0:x1])); sb.append(sharpness(b[y0:y1, x0:x1]))
    ma, mb = float(np.mean(sa)), float(np.mean(sb))
    print(f"\n== detail inside that region ==")
    print(f"  raw      : {ma:8.1f}")
    print(f"  restored : {mb:8.1f}   ({(mb / ma - 1) * 100:+.1f}%)")
    print(f"  sharper in {sum(1 for x, y in zip(sa, sb) if y > x)}/{len(sa)} frames")
    if mb <= ma:
        print("  !! restored is NOT sharper — the pass is not doing what it claims")

    if out_dir:
        od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
        for n in names[:: max(1, len(names) // 8)]:
            a = cv2.imread(str(src_dir / n))[y0:y1, x0:x1]
            b = cv2.imread(str(out_png / n))[y0:y1, x0:x1]
            a = cv2.resize(a, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            b = cv2.resize(b, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            gap = np.full((a.shape[0], 10, 3), 255, np.uint8)
            cv2.imwrite(str(od / f"face_{Path(n).stem}_raw_vs_restored.png"),
                        np.hstack([a, gap, b]))
        print(f"\n  wrote before/after face crops to {od}")
    return 0 if mb > ma else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?"); ap.add_argument("restored", nargs="?")
    ap.add_argument("--frames", default=None,
                    help="A CodeFormer work dir kept by --restore-keep-frames. "
                         "Compares frames/*.png against "
                         "restored/final_results/*.png, which is the only way "
                         "to measure the pass without an encoder in between.")
    ap.add_argument("--sample", type=int, default=12,
                    help="How many frames to measure, spread over the clip")
    ap.add_argument("--energy", type=float, default=0.98,
                    help="Fraction of total change energy the reported box "
                         "must contain. Localises the edit by density rather "
                         "than by a bbox over every stray changed pixel.")
    ap.add_argument("--noise-threshold", type=int, default=8,
                    help="Per-pixel max-channel difference below which a "
                         "change is treated as H.264 re-encode noise rather "
                         "than an edit. Both files come from different "
                         "encoders, so without this ~98%% of pixels 'differ'.")
    ap.add_argument("--out-dir", default=None,
                    help="Write before/after face crops here for eyeballing")
    args = ap.parse_args()

    if args.frames:
        return compare_frames(Path(args.frames), args.out_dir)
    if not (args.raw and args.restored):
        print("ERROR: pass two videos, or --frames <work_dir>", file=sys.stderr)
        return 2
    print("NOTE: comparing two encoded videos. Different encoders make ~98% of\n"
          "      pixels differ before anything is restored; localisation and\n"
          "      sharpness below are unreliable. Prefer --frames.\n")
    raw, rest = Path(args.raw), Path(args.restored)
    for p in (raw, rest):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 2

    pr, pt = probe(raw), probe(rest)
    print("== container ==")
    for label, d in (("raw     ", pr), ("restored", pt)):
        print(f"  {label}  {d.get('width',['?'])[0]}x{d.get('height',['?'])[0]} "
              f"fps={d.get('r_frame_rate',['?'])[0]} "
              f"dur={float(d.get('duration',['0'])[0]):.3f}s "
              f"audio={d.get('sample_rate',['-'])[0]}Hz/"
              f"{d.get('channels',['-'])[0]}ch")

    ca, cb = cv2.VideoCapture(str(raw)), cv2.VideoCapture(str(rest))
    na = int(ca.get(cv2.CAP_PROP_FRAME_COUNT))
    nb = int(cb.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  frames: raw={na} restored={nb}"
          + ("" if na == nb else "   <-- MISMATCH"))

    thr = args.noise_threshold
    idxs = np.linspace(0, min(na, nb) - 1, args.sample).astype(int)
    rows, union, accum = [], None, None
    for i in idxs:
        ca.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok_a, fa = ca.read()
        cb.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok_b, fb = cb.read()
        if not (ok_a and ok_b) or fa.shape != fb.shape:
            continue
        diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(2)
        mask = diff > thr
        union = mask if union is None else (union | mask)
        # Accumulate the magnitude of the above-noise change only, so the
        # localisation below is driven by the edit and not by the codec.
        energy = np.where(mask, diff, 0).astype(np.float64)
        accum = energy if accum is None else (accum + energy)
        rows.append((int(i), mask.mean(), psnr(fa, fb), ssim(fa, fb), fa, fb, mask))
    ca.release(); cb.release()

    if not rows:
        print("ERROR: no comparable frames", file=sys.stderr); return 2

    print(f"\n== whole frame, {len(rows)} sampled frames (noise thr >{thr}) ==")
    print(f"  changed pixels : {np.mean([r[1] for r in rows]) * 100:.2f}% "
          f"(min {min(r[1] for r in rows) * 100:.2f}%, "
          f"max {max(r[1] for r in rows) * 100:.2f}%)")
    print(f"  PSNR           : {np.mean([r[2] for r in rows]):.2f} dB")
    print(f"  SSIM           : {np.mean([r[3] for r in rows]):.4f}")

    # Localise the edit by where the change *energy* is, not by a bounding box
    # over every changed pixel. A union bbox is useless here: a dozen stray
    # pixels of surviving codec noise, anywhere in the frame, stretch it to the
    # full frame and hide a result that is in fact confined to one face.
    # Collapse the accumulated difference onto each axis and keep the span
    # holding the middle `--energy` fraction of it.
    if accum.max() == 0:
        print(f"\n!! NO MATERIAL CHANGE: nothing moved by more than {thr}/255 "
              f"in any sampled frame. The pass only re-encoded the input.")
        return 1
    h, w = accum.shape

    def span(marginal: np.ndarray, keep: float) -> tuple:
        total = marginal.sum()
        if total <= 0:
            return 0, len(marginal)
        c = np.cumsum(marginal) / total
        tail = (1.0 - keep) / 2
        lo = int(np.searchsorted(c, tail))
        hi = int(np.searchsorted(c, 1.0 - tail)) + 1
        return lo, min(hi, len(marginal))

    x0, x1 = span(accum.sum(0), args.energy)
    y0, y1 = span(accum.sum(1), args.energy)
    inside = accum[y0:y1, x0:x1].sum() / accum.sum()
    print(f"\n== where the change actually is ==")
    print(f"  {args.energy * 100:.0f}% of the change energy lies in "
          f"x={x0}..{x1} y={y0}..{y1}")
    print(f"  that box is {(x1 - x0) * (y1 - y0) / (w * h) * 100:.2f}% of the "
          f"{w}x{h} frame and holds {inside * 100:.1f}% of all change")
    print(f"\n== inside that box ==")
    crop_rows = [(r[0], psnr(r[4][y0:y1, x0:x1], r[5][y0:y1, x0:x1]),
                  ssim(r[4][y0:y1, x0:x1], r[5][y0:y1, x0:x1]),
                  r[6][y0:y1, x0:x1].mean()) for r in rows]
    print(f"  changed pixels inside it: "
          f"{np.mean([c[3] for c in crop_rows]) * 100:.1f}%")
    print(f"  PSNR over it            : {np.mean([c[1] for c in crop_rows]):.2f} dB")
    print(f"  SSIM over it            : {np.mean([c[2] for c in crop_rows]):.4f}")

    # Did the face gain detail? Variance of the Laplacian is the standard
    # cheap sharpness proxy. A restoration pass that is working should raise
    # it: that is the whole claim being made about the output. A pass that
    # merely re-encoded, or blurred further, will not.
    def sharpness(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    sa = [sharpness(r[4][y0:y1, x0:x1]) for r in rows]
    sb = [sharpness(r[5][y0:y1, x0:x1]) for r in rows]
    ma, mb = float(np.mean(sa)), float(np.mean(sb))
    print(f"\n== detail in that region (variance of Laplacian) ==")
    print(f"  raw      : {ma:8.1f}")
    print(f"  restored : {mb:8.1f}   ({(mb / ma - 1) * 100:+.1f}%)")
    better = sum(1 for x, y in zip(sa, sb) if y > x)
    print(f"  sharper in {better}/{len(sa)} sampled frames")
    if mb <= ma:
        print("  !! the restored face is NOT sharper than the input -- "
              "the pass is not doing what it claims")

    if args.out_dir:
        od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
        pad = 24
        cy0, cy1 = max(0, y0 - pad), min(h, y1 + pad)
        cx0, cx1 = max(0, x0 - pad), min(w, x1 + pad)
        for i, _, _, _, fa, fb, _ in rows:
            a, b = fa[cy0:cy1, cx0:cx1], fb[cy0:cy1, cx0:cx1]
            gap = np.full((a.shape[0], 8, 3), 255, np.uint8)
            cv2.imwrite(str(od / f"face_{i:06d}_raw_vs_restored.png"),
                        np.hstack([a, gap, b]))
        print(f"\n  wrote {len(rows)} before/after crops to {od}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
