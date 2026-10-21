"""A/B the lazy-frame change in MuseTalk: wall time and peak RAM.

    $PY scripts/ab_lazy_frames.py [--seconds 120] [--gpu 1]

Renders the SAME clip twice -- once with `MUSETALK_LAZY_FRAMES=0` (today's
behaviour: `read_imgs()` loads every frame into a list) and once with `=1`
(each frame decoded from its PNG on access) -- and reports wall time, peak
RSS of the render process, and whether the two outputs are identical.

Three things make the comparison fair, and each one matters:

1. **The coord cache is pre-warmed first.** The lazy path only replaces
   `read_imgs()`, which is reached on the `--use_saved_coord` branch. On a
   cache MISS `get_landmark_and_bbox` loads every frame itself, so an
   un-warmed A/B would measure that spike in both arms and show no
   difference. The warm-up render is discarded.
2. **Segmenting is disabled** (`max_segment_s=0`). Segmenting exists to
   bound exactly the memory being measured; leaving it on would cap both
   arms at the same number and hide the effect.
3. **One GPU.** Two shards are two processes with half the frames each,
   which changes what "peak RSS" means mid-comparison.

The output check is not a formality. Lazy access must be a pure memory
optimisation: if frame `i` ever resolves to a different image than it did
eagerly -- a slice that is not reversed the same way, an index off by one
through `frame_list_cycle` -- the render is silently wrong rather than
failed, which is the expensive kind of bug.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))

SRC_4MIN = ROOT / "data/clips/iit_1080p_4min.mp4"
DUB_4MIN = ROOT / "data/clips/seg30_2026-10-20/joined/iit_dub_joined.mp4"
OUT = ROOT / "data/gender_bench/cuts/ab_lazy_frames"


class Sampler(threading.Thread):
    """Peak summed RSS of the MuseTalk render processes."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.peak_kb = 0

    def run(self):
        while not self.stop:
            try:
                r = subprocess.run(["ps", "-eo", "rss,args", "--no-headers"],
                                   capture_output=True, text=True, timeout=10)
                tot = sum(int(l.split()[0]) for l in r.stdout.splitlines()
                          if "scripts.inference" in l)
                self.peak_kb = max(self.peak_kb, tot)
            except Exception:
                pass
            time.sleep(1.0)


def cut(src: Path, dst: Path, seconds: int) -> Path:
    from backend.media_io import _ffmpeg_bin
    if dst.is_file() and dst.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src), "-t", str(seconds),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", str(dst)], check=True)
    return dst


def extract_wav(video: Path, dst: Path) -> Path:
    """The driving audio must be a WAV, not the dubbed MP4.

    Passing the .mp4 does NOT fail loudly: MuseTalk's Whisper front-end
    accepts the path, produces degenerate features, and renders a
    near-neutral mouth that composites back almost identical to the
    source. Measured 2026-10-20 -- mouth-region MAE 1.79 and max pixel
    diff 27, against 7.38 / 114 for a known-good render driven by a WAV.
    The run looks successful at every level: exit 0, right frame count,
    right duration, a playable file. Only the mouth is wrong.
    """
    from backend.media_io import _ffmpeg_bin
    if dst.is_file() and dst.stat().st_size:
        return dst
    subprocess.run([_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(video), "-vn", "-ac", "2", "-ar", "48000",
                    "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def render(tag: str, *, lazy: bool, src: Path, dub: Path, wav: Path, gpu: int,
           coord_cache: bool):
    """One render. Returns (wall_s, peak_rss_gib, out_path, error)."""
    from backend.lipsync import run_lipsync

    job = OUT / tag
    job.mkdir(parents=True, exist_ok=True)
    merged = job / "merged.json"
    merged.write_text(json.dumps({"segments": [], "extra": {}}))

    os.environ["MUSETALK_LAZY_FRAMES"] = "1" if lazy else "0"
    os.environ["MUSETALK_FP16"] = "1"

    s = Sampler(); s.start(); t0 = time.time()
    r = run_lipsync(
        merged, model="musetalk", source_video=src, driving_audio=wav,
        pre_muxed_video=dub, out_dir=job,
        num_gpus=1,
        # See the module docstring: segmenting would cap both arms.
        max_segment_s=0,
        lipsync_kwargs={"batch_size": 8, "cuda_device": gpu,
                        "use_coord_cache": coord_cache},
    )
    wall = time.time() - t0
    s.stop = True; s.join(timeout=3)
    out = Path(r.out_video) if r.out_video else None
    ok = out and out.is_file() and out.stat().st_size > 0
    return (round(wall, 1), round(s.peak_kb / 1048576.0, 2),
            out if ok else None, r.error)


def lipsync_strength(src: Path, out: Path) -> str:
    """Did the render actually REPAINT the mouth, or just re-encode?

    Comparing the two arms against each other proves they are consistent,
    not that either worked -- identical-but-wrong passes that check. This
    compares the mouth region against the SOURCE, which is the question.

    Calibrated on this host, 2026-10-20: a known-good MuseTalk render
    scored mouth MAE 7.38 / max pixel 114. The same clip driven by an MP4
    instead of a WAV -- a silent failure that exits 0 with the right frame
    count -- scored 1.79 / 27. The floor sits between them, nearer the
    failure, because the point is to catch "nothing happened" rather than
    to grade quality.
    """
    try:
        import cv2
        import numpy as np
        sys.path.insert(0, str(ROOT))
        from backend.gpu import ensure_onnxruntime_cuda
        ensure_onnxruntime_cuda()
        from insightface.app import FaceAnalysis
    except Exception as e:                               # noqa: BLE001
        return f"not checked ({e})"

    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    cs, co = cv2.VideoCapture(str(src)), cv2.VideoCapture(str(out))
    n = min(int(cs.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(co.get(cv2.CAP_PROP_FRAME_COUNT)))
    maes, peaks = [], []
    for i in range(50, n, max(1, n // 12)):
        cs.set(cv2.CAP_PROP_POS_FRAMES, i); oka, a = cs.read()
        co.set(cv2.CAP_PROP_POS_FRAMES, i); okb, b = co.read()
        if not (oka and okb):
            continue
        faces = app.get(a)
        if not faces:
            continue
        f = max(faces, key=lambda x: x.bbox[3] - x.bbox[1])
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        my1 = y1 + int((y2 - y1) * 0.55)
        d = np.abs(a[my1:y2, x1:x2].astype("f4") - b[my1:y2, x1:x2].astype("f4"))
        maes.append(float(d.mean())); peaks.append(float(d.max()))
    cs.release(); co.release()
    if not maes:
        return "not checked (no face found in the source)"
    mae, peak = float(np.mean(maes)), float(np.mean(peaks))
    verdict = ("LIP-SYNCED" if mae >= 4.0 and peak >= 60
               else "NOT LIP-SYNCED -- the mouth barely changed")
    return (f"{verdict}  (mouth MAE {mae:.2f}, max pixel {peak:.0f}; "
            f"known-good is ~7.4 / ~114, a silent failure ~1.8 / ~27)")


def compare(a: Path, b: Path) -> str:
    """Are the two renders the same pictures? Sampled, not exhaustive."""
    try:
        import cv2
        import numpy as np
    except Exception as e:                               # noqa: BLE001
        return f"not compared ({e})"
    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    na = int(ca.get(cv2.CAP_PROP_FRAME_COUNT))
    nb = int(cb.get(cv2.CAP_PROP_FRAME_COUNT))
    if na != nb:
        return f"DIFFERENT frame counts: {na} vs {nb}"
    worst = 0.0
    for i in range(0, max(1, na), max(1, na // 20)):
        ca.set(cv2.CAP_PROP_POS_FRAMES, i); oka, fa = ca.read()
        cb.set(cv2.CAP_PROP_POS_FRAMES, i); okb, fb = cb.read()
        if not (oka and okb):
            continue
        worst = max(worst, float(np.abs(fa.astype("f4") - fb.astype("f4")).mean()))
    ca.release(); cb.release()
    verdict = "identical" if worst < 1.0 else "DIFFERENT"
    return f"{verdict} ({na} frames, worst sampled MAE {worst:.3f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()

    import logging
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-22s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(OUT / "ab.log", mode="w"),
                  logging.StreamHandler(sys.stdout)])

    for p in (SRC_4MIN, DUB_4MIN):
        if not p.is_file():
            print(f"missing input: {p}")
            return 2

    src = cut(SRC_4MIN, OUT / f"src_{a.seconds}s.mp4", a.seconds)
    dub = cut(DUB_4MIN, OUT / f"dub_{a.seconds}s.mp4", a.seconds)
    wav = extract_wav(dub, OUT / f"dub_{a.seconds}s.wav")
    print(f"\nclip: {src.name}  face={dub.name}  audio={wav.name} "
          f"({a.seconds}s), gpu {a.gpu}\n")

    # Warm the coord cache so BOTH arms take the --use_saved_coord branch.
    print("=== warm-up (populating the coord cache; result discarded) ===")
    w = render("warm", lazy=False, src=src, dub=dub, wav=wav,
               gpu=a.gpu, coord_cache=True)
    print(f"    warm-up done in {w[0]}s (error={w[3]})\n")

    rows = {}
    for tag, lazy in (("eager", False), ("lazy", True)):
        print(f"=== {tag} (MUSETALK_LAZY_FRAMES={int(lazy)}) ===")
        wall, rss, out, err = render(tag, lazy=lazy, src=src, dub=dub,
                                     wav=wav, gpu=a.gpu, coord_cache=True)
        rows[tag] = {"wall_s": wall, "peak_rss_gib": rss,
                     "out": str(out) if out else None, "error": err}
        print(f"    wall {wall}s   peak RSS {rss} GiB   error={err}\n")

    print("=" * 62)
    print(f"{'':8} {'wall (s)':>10} {'peak RSS (GiB)':>16}")
    for tag in ("eager", "lazy"):
        r = rows[tag]
        print(f"{tag:8} {r['wall_s']:>10} {r['peak_rss_gib']:>16}")
    e, l = rows["eager"], rows["lazy"]
    if e["wall_s"] and l["wall_s"]:
        print(f"\n  wall     {l['wall_s'] - e['wall_s']:+.1f}s "
              f"({100 * (l['wall_s'] / e['wall_s'] - 1):+.1f}%)")
    if e["peak_rss_gib"] and l["peak_rss_gib"]:
        print(f"  peak RSS {l['peak_rss_gib'] - e['peak_rss_gib']:+.2f} GiB "
              f"({100 * (l['peak_rss_gib'] / e['peak_rss_gib'] - 1):+.1f}%)")

    if e["out"] and l["out"]:
        print(f"\n  eager vs lazy : {compare(Path(e['out']), Path(l['out']))}")
        # The gate. Two identical arms prove consistency, not correctness --
        # identical-but-not-lip-synced passes the comparison above.
        print(f"  did it sync?  : {lipsync_strength(src, Path(e['out']))}")
    else:
        print("\n  outputs: NOT COMPARED -- a render failed")

    (OUT / "ab_lazy_frames.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {OUT / 'ab_lazy_frames.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
