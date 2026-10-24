#!/usr/bin/env python
"""Marked stills of a job's ASD output, at chosen moments.

    $PY scripts/asd_frames.py --job <job_id> --at 17.0 25.0 --count 4

Writes `<job>/review/frames/t17.000_f0425.jpg` and so on, and prints one
line per frame saying what it shows.

Why stills and not the review video
-----------------------------------
`ui/bbox_video.py` already renders the whole clip with its boxes drawn, and
that is the right tool for watching a shot. It is the wrong tool for
answering "who did ASD think was talking at 17 seconds": a 60 s 4K render is
minutes of decode and re-encode, and the answer is four frames.

**The marking itself is NOT reimplemented here.** `load_overlay`,
`verdict` and `draw_frame` are imported from `ui.bbox_video`, so a still and
the review video cannot disagree about who the pipeline believed was
speaking — which would make this script worse than useless, since its whole
job is to show what the pipeline actually decided.

What the colours mean
---------------------
    green   this face is the one assigned to the speaking segment, on screen
    red     another track owns this moment (or this one owns it but is off
            screen) -- the label reads `-` or `<score> off`
    grey    no segment covers this frame, so nothing is known and the script
            claims nothing. `?`

The label is `FACE_nn <score> SPK_nn`.

Decoding is SEQUENTIAL, never a seek. `cv2.CAP_PROP_POS_FRAMES` lands on the
nearest keyframe for some codecs, and an off-by-a-few-frames still would be
annotated with the boxes of a frame it is not showing — the one failure that
would make the output confidently wrong rather than merely absent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui import bbox_video as B                            # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", help="job id, or the newest UI run when omitted")
    ap.add_argument("--range", dest="ranges", type=float, nargs=2,
                    action="append", metavar=("START", "END"),
                    help="a window to sample, in seconds. Repeatable: "
                         "--range 17 19 --range 25 27")
    ap.add_argument("--at", type=float, nargs="+", metavar="SECONDS",
                    help="single moments to mark, in seconds")
    ap.add_argument("--count", type=int, default=1,
                    help="frames per window (spread evenly, both ends "
                         "included) or per moment (consecutive)")
    ap.add_argument("--step-s", type=float, default=None,
                    help="for --at only: gap between its frames. The default "
                         "is one frame, i.e. consecutive frames AT the moment.")
    ap.add_argument("--max-edge", type=int, default=1920,
                    help="downscale the longest edge to this (0 = full size). "
                         "A 4K still is ~8 MB; 1920 is plenty to read a box.")
    ap.add_argument("--out", default=None,
                    help="output directory (default <job>/review/frames)")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args(argv)

    import cv2

    data_dir = Path(args.data_dir)
    jobs = data_dir / "jobs"
    if args.job:
        job_root = jobs / args.job
    else:
        # Newest job that actually HAS the three things needed, rather than
        # newest full stop: a run that died before stage 5 has no tracks and
        # would fail with a message about the wrong job.
        cands = [d for d in jobs.iterdir() if d.is_dir()
                 and any(d.glob("faces/*.npz"))
                 and any(d.glob("asd/*.assignments.json"))
                 and any(d.glob("videos/*.mp4"))]
        if not cands:
            print("no job with face tracks + ASD + a clip", file=sys.stderr)
            return 2
        job_root = max(cands, key=lambda d: d.stat().st_mtime)
    if not job_root.is_dir():
        print(f"no such job: {job_root}", file=sys.stderr)
        return 2

    src = B.source_clip(job_root)
    if src is None:
        print(f"no clip in {job_root}/videos/", file=sys.stderr)
        return 2
    try:
        box_at, spk_of, spans, fps = B.load_overlay(job_root, src)
    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        return 2

    try:
        from ui.speakers import read_labels
        labels = read_labels(job_root)
    except Exception:                                      # noqa: BLE001
        labels = {}

    n = max(1, args.count)
    wanted: dict = {}
    # A WINDOW is sampled across its whole length, both ends included, so
    # `--range 17 19 --count 4` gives 17.00, 17.67, 18.33, 19.00. Four
    # consecutive frames would all be the same 1/25 s of video and would say
    # nothing about whether the verdict holds across the window, which is
    # the question a window is being asked.
    for a, b in (args.ranges or []):
        a, b = sorted((float(a), float(b)))
        for k in range(n):
            t = a if n == 1 else a + (b - a) * k / (n - 1)
            wanted.setdefault(int(round(t * fps)), t)
    # A MOMENT is consecutive frames at that moment, unless --step-s says
    # otherwise.
    step = int(round((args.step_s or 0) * fps)) or 1
    for t in (args.at or []):
        base = int(round(float(t) * fps))
        for k in range(n):
            wanted.setdefault(base + k * step, float(t))

    if not wanted:
        print("nothing to do: pass --range and/or --at", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else (job_root / "review" / "frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last = max(wanted)
    if total and last >= total:
        print(f"warning: frame {last} is past the end of a {total}-frame "
              f"clip ({total / fps:.2f}s); it will be skipped",
              file=sys.stderr)

    print(f"job   {job_root.name}")
    print(f"clip  {src.name}  {fps:g} fps  {total} frames "
          f"({total / fps:.2f}s)")
    print(f"out   {out_dir}")
    print()

    written: list = []
    done: set = set()
    idx = 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                B.draw_frame(frame, idx, box_at=box_at, spk_of=spk_of,
                             spans=spans, labels=labels)
                if args.max_edge:
                    h, w = frame.shape[:2]
                    scale = min(1.0, args.max_edge / float(max(w, h)))
                    if scale < 1.0:
                        frame = cv2.resize(
                            frame, (int(round(w * scale)), int(round(h * scale))),
                            interpolation=cv2.INTER_AREA)
                dest = out_dir / f"t{idx / fps:07.3f}_f{idx:05d}.jpg"
                cv2.imwrite(str(dest), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                written.append(dest)
                done.add(idx)
                boxes = box_at.get(idx, [])
                marks = ", ".join(
                    f"{tid}={B.verdict(spans, tid, idx)[1]}"
                    f"{'/' + spk_of[tid] if spk_of.get(tid) else ''}"
                    for tid, _ in boxes) or "no faces tracked"
                print(f"{idx / fps:7.3f}s  f{idx:<6d} {len(boxes)} face(s)  "
                      f"{marks}")
            idx += 1
    finally:
        cap.release()

    print()
    print(f"wrote {len(written)} of {len(wanted)} frame(s) to {out_dir}")
    missing = sorted(set(wanted) - done)
    if missing:
        # Said out loud rather than left to be noticed by counting files: a
        # window that runs past the end of the clip silently produces fewer
        # stills than asked for, and the gap is exactly where the answer was.
        print("not written (past the end of the clip): "
              + ", ".join(f"{m / fps:.3f}s" for m in missing),
              file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
