"""Join a lane's 30-second segments back into one continuous video.

Order comes from the START SECOND parsed out of the filename, not from
`sorted()`: the clips are named `<tag>_<start>to<end>.mp4`, so a plain string
sort puts 120to150 before 30to60 and the result plays out of order while
looking perfectly fine.

Joined with the concat FILTER rather than ffmpeg's concat demuxer. The
demuxer needs byte-identical stream parameters and DROPS a mismatched input
without failing -- measured 2026-10-19, two 480-frame shards joined to 504
frames at rc=0. The filter decodes everything into one graph and normalises
timebases. The frame count is checked against the sum of the inputs either
way, because a silent join is the failure mode that matters here.
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))
from backend.lipsync import _frame_count                 # noqa: E402
from backend.media_io import _ffmpeg_bin, _ffprobe_bin   # noqa: E402

FF, FP = _ffmpeg_bin(), _ffprobe_bin()
KIND = sys.argv[1] if len(sys.argv) > 1 else "dub"       # dub | lipsync
OUT = ROOT / "data/clips/seg30_2026-10-20/joined"
OUT.mkdir(parents=True, exist_ok=True)

LANES = {"batman": "batman", "iit": "iit",
         "dk": "L3oOldViIgY", "w4l11": "IBfVkrvqAI"}


def start_of(name: str) -> int:
    m = re.search(r"_(\d+)to\d+", name)
    return int(m.group(1)) if m else 0


for lane, stem_prefix in LANES.items():
    parts = []
    for j in sorted((ROOT / "data/jobs").glob(f"seg30_{lane}_{stem_prefix}_*")):
        if KIND == "dub":
            f = next(iter(j.glob("mix/*.dubbed.mp4")), None)
        else:
            f = next((p for p in j.glob("lipsync/*.mp4")
                      if "shards" not in str(p) and "conform" not in str(p)), None)
        if f:
            parts.append((start_of(j.name), f))
    parts.sort(key=lambda t: t[0])
    if not parts:
        print(f"{lane:8} {KIND}: nothing to join")
        continue

    want = sum(_frame_count(f) for _, f in parts)
    ins, labels = [], []
    for i, (_s, f) in enumerate(parts):
        ins += ["-i", str(f)]
        labels.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}];")
    graph = ("".join(labels)
             + "".join(f"[v{i}][a{i}]" for i in range(len(parts)))
             + f"concat=n={len(parts)}:v=1:a=1[v][a]")
    out = OUT / f"{lane}_{KIND}_joined.mp4"
    r = subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", *ins,
                        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{lane:8} {KIND}: FAILED {r.stderr.strip()[-160:]}")
        continue
    got = _frame_count(out)
    dur = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout.strip()
    flag = "" if got >= want else f"  !! lost {want - got} frame(s)"
    print(f"{lane:8} {KIND}: {len(parts):2d} parts -> {got} frames "
          f"(want {want}), {float(dur):.2f}s{flag}")
    print(f"         {out.relative_to(ROOT)}")
