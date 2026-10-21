"""Re-run ONLY lip-sync over segments whose dub already exists.

The 2026-10-20 sequential run produced 32 dubs and 2 renders: every segment
reached stage 9h and was refused because another account's process held the
card the lane was pinned to. The dubs are correct and were expensive --
ASR, MT and TTS per segment -- so re-running the whole pipeline to recover
the cheap half would pay for the costly half twice.

Pinned to one GPU and strictly sequential, matching how the lane ran.
"""
import json, logging, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/data1/dipak_data/dubbingstudio")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUSETALK_FP16", "1")

GPU = sys.argv[1] if len(sys.argv) > 1 else "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)-24s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(ROOT / "logs/seg30_lipsync_only.log", mode="a"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("relipsync")

from backend.lipsync import run_lipsync, _frame_count
from backend.gpu import free_devices, visible_cuda_devices

log.info("visible=%s free=%s (CUDA_VISIBLE_DEVICES=%s)",
         visible_cuda_devices(), free_devices(), GPU)

CLIPS = ROOT / "data/clips/seg30_2026-10-20"
jobs = sorted(p for p in (ROOT / "data/jobs").glob("seg30_*") if p.is_dir())
todo = []
for j in jobs:
    if list(j.glob("lipsync/*.mp4")):
        continue                              # already rendered
    mp4 = next(iter(j.glob("mix/*.dubbed.mp4")), None)
    wav = next(iter(j.glob("mix/*.dubbed.wav")), None)
    if not mp4 or not wav:
        log.warning("%s: no dub on disk, skipping", j.name)
        continue
    stem = mp4.name.replace(".dubbed.mp4", "")
    src = CLIPS / f"{stem}.mp4"
    if not src.is_file():
        log.warning("%s: source clip missing at %s", j.name, src)
        continue
    todo.append((j, stem, src, wav, mp4))

log.info("=== %d segment(s) need a render ===", len(todo))
ok = fail = 0
for i, (j, stem, src, wav, pre) in enumerate(todo, 1):
    merged = next(iter(j.glob("mt/*.mt.json")), None) or (j / "relipsync.json")
    if not merged.exists():
        merged.write_text('{"segments": [], "extra": {}}')
    log.info("[%d/%d] %s (%d frames)", i, len(todo), stem, _frame_count(src))
    t0 = time.time()
    try:
        r = run_lipsync(merged, model="musetalk", source_video=src,
                        driving_audio=wav, pre_muxed_video=pre,
                        out_dir=j / "lipsync", num_gpus=1)
        out = Path(r.out_video) if r.out_video else None
        if out and out.is_file() and out.stat().st_size > 0:
            ok += 1
            log.info("    ok in %.0fs -> %s (%d frames)", time.time() - t0,
                     out.name, _frame_count(out))
        else:
            fail += 1
            log.error("    FAILED in %.0fs: %s", time.time() - t0, r.error)
    except Exception as e:                               # noqa: BLE001
        fail += 1
        log.error("    FAILED in %.0fs: %s", time.time() - t0, e)
    (ROOT / "logs/seg30_lipsync_only.STATUS").write_text(
        f"{time.strftime('%FT%T')} relipsync {i}/{len(todo)} ok={ok} fail={fail}\n")
log.info("=== DONE: %d ok, %d failed ===", ok, fail)
