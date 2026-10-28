"""Hold a command until BOTH GPUs have been continuously free for N minutes.

The machine is shared. On 2026-10-19 two `train_sites.py` jobs from another
account held ~11.8 GiB on each card for over an hour, and a lip-sync batch
launched into that would either OOM or crawl. So the batch waits.

**Continuously** is the point, and it is why this is not a one-shot check.
A training script between epochs, or a job that has just been killed and is
about to be relaunched by a supervisor, both show an idle card for a few
seconds. The free-since clock RESETS the moment either GPU looks busy, so a
brief gap cannot be mistaken for the machine being available.

Free is `backend.gpu.free_devices()`, not a memory threshold: it already
knows to ignore this process group's own PIDs, so the watcher does not see
itself or anything it launched as a blocker.

    $PY scripts/gpu_watcher.py --minutes 30 -- bash scripts/run_lipsync.sh musetalk <stem>

Writes a line on every state change, not every poll, so the log stays
readable across a multi-hour wait and still records exactly when the
machine freed up and who was holding it.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("gpu_watcher")


def holders() -> list[str]:
    """Who is on the cards right now, as readable lines."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines():
        pid = line.split(",")[0].strip()
        try:
            who = subprocess.run(["ps", "-o", "user=,args=", "-p", pid],
                                 capture_output=True, text=True,
                                 timeout=10).stdout.strip()
        except Exception:                                # noqa: BLE001
            who = "?"
        rows.append(f"{line.strip()}  {who[:90]}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="how long both GPUs must stay free before launching")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--need", type=int, default=2,
                    help="how many GPUs must be free (2 = both)")
    ap.add_argument("--max-wait-hours", type=float, default=72.0)
    ap.add_argument("--log", default="logs/gpu_watcher.log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.FileHandler(a.log, mode="a"),
                  logging.StreamHandler(sys.stdout)])

    from backend.gpu import free_devices

    need_s = a.minutes * 60.0
    deadline = time.time() + a.max_wait_hours * 3600.0
    free_since: float | None = None
    last_state: str | None = None

    log.info("watcher armed: need %d GPU(s) free for %.0f min, poll %.0fs",
             a.need, a.minutes, a.poll)
    log.info("will run: %s", " ".join(cmd))
    for h in holders():
        log.info("  currently on GPU: %s", h)

    while True:
        if time.time() > deadline:
            log.error("gave up after %.1f h without a free window",
                      a.max_wait_hours)
            return 3
        try:
            free = free_devices()
        except Exception as e:                           # noqa: BLE001
            log.warning("could not read GPU state (%s); treating as busy", e)
            free = []

        ok = len(free) >= a.need
        if not ok:
            if last_state != "busy":
                log.info("GPUs busy (free=%s) -- clock reset", free)
                for h in holders():
                    log.info("  held by: %s", h)
                last_state = "busy"
            free_since = None
        else:
            if free_since is None:
                free_since = time.time()
                log.info("GPUs free (%s) -- starting %.0f min countdown",
                         free, a.minutes)
                last_state = "free"
            held = time.time() - free_since
            if held >= need_s:
                log.info("free for %.1f min -- launching", held / 60.0)
                break
            if last_state == "free" and int(held) % 600 < a.poll:
                log.info("  still free, %.1f/%.0f min", held / 60.0, a.minutes)
        time.sleep(a.poll)

    log.info("=== launching: %s ===", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    log.info("=== finished rc=%d after %.1f min ===", rc, (time.time() - t0) / 60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
