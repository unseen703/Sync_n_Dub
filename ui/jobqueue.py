"""A serial queue whose state lives on disk, not in this process.

Two GPUs with 16 GiB each, and MuseTalk alone peaks at 15.3 GiB of one of
them. Running two jobs at once is not a throughput win, it is an OOM. So
`DUBBING_UI_MAX_CONCURRENT` defaults to 1.

**Occupancy is read from `data/jobs/*/progress.json`, never from a counter in
the server.** The runner is detached — it is launched into tmux, or at worst
with `start_new_session=True` — precisely so a four-hour render survives the
UI being restarted. A counter in server memory would reset on that restart
and happily launch a second job onto a GPU the first one is still holding.
The disk has the answer, and `os.kill(pid, 0)` says whether it is still true.

That makes the queue tolerant of exactly the failures that happen here: a
`kill -9`'d runner leaves `state: "running"` behind, and the next occupancy
scan notices the pid is gone and does not count it.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._io import env_float as _env_float, env_int as _env_int
from .progress import PROGRESS_FILE, write_atomic

log = logging.getLogger("ui.jobqueue")

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_SH = REPO_ROOT / "scripts" / "ui_run_job.sh"

#: The repo's long-run convention (`.claude/skills/long-run`): detached jobs
#: live in one tmux session so a human can attach to a running render.
TMUX_SESSION = os.environ.get("DUBBING_UI_TMUX_SESSION", "dubsync")

MAX_CONCURRENT_ENV = "DUBBING_UI_MAX_CONCURRENT"

#: How long a spawned-but-not-yet-registered job keeps holding its slot.
#:
#: `spawn()` returns in microseconds; the job only becomes `running` on disk
#: once `ProgressWriter.__init__` runs INSIDE the new process, after
#: `ui_run_job.sh` has started python and imported torch. Without a marker
#: covering that window, occupancy reads 0 the whole time and `pump()` fires
#: the entire backlog at once (measured: 10 queued -> 10 launched, with
#: max_concurrent=1). The grace is generous because overshooting only delays
#: the next job, while undershooting re-creates the flood.
SPAWN_GRACE_ENV = "DUBBING_UI_SPAWN_GRACE_S"


def max_concurrent() -> int:
    return max(1, _env_int(MAX_CONCURRENT_ENV, 1))


def spawn_grace_s() -> float:
    return max(1.0, _env_float(SPAWN_GRACE_ENV, 300.0))


# ---------------------------------------------------------------------------
# occupancy, from disk
# ---------------------------------------------------------------------------

def pid_alive(pid: Optional[int]) -> bool:
    """Does a process with this pid exist on this machine?

    Signal 0 does not deliver anything; it only runs the kernel's permission
    and existence checks. `EPERM` means the process exists and is not ours,
    which still counts as occupied.
    """
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (OSError, TypeError, ValueError):
        return False
    return True


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def scan(jobs_dir: Path) -> List[Dict[str, Any]]:
    """Every job folder that has a `progress.json`, as raw docs.

    Deliberately NOT `read_progress`: that normaliser rewrites a dead-pid
    `running` into `failed`, which is right for display and wrong here —
    the queue needs the stored state and the pid separately.
    """
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.is_dir():
        return []
    out = []
    for p in sorted(jobs_dir.glob("*/" + PROGRESS_FILE)):
        doc = _read(p)
        if doc is None:
            continue
        doc.setdefault("job_id", p.parent.name)
        doc["_path"] = p
        out.append(doc)
    return out


def _age_s(ts: Optional[str]) -> Optional[float]:
    """Seconds since an ISO-8601 timestamp, or None if it will not parse."""
    if not ts:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds()


def spawn_pending(doc: Dict[str, Any]) -> bool:
    """Has this job been launched but not yet reported itself as running?

    Occupancy has to count these. The alternative -- trusting `state` alone --
    is what let one `pump()` launch a whole backlog onto two GPUs.

    The grace expires so a runner that never came up (a bad interpreter, a
    crash before `ProgressWriter`) releases the slot instead of wedging the
    queue forever, which is the failure mode this module's header warns about
    in the other direction.
    """
    age = _age_s(doc.get("spawned_at"))
    return age is not None and age < spawn_grace_s()


def running(jobs_dir: Path) -> List[Dict[str, Any]]:
    """Jobs holding a slot: registered and alive, or just launched.

    Two shapes count, and leaving either out has been a live bug:
      * `state: running` with a pid that still exists -- the steady state.
      * `state: queued` with a recent `spawned_at` -- launched, still booting.
    """
    out: List[Dict[str, Any]] = []
    for d in scan(jobs_dir):
        state = d.get("state")
        if state == "running" and pid_alive(d.get("pid")):
            out.append(d)
        elif state == "queued" and spawn_pending(d):
            out.append(d)
    return out


def queued(jobs_dir: Path) -> List[Dict[str, Any]]:
    """Jobs waiting for a slot, oldest first.

    A job `pump()` has already launched is NOT waiting, even though its doc
    still reads `queued` until its runner boots -- excluding it here is what
    stops a second pump from spawning a duplicate onto the same job folder.
    """
    rows = [d for d in scan(jobs_dir)
            if d.get("state") == "queued" and not spawn_pending(d)]
    rows.sort(key=lambda d: str(d.get("queued_at") or d.get("started_at") or ""))
    return rows


def occupancy(jobs_dir: Path) -> int:
    return len(running(jobs_dir))


def has_capacity(jobs_dir: Path) -> bool:
    return occupancy(jobs_dir) < max_concurrent()


def position(jobs_dir: Path, job_id: str) -> int:
    """1-based place in the queue; 0 when this job is not queued."""
    for i, d in enumerate(queued(jobs_dir), start=1):
        if d.get("job_id") == job_id:
            return i
    return 0


# ---------------------------------------------------------------------------
# marking and launching
# ---------------------------------------------------------------------------

def mark_queued(job_root: Path, job_id: str, *,
                planned: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Write a `queued` progress doc so the job is visible before it starts.

    `pid` is None on purpose: a queued job is not a process, and writing the
    *server's* pid here would make the occupancy scan count it as running.
    """
    doc = {
        "version": 1,
        "job_id": job_id,
        "kind": "run",
        "pid": None,
        "state": "queued",
        "started_at": None,
        "queued_at": _now(),
        "updated_at": _now(),
        "heartbeat": None,
        "current": None,
        "log": "run.log",
        "error": None,
        "stages": list(planned or []),
        "n_done": 0,
        "n_total": len(planned or []),
    }
    write_atomic(Path(job_root) / PROGRESS_FILE, doc)
    return doc


def mark_spawned(job_root: Path) -> None:
    """Stamp `spawned_at` on the queued doc, BEFORE the process is launched.

    Before, not after: a crash between `spawn()` and this write would leave a
    live runner that occupancy cannot see, which is the flood again. Writing
    first can only ever cost one slot for `spawn_grace_s()`.

    The doc keeps `state: "queued"`; the runner's own `ProgressWriter` is what
    turns it into `running`. Nothing new reaches the UI's `JobState` union.
    """
    path = Path(job_root) / PROGRESS_FILE
    doc = _read(path) or {}
    doc["spawned_at"] = _now()
    doc["updated_at"] = _now()
    doc.pop("_path", None)
    write_atomic(path, doc)


def spawn(job_id: str, job_root: Path, *, rerun: bool = False) -> Dict[str, Any]:
    """Start the runner detached. Returns `{how, pid?}`.

    tmux first (the repo's convention — a human can attach to a stuck
    render), then a plain detached process. Either way the argv is **only the
    job id**: the URL lives in `<job>/request.json` and never reaches a shell.
    """
    args = [str(RUNNER_SH), job_id] + (["--rerun"] if rerun else [])
    window = f"job-{job_id[:40]}"

    if _tmux_available():
        _ensure_session()
        # `tmux new-window` runs under the tmux SERVER's stored environment,
        # not this process's — if `dubsync` already existed when `ui.server`
        # started, `DUBBING_PY` (set in `ui.server.main()` to this process's
        # own interpreter) never reaches the launched script this way, and
        # `scripts/ui_run_job.sh` falls back to its own sibling-env guess
        # instead of the interpreter already proven to work. `-e` (tmux
        # >=3.0) sets it for just this window.
        env_py = os.environ.get("DUBBING_PY") or sys.executable
        cmd = ["tmux", "new-window", "-d", "-t", TMUX_SESSION, "-n", window,
               "-e", f"DUBBING_PY={env_py}", *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            log.info("launched %s in tmux %s:%s", job_id, TMUX_SESSION, window)
            return {"how": "tmux", "session": TMUX_SESSION, "window": window}
        log.warning("tmux launch failed (%s); falling back to a detached "
                    "process: %s", proc.returncode, (proc.stderr or "").strip())

    p = subprocess.Popen(args, start_new_session=True,
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    log.info("launched %s as detached pid %s", job_id, p.pid)
    return {"how": "process", "pid": p.pid}


def _tmux_available() -> bool:
    import shutil
    return shutil.which("tmux") is not None


def _ensure_session() -> None:
    have = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION],
                          capture_output=True, text=True)
    if have.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION],
                       capture_output=True, text=True)


def submit(job_id: str, job_root: Path, jobs_dir: Path, *,
           planned: Optional[List[Dict[str, Any]]] = None,
           rerun: bool = False) -> Dict[str, Any]:
    """Launch now, or queue. Returns `{state, position, launch?}`."""
    job_root = Path(job_root)
    mark_queued(job_root, job_id, planned=planned)
    if not has_capacity(jobs_dir):
        pos = position(jobs_dir, job_id)
        log.info("job %s queued at position %d (%d running, max %d)",
                 job_id, pos, occupancy(jobs_dir), max_concurrent())
        return {"state": "queued", "position": pos}
    mark_spawned(job_root)
    launch = spawn(job_id, job_root, rerun=rerun)
    return {"state": "starting", "position": 0, "launch": launch}


def pump(jobs_dir: Path) -> List[str]:
    """Start queued jobs up to `max_concurrent()`. Returns their ids.

    Called after a job finishes or is cancelled, and safe to call at any time.

    **Re-read the queue every iteration; never iterate a snapshot.** This loop
    used to be `for doc in queued(jobs_dir)` with a `has_capacity()` guard
    inside, which reads as safe and is not: `spawn()` returns long before the
    new process writes `state: running`, so occupancy stayed at 0 for the
    whole pass and every queued job was launched. Measured on 10 queued jobs
    with `max_concurrent=1`: 10 launched by one pump, and 10 more -- duplicate
    runners on the same job folders -- by the next one. `mark_spawned()` plus
    this re-read are the two halves of the fix; neither works alone.
    """
    jobs_dir = Path(jobs_dir)
    started: List[str] = []
    while has_capacity(jobs_dir):
        batch = queued(jobs_dir)
        if not batch:
            break
        doc = batch[0]
        job_id = str(doc.get("job_id"))
        root = jobs_dir / job_id
        if not root.is_dir() or not (root / "request.json").is_file():
            # Seal it rather than `continue` — this is a while loop now, and
            # a skipped job that stays queued is an infinite one.
            log.warning("queued job %s has no request.json; failing it", job_id)
            path = root / PROGRESS_FILE
            bad = _read(path) or {}
            bad.update(state="failed", updated_at=_now(),
                       error="This job has no request.json, so there is "
                             "nothing to run. It was never startable.")
            bad.pop("_path", None)
            if root.is_dir():
                write_atomic(path, bad)
            else:                                        # pragma: no cover
                break
            continue
        mark_spawned(root)
        spawn(job_id, root)
        started.append(job_id)
    return started


# ---------------------------------------------------------------------------
# cancelling
# ---------------------------------------------------------------------------

def cancel(job_root: Path, jobs_dir: Path) -> Dict[str, Any]:
    """Stop a running job, or drop a queued one.

    A running job gets SIGTERM, which `ui/runner.py` turns into an exception
    so the progress doc and the manifest are sealed as `cancelled`. Without
    that handler the folder would read `running` forever and the queue would
    treat the slot as occupied until someone noticed.
    """
    job_root = Path(job_root)
    path = job_root / PROGRESS_FILE
    doc = _read(path) or {}
    state = doc.get("state")
    pid = doc.get("pid")

    if state == "queued":
        doc.update(state="cancelled", updated_at=_now(),
                   error="Cancelled before it started.")
        doc.pop("_path", None)
        write_atomic(path, doc)
        return {"state": "cancelled", "signalled": False}

    if state == "running" and pid_alive(pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError as e:
            log.warning("could not signal pid %s: %s", pid, e)
            return {"state": state, "signalled": False, "error": str(e)}
        # The state is NOT written here. The runner owns it from now on; it
        # seals the doc on the way down, and overwriting it from this side
        # would race with that.
        return {"state": "cancelling", "signalled": True, "pid": int(pid)}

    if state == "running":
        # Stored state says running, the process is gone. Seal it, otherwise
        # it occupies a queue slot that no longer exists.
        doc.update(state="failed", updated_at=_now(),
                   error=doc.get("error") or
                   "The run stopped without reporting a result — the process "
                   "is gone. See run.log for what it was doing.")
        doc.pop("_path", None)
        write_atomic(path, doc)
        return {"state": "failed", "signalled": False}

    return {"state": state or "unknown", "signalled": False}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
