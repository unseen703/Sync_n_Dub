"""The two ways a UI job refuses to stay inside its lane.

Both bugs here shipped, both ran green, and both were found by a human
watching the machine rather than by anything in this suite.

* **One pump launched the whole backlog.** `spawn()` returns in microseconds;
  the job only reads `running` on disk once `ProgressWriter.__init__` runs
  inside the *new* process, after `ui_run_job.sh` has started python and
  imported torch. The old loop iterated a snapshot of `queued()` and re-checked
  `has_capacity()` inside it -- which reads as careful and is not, because the
  thing it re-read could not have changed yet. Ten queued jobs became ten
  simultaneous runners at `max_concurrent=1`, and because nothing marked a
  spawned job as no-longer-queued, the next pump launched all ten *again* onto
  the same job folders. On two 16 GiB cards, with MuseTalk peaking at 15.3 GiB
  per shard, that is the whole machine.

* **Stop killed a segment instead of the run.** `Cancelled` derived from
  `Exception`, so every `except Exception` written to survive a flaky upstream
  ate it. `models/tts/bodhan_adapter.py` catches one per segment and moves on,
  so a Stop click during TTS was logged 26 times as "synth failed for
  SEG_xxxx: Stopped on signal 15" while the run carried on to mix and lip-sync
  a dub with 26 silent gaps in it.

Nothing here launches a real process: `spawn` is monkeypatched to a recorder.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui import jobqueue                                        # noqa: E402
from ui.progress import PROGRESS_FILE                          # noqa: E402


@pytest.fixture
def jobs_dir(tmp_path):
    d = tmp_path / "jobs"
    d.mkdir()
    return d


@pytest.fixture
def spawns(monkeypatch):
    """Record launches instead of making them."""
    calls = []

    def _fake(job_id, job_root, **kw):
        calls.append(job_id)
        return {"how": "fake"}

    monkeypatch.setattr(jobqueue, "spawn", _fake)
    return calls


def enqueue(jobs_dir: Path, n: int) -> None:
    for i in range(n):
        root = jobs_dir / f"job{i:02d}"
        root.mkdir()
        (root / "request.json").write_text("{}")
        jobqueue.mark_queued(root, root.name)


def state_of(jobs_dir: Path, job_id: str) -> dict:
    return json.loads((jobs_dir / job_id / PROGRESS_FILE).read_text())


# ---------------------------------------------------------------------------
# the flood
# ---------------------------------------------------------------------------

def test_one_pump_launches_at_most_max_concurrent(jobs_dir, spawns, monkeypatch):
    """The regression itself: 10 queued, max_concurrent=1, 1 launched."""
    monkeypatch.delenv(jobqueue.MAX_CONCURRENT_ENV, raising=False)
    enqueue(jobs_dir, 10)
    assert jobqueue.max_concurrent() == 1
    assert jobqueue.occupancy(jobs_dir) == 0

    jobqueue.pump(jobs_dir)

    assert len(spawns) == 1, f"pump launched {len(spawns)} jobs onto one slot"
    assert len(jobqueue.queued(jobs_dir)) == 9


def test_a_second_pump_does_not_duplicate_the_first(jobs_dir, spawns):
    """Two runners on one job folder write the same progress.json."""
    enqueue(jobs_dir, 5)
    jobqueue.pump(jobs_dir)
    first = list(spawns)
    spawns.clear()

    jobqueue.pump(jobs_dir)                 # a finishing runner, or a cancel

    assert spawns == [], f"re-launched {spawns} on top of {first}"


def test_a_launched_job_occupies_its_slot_before_it_registers(jobs_dir, spawns):
    """The window the whole fix exists for: spawned, not yet `running`."""
    enqueue(jobs_dir, 2)
    jobqueue.pump(jobs_dir)

    doc = state_of(jobs_dir, "job00")
    assert doc["state"] == "queued", "the runner owns the running state"
    assert doc.get("spawned_at"), "nothing marks the job as launched"
    assert jobqueue.occupancy(jobs_dir) == 1
    assert not jobqueue.has_capacity(jobs_dir)


def test_max_concurrent_two_launches_two(jobs_dir, spawns, monkeypatch):
    """The cap is honoured, not hard-coded to one."""
    monkeypatch.setenv(jobqueue.MAX_CONCURRENT_ENV, "2")
    enqueue(jobs_dir, 6)
    jobqueue.pump(jobs_dir)
    assert len(spawns) == 2


def test_a_runner_that_never_boots_releases_its_slot(jobs_dir, spawns, monkeypatch):
    """The opposite failure: a wedged queue.

    If `spawned_at` held the slot forever, one bad interpreter would stop the
    queue permanently -- the same shape as the `kill -9` case this module's
    header warns about, just from the other side.
    """
    monkeypatch.setenv(jobqueue.SPAWN_GRACE_ENV, "1")
    enqueue(jobs_dir, 2)
    jobqueue.pump(jobs_dir)
    assert len(spawns) == 1

    stale = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=30)
    path = jobs_dir / "job00" / PROGRESS_FILE
    doc = json.loads(path.read_text())
    doc["spawned_at"] = stale.isoformat(timespec="seconds")
    path.write_text(json.dumps(doc))

    assert jobqueue.has_capacity(jobs_dir), "a dead launch wedged the queue"
    jobqueue.pump(jobs_dir)
    assert len(spawns) == 2


def test_a_job_without_a_request_is_failed_not_skipped(jobs_dir, spawns):
    """`pump` is a while loop now; a skipped-but-still-queued job never ends."""
    enqueue(jobs_dir, 1)
    (jobs_dir / "job00" / "request.json").unlink()
    broken = jobs_dir / "jobXX"
    broken.mkdir()
    (broken / "request.json").write_text("{}")
    jobqueue.mark_queued(broken, broken.name)

    jobqueue.pump(jobs_dir)                 # must terminate

    assert state_of(jobs_dir, "job00")["state"] == "failed"
    assert spawns == ["jobXX"], "the startable job behind it never started"


def test_submit_marks_before_it_spawns(jobs_dir, spawns):
    """A crash between the two would leave a runner occupancy cannot see."""
    root = jobs_dir / "job00"
    root.mkdir()
    (root / "request.json").write_text("{}")

    res = jobqueue.submit("job00", root, jobs_dir)

    assert res["state"] == "starting"
    assert state_of(jobs_dir, "job00").get("spawned_at")
    assert jobqueue.occupancy(jobs_dir) == 1

    second = jobs_dir / "job01"
    second.mkdir()
    (second / "request.json").write_text("{}")
    assert jobqueue.submit("job01", second, jobs_dir)["state"] == "queued"


# ---------------------------------------------------------------------------
# the Stop button
# ---------------------------------------------------------------------------

def test_cancelled_is_not_caught_by_except_exception():
    """The whole bug, in four lines.

    `backend/` and `models/` are full of `except Exception` blocks that exist
    to survive a flaky upstream. Cancellation must pass straight through them.
    """
    from ui.runner import Cancelled

    assert issubclass(Cancelled, BaseException)
    assert not issubclass(Cancelled, Exception)

    swallowed = False
    try:
        try:
            raise Cancelled("Stopped on signal 15.")
        except Exception:                    # noqa: BLE001 - the point
            swallowed = True
    except Cancelled:
        pass
    assert not swallowed, "a broad handler ate the cancellation"


def test_the_bodhan_retry_loop_cannot_swallow_a_cancel():
    """The exact shape that lost 26 segments, run against the real class."""
    from ui.runner import Cancelled

    done, errors = [], []
    with pytest.raises(Cancelled):
        for seg in ("SEG_0078", "SEG_0079", "SEG_0080"):
            try:
                if seg == "SEG_0079":
                    raise Cancelled("Stopped on signal 15.")
                done.append(seg)
            except Exception as e:           # noqa: BLE001 - bodhan_adapter.py
                errors.append(f"synth failed for {seg}: {e}")

    assert done == ["SEG_0078"]
    assert errors == [], f"cancellation was logged as a synth failure: {errors}"


def test_the_runner_still_seals_the_progress_doc_on_cancel():
    """`except Cancelled` must stay ABOVE `except BaseException` in runner.py.

    Flipping to BaseException is only safe because both handlers already
    catch it; if either ordering is reversed, a cancelled run gets sealed as
    `failed` and the UI stops saying who stopped it.
    """
    src = (REPO_ROOT / "ui" / "runner.py").read_text()
    assert "class Cancelled(BaseException):" in src
    for block in ("except Cancelled as e:", "except BaseException as e:"):
        assert src.count(block) == 2, f"{block} moved; re-check the ordering"
    assert src.index("except Cancelled as e:") < src.index("except BaseException as e:")


def test_cancelling_also_stops_the_subprocesses():
    """Lip-sync is one process per GPU; the parent raising frees neither."""
    import inspect

    from ui import runner

    assert hasattr(runner, "_terminate_children")
    handler_src = inspect.getsource(runner._install_sigterm)
    assert "_terminate_children()" in handler_src, \
        "the signal handler no longer reaps the shelled-out renderers"
