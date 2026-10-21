"""The job rail is ordered by when a job last RAN, newest first.

`_list_jobs` used to return `sorted(JOBS_DIR.iterdir(), reverse=True)` — the
folder NAME, descending. That is correct only for a job whose name begins
with an allocation timestamp, and it was wrong twice:

1. **A hand-named job sorts by ASCII**, where lowercase letters sort above
   digits. `w1l2_60to100_hi_musetalk_cli` and `bench_hi_120s` therefore
   pinned themselves above every timestamped job on the machine, however
   old they were — and this repo has several of both.
2. **The name records ALLOCATION, not execution.** A re-run (the edit
   agent's replay, or the new lip-sync step) writes fresh progress into a
   folder whose name never changes, so a job that ran minutes ago sorted as
   though it had run whenever it was created.

`JobRail.tsx` deliberately does not re-sort, so this is the only place the
order can be fixed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ui.server as S                                          # noqa: E402


def _job(jobs_dir: Path, name: str, *, started=None, finished=None,
         status_updated=None) -> Path:
    d = jobs_dir / name
    d.mkdir(parents=True)
    manifest = {"job_id": name}
    if started:
        manifest["started_at"] = started
    if finished:
        manifest["finished_at"] = finished
    (d / "manifest.json").write_text(json.dumps(manifest))
    if status_updated:
        (d / "STATUS.json").write_text(
            json.dumps({"state": "done", "updated": status_updated}))
    return d


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(S, "JOBS_DIR", jobs_dir)
    return jobs_dir


def _order(rows):
    return [r["job_id"] for r in rows]


# ---------------------------------------------------------------------------
# the two bugs
# ---------------------------------------------------------------------------

def test_a_hand_named_job_does_not_pin_itself_to_the_top(jobs):
    """`w…` > `2…` in ASCII, so name-descending put every CLI job first."""
    _job(jobs, "20261020-120000__abc__cfg",
         finished="2026-10-20T12:30:00+00:00")
    _job(jobs, "w1l2_60to100_hi_musetalk_cli",
         finished="2026-10-01T09:00:00+00:00")

    assert _order(S._list_jobs())[0].startswith("20261020"), (
        "the old CLI job is still sorted above a job that ran 19 days later")


def test_a_rerun_moves_a_job_back_to_the_top(jobs):
    """The folder name still says it was allocated first; the stamps say it
    ran last. The stamps win — that is what "recently executed" means."""
    _job(jobs, "20261001-090000__old__cfg",
         started="2026-10-01T09:00:00+00:00",
         status_updated="2026-10-20T18:00:00+00:00")      # re-run today
    _job(jobs, "20261015-090000__newer__cfg",
         finished="2026-10-15T09:30:00+00:00")

    assert _order(S._list_jobs())[0].startswith("20261001"), (
        "a job re-run today is not at the top")


def test_plain_recency_still_holds(jobs):
    _job(jobs, "20261018-000000__a__cfg", finished="2026-10-18T00:10:00+00:00")
    _job(jobs, "20261020-000000__b__cfg", finished="2026-10-20T00:10:00+00:00")
    _job(jobs, "20261019-000000__c__cfg", finished="2026-10-19T00:10:00+00:00")

    assert _order(S._list_jobs()) == ["20261020-000000__b__cfg",
                                      "20261019-000000__c__cfg",
                                      "20261018-000000__a__cfg"]


# ---------------------------------------------------------------------------
# the fallbacks
# ---------------------------------------------------------------------------

def test_a_job_with_no_stamps_at_all_still_sorts(jobs):
    """CLAUDE.md §2 counts four job-folder shapes plus 16 folders with no
    status file. None of them may sort as epoch 0 and sink out of sight."""
    d = jobs / "bench_hi_120s"
    d.mkdir()
    rows = S._list_jobs()

    assert len(rows) == 1
    assert rows[0]["last_activity"] > 0, "fell back to nothing"


def test_the_id_timestamp_is_used_when_there_is_no_status(jobs):
    d = jobs / "20261020-043027__dIYDPtHWBNA__cfg"
    d.mkdir()
    rows = S._list_jobs()

    assert rows[0]["last_activity_iso"].startswith("2026-10-20")


def test_order_is_stable_when_two_jobs_share_a_timestamp(jobs):
    """A batch submission gives two folders the same stamp. Without a
    tie-break the rail would reshuffle between polls and jump under the
    cursor."""
    for name in ("20261020-090000__a__cfg", "20261020-090000__b__cfg"):
        _job(jobs, name, finished="2026-10-20T09:00:00+00:00")

    assert _order(S._list_jobs()) == _order(S._list_jobs())


# ---------------------------------------------------------------------------
# the timestamp reader
# ---------------------------------------------------------------------------

def test_a_naive_timestamp_is_read_as_utc_not_local():
    """Every writer here stamps with `datetime.now(timezone.utc)`. Reading a
    naive one as local time would shift it by the machine's offset — five
    and a half hours here — and interleave it wrongly with the aware ones.
    """
    aware = S._epoch("2026-10-20T09:00:00+00:00")
    naive = S._epoch("2026-10-20 09:00:00")
    assert aware == naive


def test_unreadable_stamps_do_not_raise():
    for bad in (None, "", "not a date", {}, []):
        assert S._epoch(bad) is None


def test_the_newest_stamp_wins_not_the_first_one_found(tmp_path):
    """`max()` over the candidates, not a first-match chain: `started_at`
    exists on every job and would otherwise mask a later `finished_at`."""
    d = tmp_path / "j"
    d.mkdir()
    ts = S._last_activity(
        d, {"timestamp": "2026-10-01 00:00:00"}, None,
        {"started_at": "2026-10-01T00:00:00+00:00",
         "finished_at": "2026-10-20T00:00:00+00:00"}, None)

    assert S._epoch("2026-10-20T00:00:00+00:00") == ts
