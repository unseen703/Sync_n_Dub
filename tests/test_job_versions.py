"""The job version index maps every edit to the files it produced, and the
live paths the UI reads always belong to `current`.

Version FILES are written by `mt_versions` / `mix_versions` and are tested in
their own modules; these tests cover only the map over them.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend import job_versions as jv  # noqa: E402
from backend import mix_versions, mt_versions  # noqa: E402


def _job(tmp_path: Path) -> Path:
    root = tmp_path / "jobs" / "j1"
    (root / "mt").mkdir(parents=True)
    (root / "mix").mkdir()
    (root / "lipsync").mkdir()
    (root / "mt" / "clip.bodhan.hi.mt.json").write_text(
        json.dumps({"segments": [{"id": "SEG_0000", "mt_text": "a"}]}))
    return root


def _live(root: Path) -> Path:
    return root / "mt" / "clip.bodhan.hi.mt.json"


def _mix(root: Path, tag: str, body: bytes = b"x"):
    wav = root / "mix" / "clip.dubbed.wav"
    wav.write_bytes(body)
    mix_versions.snapshot(wav, None, tag)


# --- numbering -------------------------------------------------------------

def test_a_version_number_is_never_reused(tmp_path):
    root = _job(tmp_path)
    a = jv.append(root, "edit", edit_id="edit_000")
    b = jv.append(root, "edit", edit_id="edit_001")
    assert (a, b) == (0, 1)
    assert jv.read(root)["versions"][1]["parent"] == 0
    assert jv.read(root)["current"] == 1


def test_two_writers_at_once_get_different_numbers(tmp_path):
    """The UI server and the detached runner are separate processes; the
    lock is what keeps them from taking the same number."""
    root = _job(tmp_path)
    jv.read(root)
    got = []
    ts = [threading.Thread(target=lambda: got.append(jv.append(root, "edit")))
          for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(got) == list(range(8))


def test_a_corrupt_index_does_not_break_the_next_write(tmp_path):
    root = _job(tmp_path)
    jv.append(root, "edit")
    jv.index_path(root).write_text("{{{")
    assert isinstance(jv.append(root, "edit"), int)


# --- linking, not copying --------------------------------------------------

def test_an_edit_links_the_snapshots_that_already_exist(tmp_path):
    root = _job(tmp_path)
    mt_versions.snapshot(_live(root), "edit_000")
    _mix(root, "edit_000")
    arts = {**jv.link_transcript(root, "edit_000"), **jv.link_mix(root, "edit_000")}
    v = jv.append(root, "edit", edit_id="edit_000", artifacts=arts)
    row = jv.read(root)["versions"][v]
    assert row["artifacts"]["transcript"]["path"].startswith("mt/versions/")
    assert row["artifacts"]["mix_wav"]["path"].startswith("mix/versions/")
    assert len(row["artifacts"]["mix_wav"]["sha256"]) == 64


def test_linking_adds_no_files(tmp_path):
    root = _job(tmp_path)
    mt_versions.snapshot(_live(root), "edit_000")
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                    if p.is_file() and "versions/index.json" not in p.as_posix())
    jv.link_transcript(root, "edit_000")
    after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and "versions/index.json" not in p.as_posix())
    assert before == after


# --- latest ----------------------------------------------------------------

def test_latest_follows_current_not_the_newest_file(tmp_path):
    root = _job(tmp_path)
    jv.read(root)        # a live job has its index before its first snapshot
    _mix(root, "original", b"one")
    v0 = jv.append(root, "original", edit_id="original",
                   artifacts=jv.link_mix(root, "original"))
    _mix(root, "edit_000", b"two")
    jv.append(root, "edit", edit_id="edit_000",
              artifacts=jv.link_mix(root, "edit_000"))
    assert jv.latest(root, "mix_wav").read_bytes() == b"two"
    assert v0 == 0


def test_after_a_revert_latest_mix_is_still_the_one_on_disk(tmp_path):
    """A revert records a transcript only: the audio carries the edit until a
    re-render, so `latest("mix_wav")` must not point at an older mix."""
    root = _job(tmp_path)
    mt_versions.snapshot(_live(root), "original")
    jv.append(root, "original", edit_id="original",
              artifacts=jv.link_transcript(root, "original"))
    _live(root).write_text(json.dumps({"segments": [{"id": "SEG_0000",
                                                    "mt_text": "edited"}]}))
    mt_versions.snapshot(_live(root), "edit_000")
    _mix(root, "edit_000", b"edited audio")
    jv.append(root, "edit", edit_id="edit_000",
              artifacts={**jv.link_transcript(root, "edit_000"),
                         **jv.link_mix(root, "edit_000")})
    jv.restore(root, 0)
    assert jv.latest(root, "mix_wav").read_bytes() == b"edited audio"


# --- revert ----------------------------------------------------------------

def test_restore_is_a_new_version_and_keeps_the_old_ones(tmp_path):
    root = _job(tmp_path)
    jv.read(root)        # see above: otherwise the snapshot is back-filled
    mt_versions.snapshot(_live(root), "original")
    jv.append(root, "original", edit_id="original",
              artifacts=jv.link_transcript(root, "original"))
    _live(root).write_text(json.dumps({"segments": [{"id": "SEG_0000",
                                                    "mt_text": "edited"}]}))
    mt_versions.snapshot(_live(root), "edit_000")
    jv.append(root, "edit", edit_id="edit_000",
              artifacts=jv.link_transcript(root, "edit_000"))

    new = jv.restore(root, 0)

    doc = jv.read(root)
    assert [r["v"] for r in doc["versions"]] == [0, 1, new]
    assert doc["current"] == new
    assert doc["versions"][-1]["kind"] == "revert"
    assert doc["versions"][-1]["restores"] == 0
    assert json.loads(_live(root).read_text())["segments"][0]["mt_text"] == "a"


def test_restore_never_leaves_a_stray_mt_json(tmp_path):
    """`_find_merged_json` takes the newest `mt/*.mt.json`; a restore must
    leave exactly the live one."""
    root = _job(tmp_path)
    mt_versions.snapshot(_live(root), "original")
    jv.append(root, "original", artifacts=jv.link_transcript(root, "original"))
    jv.restore(root, 0)
    assert [p.name for p in (root / "mt").glob("*.mt.json")] == [_live(root).name]
    assert not list((root / "mt").glob("*.tmp"))


def test_restoring_an_unknown_version_is_refused(tmp_path):
    root = _job(tmp_path)
    jv.read(root)
    with pytest.raises(ValueError):
        jv.restore(root, 9)


# --- render lifecycle ------------------------------------------------------

def test_a_render_is_marked_and_then_linked(tmp_path):
    root = _job(tmp_path)
    jv.append(root, "edit", edit_id="edit_000", status="applied")
    v = jv.begin_render(root, edit_id="edit_000")
    assert jv.read(root)["versions"][v]["status"] == "rendering"
    _mix(root, "edit_000")
    jv.finish_render(root, v, "done")
    row = jv.read(root)["versions"][v]
    assert row["status"] == "rendered" and "mix_wav" in row["artifacts"]


def test_a_failed_render_is_recorded_with_its_error(tmp_path):
    root = _job(tmp_path)
    v = jv.begin_render(root, edit_id="cast", kind="cast")
    jv.finish_render(root, v, "failed", "boom")
    row = jv.read(root)["versions"][v]
    assert row["status"] == "failed" and row["error"] == "boom"


# --- lip-sync --------------------------------------------------------------

def test_stashing_lipsync_keeps_the_old_render_and_repoints_its_version(tmp_path):
    root = _job(tmp_path)
    ls = root / "lipsync" / "clip.musetalk.mp4"
    ls.write_bytes(b"old render")
    v = jv.append(root, "lipsync", edit_id="edit_000",
                  artifacts=jv.link_lipsync(root))
    moves = jv.stash_lipsync(root)

    assert not ls.exists()
    stashed = root / moves[0]["to_rel"]
    assert stashed.read_bytes() == b"old render"
    assert jv.read(root)["versions"][v]["artifacts"]["lipsync_mp4"]["path"] \
        == moves[0]["to_rel"]


def test_a_failed_render_puts_the_old_lipsync_back(tmp_path):
    root = _job(tmp_path)
    ls = root / "lipsync" / "clip.musetalk.mp4"
    ls.write_bytes(b"old render")
    v = jv.append(root, "lipsync", artifacts=jv.link_lipsync(root))
    moves = jv.stash_lipsync(root)
    jv.unstash_lipsync(root, moves)
    assert ls.read_bytes() == b"old render"
    assert jv.read(root)["versions"][v]["artifacts"]["lipsync_mp4"]["path"] \
        == "lipsync/clip.musetalk.mp4"


# --- old jobs --------------------------------------------------------------

def test_an_old_job_is_back_filled_from_what_is_on_disk(tmp_path):
    root = _job(tmp_path)
    mt_versions.snapshot(_live(root), "original")
    mt_versions.snapshot(_live(root), "edit_000")
    _mix(root, "edit_000")
    (root / "edits").mkdir()
    (root / "edits" / "edit_000.json").write_text(json.dumps(
        {"nlq": "speed up", "ops": [{"op": "set_speed"}],
         "result": {"dirty_stages": ["tts", "mix"]}}))

    doc = jv.read(root)

    assert [r["edit_id"] for r in doc["versions"]] == ["original", "edit_000"]
    edit = doc["versions"][1]
    assert edit["nlq"] == "speed up" and edit["status"] == "rendered"
    assert doc["current"] == 1


def test_back_fill_runs_once(tmp_path):
    root = _job(tmp_path)
    mt_versions.snapshot(_live(root), "original")
    first = jv.read(root)
    assert jv.backfill(root) == 0
    assert jv.read(root) == first


# --- wiring ----------------------------------------------------------------

def test_every_writer_records_a_version():
    """A path that changes the dub without a row silently loses its history."""
    for rel, needle in (("backend/edit_agent.py", "job_versions"),
                        ("ui/runner.py", "_jv_begin"),
                        ("ui/runner.py", "_jv_end"),
                        ("ui/routes_edits.py", "record_revert")):
        assert needle in (REPO / rel).read_text(), f"{rel} lacks {needle}"
