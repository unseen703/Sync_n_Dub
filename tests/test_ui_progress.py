"""Can the UI tell what the pipeline is doing, and whose job folder is this?

Two things are tested here, and the first is the important one.

**Anti-drift.** Progress is read by matching the `=== stage N: … ===` lines
`backend/pipeline.py` already logs. That costs no edit to shared code, but it
means a renamed marker silently stops being matched and the UI quietly shows
a run stuck at stage 4 forever. So: grep the real source for every marker
literal and assert this module still matches all of them and has a
user-facing label for each key. This test is the whole reason the approach is
safe.

**Normalisation.** Three runners produce three different shapes under
`data/jobs/` (`progress.json`, `STATUS.json`, bare `manifest.json`), plus 16
folders with no status file at all. If any of that leaks upward the frontend
grows a branch per shape and rots. `read_progress` must flatten all four.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ui import progress as P

REPO = Path(__file__).resolve().parent.parent
PIPELINE_SRC = (REPO / "backend" / "pipeline.py").read_text()

#: Every `=== stage … ===` literal in the pipeline, as written.
MARKER_LITERALS = sorted(set(re.findall(r'"(=== stages? [^"]*)"', PIPELINE_SRC)))


def _probe(literal: str) -> str:
    """A marker literal with its %s/%d placeholders filled, as logged."""
    return re.sub(r"%[sdf]", "X", literal)


# ---------------------------------------------------------------------------
# anti-drift
# ---------------------------------------------------------------------------

def test_pipeline_still_has_stage_markers():
    """If this drops to zero the grep broke, and every other test here is
    vacuously passing."""
    assert len(MARKER_LITERALS) >= 20, MARKER_LITERALS


@pytest.mark.parametrize("literal", MARKER_LITERALS)
def test_every_marker_is_matched(literal):
    assert P.STAGE_MARKER_RE.match(_probe(literal).strip()), (
        f"backend/pipeline.py logs {literal!r}, which ui/progress.py no longer "
        f"recognises. Progress will silently stall at the previous stage.")


@pytest.mark.parametrize("literal", MARKER_LITERALS)
def test_every_marker_key_has_a_user_facing_label(literal):
    key = P.STAGE_MARKER_RE.match(_probe(literal).strip()).group("key")
    assert key in P.STAGE_LABELS, (
        f"stage {key!r} has no entry in STAGE_LABELS, so the UI would show "
        f"the runner's own wording for it.")


def test_both_marker_shapes_parse():
    """`stage 9f1b: …` has a colon; `stages 1-9a SKIPPED …` does not."""
    m = P.STAGE_MARKER_RE.match("=== stage 9f1b: viseme extraction (mms_char) ===")
    assert m.group("key") == "9f1b"
    m = P.STAGE_MARKER_RE.match("=== stages 1-9a SKIPPED (post-MT replay) ===")
    assert m.group("key") == "1-9a"
    assert m.group("title").startswith("SKIPPED")


def test_non_marker_lines_are_ignored():
    for line in ("tts: synthesising 12 segments", "=== not a stage ===", ""):
        assert P.STAGE_MARKER_RE.match(line) is None


def test_every_stage_key_is_grouped_or_deliberately_hidden():
    """A key that is neither grouped nor hidden falls into a nameless
    "Other" segment on the bar — a stage the user never asked about,
    appearing with a title-cased version of its internal name."""
    stray = [k for k in P.STAGE_LABELS
             if k not in P._GROUP_OF and k not in P.HIDDEN_KEYS]
    assert not stray, (
        f"no ribbon group for {stray} — add them to PHASE_GROUPS, or to "
        f"HIDDEN_KEYS if they produce no part of the dub")


def test_the_bar_shows_the_seven_named_stages():
    """The pipeline as a person describes it. Seven, in order, no tree.

    Pinned because the grouping is the one place the UI editorialises about
    the pipeline, and a stage quietly appearing or being renamed changes
    what the whole product looks like it does.
    """
    assert [label for _id, label, _keys in P.PHASE_GROUPS] == [
        'Download video', 'Speaker annotation', 'Transcribe', 'Translate',
        'Speech generation', 'Mix', 'Lip sync',
    ]


def test_evaluation_stages_stay_off_the_bar():
    """10a/10b score ASR and MT against subtitles. They are an optional QA
    pass that produces no part of the dub, so they must not appear as an
    eighth stage."""
    plan = P.plan_stages({'asr': 'whisperx', 'mt': 'bodhan'})
    plan += [{'key': k, 'title': None, 'label': P.label_for(k),
              'state': 'done', 'started_at': None, 'wall_s': 1.0}
             for k in P.HIDDEN_KEYS]
    labels = [ph['label'] for ph in P._group_phases(plan)]
    assert 'Checks' not in labels
    assert len(labels) <= 7


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def _writer(tmp_path, planned=None):
    return P.ProgressWriter(tmp_path, job_id="job", planned=planned,
                            heartbeat_s=3600)


def _log(w, msg):
    import logging
    w.emit(logging.LogRecord("backend.pipeline", logging.INFO, __file__, 1,
                             msg, None, None))


def test_writer_tracks_the_running_stage(tmp_path):
    w = _writer(tmp_path, P.plan_stages({"asr": "bodhan", "mt": "bodhan"}))
    _log(w, "=== stage 1: download / resolve ===")
    _log(w, "=== stage 2: extract audio ===")
    doc = json.loads((tmp_path / "progress.json").read_text())
    states = {s["key"]: s["state"] for s in doc["stages"]}
    assert states["1"] == "done" and states["2"] == "running"
    assert doc["current"] == "2"
    w.finish("done")
    doc = json.loads((tmp_path / "progress.json").read_text())
    assert doc["state"] == "done"
    assert all(s["state"] in ("done", "skipped") for s in doc["stages"])


def test_writer_marks_a_skipped_stage(tmp_path):
    w = _writer(tmp_path, P.plan_stages({}))
    _log(w, "=== stage 3: SKIPPED ===")
    doc = json.loads((tmp_path / "progress.json").read_text())
    assert [s for s in doc["stages"] if s["key"] == "3"][0]["state"] == "skipped"
    w.finish()


def test_stages_jumped_over_are_skipped_not_left_pending(tmp_path):
    """A conditional stage that never ran must not read "pending" forever."""
    w = _writer(tmp_path, P.plan_stages(
        {"asr": "bodhan", "align": "mms_fa", "mt": "bodhan"}))
    _log(w, "=== stage 7: ASR (bodhan, full_audio=False) ===")
    _log(w, "=== stage 9: machine translation (bodhan -> hi, script=codemix) ===")
    doc = json.loads((tmp_path / "progress.json").read_text())
    states = {s["key"]: s["state"] for s in doc["stages"]}
    assert states["8"] == "skipped", "alignment never ran but was planned"
    assert states["9"] == "running"
    w.finish()


def test_unplanned_stage_is_appended_rather_than_dropped(tmp_path):
    """Reality beats the forecast when plan_stages drifts."""
    w = _writer(tmp_path, P.plan_stages({}))
    _log(w, "=== stage 9h: lip sync (musetalk) ===")
    doc = json.loads((tmp_path / "progress.json").read_text())
    assert any(s["key"] == "9h" and s["state"] == "running"
               for s in doc["stages"])
    w.finish()


def test_writer_records_failure_with_the_message(tmp_path):
    w = _writer(tmp_path, P.plan_stages({}))
    _log(w, "=== stage 1: download / resolve ===")
    w.finish("failed", error="RuntimeError: no such video")
    doc = json.loads((tmp_path / "progress.json").read_text())
    assert doc["state"] == "failed"
    assert "no such video" in doc["error"]


def test_write_is_atomic(tmp_path):
    """A reader polling every 1.5 s must never see a half-written file."""
    P.write_atomic(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# the reader / normaliser
# ---------------------------------------------------------------------------

def test_reads_status_json_shape(tmp_path):
    (tmp_path / "STATUS.json").write_text(json.dumps({
        "state": "running P3_ASD", "updated": "2026-10-15T02:30:07",
        "phases": [{"name": "P1_CUT", "wall_s": 6.7, "done": True},
                   {"name": "P2_FACES", "wall_s": 40.3, "done": True},
                   {"name": "P3_ASD", "wall_s": 0, "done": False}]}))
    doc = P.read_progress(tmp_path)
    assert doc["source"] == "STATUS.json"
    assert doc["state"] == "running"
    assert doc["current"] == "P3_ASD"
    # The denominator is what has been seen — this runner publishes no plan,
    # and inventing a total would be a lie with a progress bar on it.
    assert doc["n_total"] == 3 and doc["n_total_is_partial"] is True
    # Clip phases join the same ribbon rather than a nameless "Other" pill.
    assert "other" not in {ph["id"] for ph in doc["phases"]}


def test_status_json_failure_carries_the_reason(tmp_path):
    (tmp_path / "STATUS.json").write_text(json.dumps({
        "state": "failed: CUDA out of memory", "phases": []}))
    doc = P.read_progress(tmp_path)
    assert doc["state"] == "failed"
    assert "CUDA out of memory" in doc["error"]


def test_reads_manifest_only_shape(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 2, "status": "ok", "started_at": "2026-10-16T16:55:44Z",
        "config": {"asr": "bodhan", "mt": "bodhan", "tts": "edge"}}))
    doc = P.read_progress(tmp_path)
    assert doc["source"] == "manifest.json"
    assert doc["state"] == "done" and doc["pct"] == 100
    assert doc["coarse"] is True


def test_reads_a_folder_with_no_status_file(tmp_path):
    """16 of the 51 folders on disk are flat script output. They are real
    results and must still appear."""
    (tmp_path / "final.hi.wav2lip.mp4").write_bytes(b"\0" * 32)
    doc = P.read_progress(tmp_path)
    assert doc is not None and doc["source"] == "files"
    assert doc["state"] == "done"


def test_empty_folder_reads_as_nothing(tmp_path):
    assert P.read_progress(tmp_path) is None


def test_dead_process_is_not_reported_as_running(tmp_path):
    """A `kill -9` leaves state:"running" on disk forever otherwise."""
    P.write_atomic(tmp_path / "progress.json", {
        "version": 1, "job_id": "j", "state": "running", "pid": 2 ** 31 - 1,
        "stages": [], "n_done": 0, "n_total": 1})
    doc = P.read_progress(tmp_path)
    assert doc["state"] == "failed"
    assert "stopped" in (doc["error"] or "")


def test_live_process_stays_running(tmp_path):
    import os
    P.write_atomic(tmp_path / "progress.json", {
        "version": 1, "job_id": "j", "state": "running", "pid": os.getpid(),
        "stages": [], "n_done": 0, "n_total": 1})
    assert P.read_progress(tmp_path)["state"] == "running"


# ---------------------------------------------------------------------------
# milestones — what the UI is allowed to reveal
# ---------------------------------------------------------------------------

def test_diarization_milestone_from_the_rttm_not_only_the_marker(tmp_path):
    """A stage is only MARKED done when the next marker fires. For a
    diarization-only run that would be process exit, so the video and speaker
    table would appear minutes late. The RTTM closes that."""
    (tmp_path / "diarization").mkdir()
    (tmp_path / "diarization" / "x.pyannote.rttm").write_text("SPEAKER x 1 0 1 <NA> <NA> S0 <NA> <NA>\n")
    assert P.milestones(tmp_path, {"stages": []})["diarization_done"] is True


@pytest.mark.parametrize("name", [
    "final.hi.musetalk.mp4",          # clip runner / flat scripts
    "x.wav2lip.lipsync.mp4",          # pipeline, at the root
    "x.wav2lip.codeformer.mp4",       # restored
])
def test_lipsync_milestone_recognises_every_naming(tmp_path, name):
    (tmp_path / name).write_bytes(b"\0" * 16)
    assert P.milestones(tmp_path, {"stages": []})["lipsync_ready"] is True


def test_lipsync_milestone_is_false_before_any_render(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "clip.mp4").write_bytes(b"\0" * 16)
    ms = P.milestones(tmp_path, {"stages": []})
    assert ms["video_ready"] is True
    assert ms["lipsync_ready"] is False


# ---------------------------------------------------------------------------
# against the real job folders
# ---------------------------------------------------------------------------

JOBS = REPO / "data" / "jobs"


@pytest.mark.skipif(not JOBS.is_dir(), reason="no data/jobs on this host")
def test_every_real_job_folder_normalises():
    folders = [d for d in JOBS.iterdir() if d.is_dir()]
    if not folders:
        pytest.skip("data/jobs is empty")
    unresolved = [d.name for d in folders if P.read_progress(d) is None]
    assert not unresolved, (
        f"{len(unresolved)} job folder(s) report no progress at all, so they "
        f"would be invisible in the UI: {unresolved[:5]}")


@pytest.mark.skipif(not JOBS.is_dir(), reason="no data/jobs on this host")
def test_a_finished_job_reports_a_finished_render():
    folders = [d for d in JOBS.iterdir() if d.is_dir()]
    done = [(d, p) for d in folders
            if (p := P.read_progress(d)) and p["state"] == "done"]
    if not done:
        pytest.skip("no finished jobs on disk")
    bad = [d.name for d, p in done if not p["milestones"]["lipsync_ready"]]
    assert not bad, f"finished jobs whose render was not found: {bad[:5]}"


# ---------------------------------------------------------------------------
# the bar must show every stage that will actually run
# ---------------------------------------------------------------------------

def test_lipsync_is_planned_when_the_model_is_still_auto():
    """`auto` means "not chosen yet", not "not running".

    The default preset sends `lipsync_model: "auto"` and the real model is
    picked by the face probe a few seconds into the run, so the pipeline
    kwargs still read None when the stage list is built. Keying the plan off
    those kwargs dropped stage 9h, and every default-preset job rendered a
    six-node bar that grew a seventh node partway through. Three real jobs on
    disk were written that way before this was caught.
    """
    labels = [ph["label"] for ph in P._group_phases(
        P.plan_stages({"asr": "whisperx", "mt": "bodhan", "tts": "edge",
                       "mix": True, "lipsync": "auto"}))]
    assert "Lip sync" in labels


def test_no_lipsync_node_when_none_was_requested():
    """The converse: an audio-only dub must not show a stage it will skip."""
    labels = [ph["label"] for ph in P._group_phases(
        P.plan_stages({"asr": "whisperx", "mt": "bodhan", "tts": "edge",
                       "mix": True, "lipsync": None}))]
    assert "Lip sync" not in labels


def test_the_runner_plans_the_requested_lipsync():
    """Pins the call site, not just `plan_stages` — the bug lived there."""
    import inspect

    from ui import runner
    src = inspect.getsource(runner)
    assert 'plan_cfg["lipsync"] = requested_ls' in src, (
        "ui/runner.py no longer carries the requested lip-sync model into the "
        "stage plan; an `auto` job will render a bar with no Lip sync node")


@pytest.mark.parametrize("name", sorted(P.CLIP_PHASE_LABELS) + ["P5_SEGMENT"])
def test_every_clip_runner_phase_is_mapped(name):
    """An unmapped phase name renders as a nameless eighth node labelled
    "Other". `P5_SEGMENT` did exactly that in 11 jobs on disk."""
    assert name in P._GROUP_OF, f"{name} has no ribbon group"


def test_legacy_job_stages_come_from_evidence_not_assumption(tmp_path):
    """A folder with only a finished render proves lip-sync ran. It proves
    nothing about transcription — several of these were driven by a pre-made
    dub with no ASR at all — so the bar stays short rather than inventing
    seven nodes of history."""
    (tmp_path / "final.hi.wav2lip.mp4").write_bytes(b"\0" * 16)
    (tmp_path / "mix").mkdir()
    (tmp_path / "mix" / "clip.dubbed.wav").write_bytes(b"\0" * 16)
    doc = P.read_progress(tmp_path)
    labels = [ph["label"] for ph in doc["phases"]]
    assert "Lip sync" in labels and "Mix" in labels
    assert "Transcribe" not in labels, "claimed a stage nothing vouches for"
    assert doc["n_total_is_partial"] is True
