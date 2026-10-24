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


def test_the_bar_shows_the_eight_named_stages():
    """The pipeline as a person describes it. Eight, in order, no tree.

    Pinned because the grouping is the one place the UI editorialises about
    the pipeline, and a stage quietly appearing or being renamed changes
    what the whole product looks like it does.

    `Enhance resolution` was added 2026-10-23 by request. It is its own pill
    rather than a substage of `Lip sync` because it runs on the finished
    render from any model and costs 2.6x the MuseTalk pass it cleans
    (1156 s vs 447 s, measured 2026-10-22) -- folding it in would make a lit
    `Lip sync` mean two different amounts of work.
    """
    assert [label for _id, label, _keys in P.PHASE_GROUPS] == [
        'Download video', 'Speaker annotation', 'Transcribe', 'Translate',
        'Speech generation', 'Mix', 'Lip sync', 'Enhance resolution',
    ]


def test_the_bar_is_always_nine_pills_whatever_the_config():
    """Eight groups plus `Finished`, identical on every run.

    `_group_phases` used to drop a group with no planned substages, and
    `plan_stages` is config-driven — so a run enabling few optional stages
    rendered a bar with one or two pills on it, and the bar's length changed
    between two runs of the same video. An operator cannot tell "this
    pipeline has no transcribe step" from "the bar is broken".
    """
    expected = ['Download video', 'Speaker annotation', 'Transcribe',
                'Translate', 'Speech generation', 'Mix', 'Lip sync',
                'Enhance resolution', 'Finished']
    for cfg in ({'asr': 'whisperx'},
                {'asr': 'whisperx', 'mt': 'bodhan'},
                {'asr': 'bodhan', 'mt': 'bodhan', 'tts': 'bodhan',
                 'mix': True, 'lipsync': 'musetalk'},
                {}):
        labels = [ph['label'] for ph in P._group_phases(P.plan_stages(cfg))]
        assert labels == expected, f"config {cfg} drew {labels}"


def test_an_unplanned_group_reads_skipped_not_pending():
    """The pipeline HAS the step; this run is not taking it. `pending` would
    read as "still to come" and leave a finished run looking unfinished."""
    phases = {p['id']: p for p in
              P._group_phases(P.plan_stages({'asr': 'whisperx'}))}
    assert phases['translate']['state'] == 'skipped'
    assert phases['lipsync']['state'] == 'skipped'


def test_finished_waits_for_a_deferred_lipsync():
    """The load-bearing case. A draft run leaves 9h pending-and-deferred
    because lip-sync is a separate explicit act; lighting `Finished` over it
    would be the screen asserting something false."""
    stages = P.plan_stages({"asr": "bodhan", "mt": "bodhan", "tts": "bodhan",
                            "mix": True, "lipsync": None})
    for s in stages:                      # everything the run DID do
        if s["key"] != "9h":
            s["state"] = "done"
    phases = {p['id']: p for p in P._group_phases(stages)}
    assert phases['lipsync']['state'] == 'pending'
    assert phases[P.FINISHED_ID]['state'] == 'pending'


def test_finished_lights_when_every_stage_is_done_or_skipped():
    stages = P.plan_stages({"asr": "bodhan", "mt": "bodhan", "tts": "bodhan",
                            "mix": True, "lipsync": "musetalk"})
    for s in stages:
        s["state"] = "done"
    phases = {p['id']: p for p in P._group_phases(stages)}
    assert phases[P.FINISHED_ID]['state'] == 'done'


def test_finished_carries_no_wall_time():
    """The ribbon sums `wall_s` across phases for elapsed time, so a total
    on this pill would report double the real runtime."""
    stages = P.plan_stages({"asr": "bodhan", "mt": "bodhan"})
    for s in stages:
        s["state"], s["wall_s"] = "done", 5.0
    fin = [p for p in P._group_phases(stages) if p["id"] == P.FINISHED_ID][0]
    assert fin["wall_s"] is None


def test_a_failure_anywhere_fails_the_finished_pill():
    stages = P.plan_stages({"asr": "bodhan", "mt": "bodhan"})
    stages[0]["state"] = "failed"
    fin = [p for p in P._group_phases(stages) if p["id"] == P.FINISHED_ID][0]
    assert fin["state"] == "failed"


def test_finished_is_not_a_pipeline_marker():
    """It is synthetic. Putting it in `PHASE_GROUPS` would make the
    marker-grep test below fail it for having no `=== stage ===` line."""
    assert P.FINISHED_ID not in {gid for gid, _l, _k in P.PHASE_GROUPS}


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
    # Eight groups + `Finished`, and nothing else. The QA pass must not
    # become a tenth pill.
    assert len(labels) == 9


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
    # Every stage the run was responsible for is resolved. A `deferred` one is
    # not: this config asks for no lip-sync, so 9h is planted pending and left
    # that way for the separate lip-sync submission to open.
    assert all(s["state"] in ("done", "skipped")
               for s in doc["stages"] if not s.get("deferred"))


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
    # Only jobs that actually RAN lip-sync can have a render. A dub-only job
    # -- `--mix` with no `--lipsync`, which is how the scaling benchmarks
    # built one dub for six separate lip-sync passes to share -- has an empty
    # `lipsync/` by design, and asserting a render for it fails the suite for
    # doing exactly what it was asked to do. Keyed off the stage list rather
    # than the manifest config, because `read_progress` normalises four
    # different on-disk job shapes and the stage list is the one thing all
    # four agree on.
    #
    # The test is the STATE of 9h, not its PRESENCE. Presence used to mean
    # "lip-sync was configured", and stopped meaning it on 2026-10-21: 9h is
    # now planned on every job so a draft dub can show the stage it is
    # waiting for. A row still `pending` is that case and has no render to
    # report; `done` is the one that claims it finished, which is precisely
    # the claim this guards.
    ran_lipsync = [(d, p) for d, p in done
                   if any(s.get("key") == "9h" and s.get("state") == "done"
                          for s in p.get("stages", []))]
    if not ran_lipsync:
        pytest.skip("no finished jobs on disk that ran lip-sync")

    # A REFUSAL is not a missing render. `run_lipsync` declines rather than
    # sharing a card -- "no GPU is free, gpu0 busy (pid …)" -- and records
    # that in `result.lipsync_error` while the audio dub, which did finish,
    # keeps the job at done. 32 jobs on this host are in exactly that state
    # from one batch that outran the GPUs, and every one of them recorded
    # its reason.
    #
    # Which is the point the assertion has to make: the bug this guards
    # against is lip-sync producing NOTHING while claiming success, so the
    # failing case is a missing render with NO recorded error. Flagging the
    # ones that said why would fail the suite for the honest-failure
    # machinery working (CLAUDE.md §4).
    def _recorded_error(job_dir):
        try:
            manifest = json.loads((job_dir / "manifest.json").read_text())
        except (OSError, ValueError):
            return None
        return (manifest.get("result") or {}).get("lipsync_error")

    bad = [d.name for d, p in ran_lipsync
           if not p["milestones"]["lipsync_ready"] and not _recorded_error(d)]
    assert not bad, (
        f"finished jobs with no render and no recorded reason: {bad[:5]}")


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


def test_lipsync_node_is_shown_even_when_the_run_will_not_do_it():
    """A draft dub still draws the Lip sync node — pending, not absent.

    This asserts the OPPOSITE of what it used to, deliberately. A `draft`
    submission sets `lipsync: None` and stops at the mix, with lip-sync a
    separate explicit act afterwards. Dropping the node made that job render a
    six-node bar reading "Finished · all 6 stages" — identical to a dub that
    will never be lip-synced, for one that is waiting to be.
    """
    stages = P.plan_stages({"asr": "whisperx", "mt": "bodhan", "tts": "bodhan",
                            "mix": True, "lipsync": None})
    labels = [ph["label"] for ph in P._group_phases(stages)]
    assert "Lip sync" in labels
    ls = next(s for s in stages if s["key"] == "9h")
    assert ls["state"] == "pending"
    # The flag is what carries "not yet" through the two places that would
    # otherwise resolve a leftover pending stage to `skipped`.
    assert ls["deferred"] is True


def test_a_deferred_lipsync_stage_survives_a_successful_finish(tmp_path):
    """`finish("done")` must not call the un-run stage `skipped`.

    `StageRibbon.isDone` counts `skipped` as done, so resolving it that way
    would complete the bar for a dub that has not been lip-synced at all.
    """
    w = _writer(tmp_path, P.plan_stages({"asr": "bodhan", "mt": "bodhan",
                                         "tts": "bodhan", "mix": True,
                                         "lipsync": None}))
    _log(w, "=== stage 9g: mix ===")
    w.finish("done")
    doc = P.read_progress(tmp_path)
    ls = next(s for s in doc["stages"] if s["key"] == "9h")
    assert ls["state"] == "pending", "the deferred stage was marked skipped"
    assert doc["n_done"] < doc["n_total"]
    lip = next(p for p in doc["phases"] if p["id"] == "lipsync")
    assert lip["state"] == "pending"


def test_a_configured_lipsync_stage_is_not_deferred():
    """The flag is about the SUBMISSION, not about the stage.

    A job that asked for lip-sync and then skipped past it really was
    skipped, and must still resolve that way.
    """
    stages = P.plan_stages({"asr": "whisperx", "mt": "bodhan", "tts": "bodhan",
                            "mix": True, "lipsync": "musetalk"})
    ls = next(s for s in stages if s["key"] == "9h")
    assert "deferred" not in ls


def test_the_later_lipsync_run_reuses_the_deferred_row(tmp_path):
    """`POST .../lipsync` completes the bar rather than appending an 8th node.

    `_open` reuses an existing row for a key, so the pending 9h planted by the
    draft run is the row the real stage marker opens.
    """
    w = _writer(tmp_path, P.plan_stages({"asr": "bodhan", "mt": "bodhan",
                                         "tts": "bodhan", "mix": True,
                                         "lipsync": None}))
    w.finish("done")
    w2 = _writer(tmp_path, P.plan_stages({"asr": "bodhan", "mt": "bodhan",
                                          "tts": "bodhan", "mix": True,
                                          "lipsync": "musetalk"}))
    _log(w2, "=== stage 9h: lip sync (musetalk) ===")
    w2.finish("done")
    doc = P.read_progress(tmp_path)
    assert [s["key"] for s in doc["stages"]].count("9h") == 1
    # `Finished` is the trailing pill now, so Lip sync is the one before it.
    assert doc["phases"][-1]["id"] == P.FINISHED_ID
    lip = next(p for p in doc["phases"] if p["id"] == "lipsync")
    assert lip["state"] == "done"


def test_the_runner_plans_no_lipsync_node():
    """The inverse of the old guard, and for the same reason: the bar must
    match the run.

    It used to pad the plan with a Lip sync node because `"auto"` meant a
    stage that WOULD run without being in `kwargs` yet. Since 2026-10-20 a
    submission ends at the mix, so padding it would promise a stage this run
    never reaches. Lip-sync is its own submission with its own progress.
    """
    import inspect

    from ui import runner
    src = inspect.getsource(runner)
    assert 'plan_cfg["lipsync"] = requested_ls' not in src, (
        "ui/runner.py is padding the stage plan with a lip-sync node again")
    # By attribute, not by string: the NAME survives in the comment that
    # records why the function went, and should.
    assert not hasattr(runner, "_probe_and_pick"), "auto selection is back"


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
