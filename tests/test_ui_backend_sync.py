"""The UI's forecast and the pipeline's record have to describe one run.

Three surfaces drift apart independently, and each drift is invisible at
runtime because every one of them degrades into a *plausible* answer:

  * `ui/presets.build_kwargs` -> `_run_pipeline`'s signature. A kwarg the
    pipeline does not take is a TypeError at launch, after the job folder
    exists and the operator has been quoted a time.
  * `_run_pipeline`'s manifest `config` -> `ui/progress.plan_stages`. The
    pipeline RE-ALLOCATES at startup and overwrites whatever the submitter
    wrote, so the manifest is the only durable record of what ran. When it
    omits a stage switch, every later reader (`ui/spec.py`, `ui/cost.py`,
    `_list_jobs`, `edit_agent._config`) reports a run that did less than it
    did, and `plan_stages` under-plans against a progress doc that is right.
  * `plan_stages` -> the `=== stage N ===` markers. Covered by
    `tests/test_ui_progress.py`; not repeated here.
"""
from __future__ import annotations

import inspect

import pytest

from backend.pipeline import _run_pipeline
from ui import presets
from ui.progress import plan_stages

PRESET_IDS = ("draft", "standard", "best")


def _kwargs(preset: str, **over):
    base = dict(target_language="hi", tts="bodhan", lipsync_model="auto")
    base.update(over)
    return presets.build_kwargs(preset, **base)


@pytest.mark.parametrize("preset", PRESET_IDS)
def test_every_ui_kwarg_is_an_explicit_pipeline_parameter(preset):
    """Not merely accepted — accepted BY NAME.

    `_run_pipeline` ends in `**adapter_kwargs`, which swallows anything. A
    kwarg that lands there is forwarded to some adapter and only fails
    where that adapter uses it, which for a hosted engine means a 4xx in
    the middle of a run: `render_mode` reached stage 7 that way and died
    on `unexpected field(s)` after diarization, faces and ASD were paid for.
    """
    params = inspect.signature(_run_pipeline).parameters
    explicit = {n for n, p in params.items() if p.kind is not p.VAR_KEYWORD}
    absorbed = sorted(k for k in _kwargs(preset) if k not in explicit)
    assert not absorbed, (
        f"{preset}: these reach the pipeline only through **kwargs and will "
        f"fail inside an adapter rather than at the boundary: {absorbed}")


@pytest.mark.parametrize("preset", PRESET_IDS)
def test_the_manifest_records_every_switch_the_planner_reads(preset):
    """`plan_stages` must be able to re-derive the run from the manifest.

    The pipeline's own `allocate(config=...)` is the dict that survives, so
    every key `plan_stages` branches on has to be in it. It was missing 11,
    including `emotion`, `refine_speakers`, `visemes` and `face_detector`.
    """
    src = inspect.getsource(plan_stages)
    read = {ln.split('config.get("')[1].split('"')[0]
            for ln in src.splitlines() if 'config.get("' in ln}

    alloc = inspect.getsource(_run_pipeline)
    start = alloc.index("_jobs.allocate(")
    written = {ln.split('"')[1]
               for ln in alloc[start:alloc.index("job_id=job_id", start)].splitlines()
               if ln.strip().startswith('"')}

    missing = sorted(read - written)
    assert not missing, (
        f"plan_stages branches on {missing}, which _run_pipeline never "
        f"writes into the manifest — a finished job cannot be re-planned")


def test_planner_and_pipeline_agree_on_a_standard_run():
    """End to end on the config a real submission produces."""
    cfg = presets.plan_config(_kwargs("standard"))
    keys = [s["key"] for s in plan_stages(cfg)]
    # The optional stages a default UI run actually executes, verified
    # against the 2026-10-17 10 s Batman run's own log markers.
    for k in ("4c", "9b1", "9b2", "9c", "9d", "9e", "9f1", "9f1b", "9f2", "9g"):
        assert k in keys, f"{k} runs on a standard job but is not planned"


def test_no_submission_plans_lipsync_but_faces_still_run():
    """Changed 2026-10-20. `draft` used to skip stage 5 along with 9h,
    because faces were only for lip-sync. Faces and ASD now run on every job
    — their output is the face size the review screen shows, and
    `assignments.json` also feeds the gender vote — while 9h is gone from
    every submission, since lip-sync is a separate, explicit step.
    """
    stages = plan_stages(presets.plan_config(
        presets.build_kwargs("draft", target_language="hi", tts="bodhan")))
    keys = [s["key"] for s in stages]
    # 9h is DRAWN (pending, since 2026-10-21) but not configured: a
    # submission still commits no GPU to it. `deferred` is what separates
    # "still to come" from "this run will do it".
    ls = next(s for s in stages if s["key"] == "9h")
    assert ls["deferred"] is True, "a submission configured lip-sync"
    assert ls["state"] == "pending"
    assert "5" in keys and "6" in keys


# ---------------------------------------------------------------------------
# the spoken language
# ---------------------------------------------------------------------------

def test_english_is_the_default_spoken_language():
    """Both layers, because a default on one of them is not a default.

    It used to be None everywhere, meaning "detect". The ASR adapter posts
    ONE REQUEST PER DIARIZATION TURN, so that detection ran on 1-5 s of
    audio and flapped: on 2026-10-17 a single English lecture came back
    transcribed into Telugu, Gujarati, Malayalam and English across nine
    turns, and MT translated each as if its script were real — which is
    why that dub was part Hindi and part untouched English.
    """
    import inspect

    from backend.pipeline import _run_pipeline

    assert presets.default_source_language() == "en"
    assert inspect.signature(_run_pipeline).parameters["asr_language"].default == "en"
    kw = presets.build_kwargs("standard", target_language="hi", tts="bodhan")
    assert kw["asr_language"] == "en"


def test_detection_is_still_reachable_on_purpose():
    """`auto` has to keep working — English is a default, not a lock-in."""
    kw = presets.build_kwargs("standard", target_language="hi", tts="bodhan",
                              source_language="auto")
    assert kw["asr_language"] is None


def test_a_stated_language_beats_the_default():
    kw = presets.build_kwargs("standard", target_language="hi", tts="bodhan",
                              source_language="ta")
    assert kw["asr_language"] == "ta"


def test_the_asr_adapter_pins_one_language_for_the_whole_file():
    """One probe, then reuse — not one detection per turn.

    Greps the adapter rather than mocking a transport, so the guard
    survives a refactor of the request loop.
    """
    import inspect

    from models.asr import bodhan_adapter as BA

    src = inspect.getsource(BA.BodhanASR.transcribe)
    assert "detected_lang" in src
    assert "language or detected_lang" in src, (
        "the detected language is recorded but not fed back into the next "
        "request — that is the per-turn flapping this guards")


# ---------------------------------------------------------------------------
# the fourth surface: how a run was PACED -> how an edit replays it
# ---------------------------------------------------------------------------
#
# `edit_agent.rerun` recovers a run's timing setup by re-parsing its argv. A
# UI job has no pipeline argv — `ui/runner.py` is launched with the job id
# alone — so the replay fell through to argparse DEFAULTS, which describe a
# run nobody launched. Measured on job 20261022-044302, which has both plans
# on disk: the run wrote `use_target_duration: True, max_tempo: 1.8`, the
# replay wrote `use_target_duration: False, max_tempo: 1.1`.

#: Every field `rerun` takes off the `SpeedSetup`. Each one that is not
#: recovered is silently replaced by a CLI default on every UI replay.
PACING_FIELDS = ("timing_producer", "timing_params", "tts_apply_stretch",
                 "mix_strategy", "mix_tempo_cap", "mix_clip_source")


@pytest.mark.parametrize("preset", PRESET_IDS)
def test_a_ui_submission_records_the_pacing_its_replay_needs(preset):
    """`plan_config` is the manifest, and the manifest is the fallback."""
    from backend.edit_agent import speed_setup_from_config

    cfg = presets.plan_config(_kwargs(preset))
    setup = speed_setup_from_config(cfg)
    assert setup is not None, (
        "the manifest records no pacing at all, so an edit replay cannot "
        "reproduce how this run was timed")
    assert setup.mix_strategy == "timing_plan"
    assert setup.tts_apply_stretch is False
    assert setup.mix_clip_source == "raw"


@pytest.mark.parametrize("preset", PRESET_IDS)
def test_use_target_duration_survives_the_round_trip(preset):
    """The single field a speed edit depends on.

    `backend/edits.py::set_speed` works ENTIRELY by shortening
    `segments[].target_duration_s`, and `backend/timing.py::naive_slot` reads
    that field only when `use_target_duration` is set — it defaults to False
    and the timing docstring says so. Lose this one key on the replay and a
    speed edit applies, re-renders, reports success and comes back at exactly
    the original speed.
    """
    from backend.edit_agent import speed_setup_from_config

    for cfg in (presets.plan_config(_kwargs(preset)), _kwargs(preset)):
        setup = speed_setup_from_config(cfg)
        assert setup is not None
        assert (setup.timing_params or {}).get("use_target_duration") is True


def test_every_pacing_field_the_replay_reads_is_recorded():
    """Anti-drift, the same shape as the `plan_stages` guard above.

    A field added to `SpeedSetup` and consumed by `rerun`, but never written
    by `plan_config`, degrades into an argparse default on every UI replay —
    which is how this whole class of bug is silent.
    """
    from backend.cli import SpeedSetup

    cfg = presets.plan_config(_kwargs("standard"))
    for field in SpeedSetup._fields:
        if field == "tts_force_align":
            continue            # recorded under its own key, asserted above
        assert field in cfg, (
            f"`SpeedSetup.{field}` is read by edit_agent.rerun but "
            f"ui.presets.plan_config never records it, so a replay will use "
            f"the argparse default instead of this run's value")


def test_a_job_with_no_recorded_pacing_returns_none_rather_than_guessing():
    """Jobs written before 2026-10-22 have none of these keys.

    Answering with a half-built setup would re-pace them from defaults while
    claiming to have recovered something, which is worse than the honest
    fallback the caller already has.
    """
    from backend.edit_agent import speed_setup_from_config

    assert speed_setup_from_config({}) is None
    assert speed_setup_from_config({"timing_producer": "naive_slot"}) is None


def test_a_ui_job_argv_is_not_fed_to_the_pipeline_parser(tmp_path, capsys):
    """No argparse usage block on a staged edit.

    `ui/runner.py` is launched with the job id alone, so `recover_args` used
    to hand `['<job_id>']` to the pipeline parser, which printed its whole
    usage block plus `error: the following arguments are required:
    --input/-i` to stderr before raising. The SystemExit was caught and
    nothing was wrong — but it ran on EVERY staged edit, and an operator who
    had just changed a voice saw a wall of text with "error" in it and read
    their change as having failed.
    """
    import json

    from backend import edit_agent as A

    root = tmp_path / "jobs" / "20261022-070010__demo__hi"
    (root / "mt").mkdir(parents=True)
    (root / "mt" / "demo.bodhan.hi.mt.json").write_text(json.dumps(
        {"segments": [{"id": "a", "start": 0.0, "end": 1.0, "mt_text": "x"}],
         "extra": {}}))
    (root / "manifest.json").write_text(json.dumps(
        {"version": 2, "job_id": root.name, "status": "ok",
         "input": "demo.mp4", "config": {"tts": "bodhan"},
         "argv": [root.name]}))

    view = A.open_job(root.name, data_dir=tmp_path)
    capsys.readouterr()                      # drop anything already buffered
    assert A.recover_args(view) is None
    captured = capsys.readouterr()
    assert "usage:" not in captured.err, "argparse printed its usage block"
    assert "error:" not in captured.err, (
        "a staged edit still prints something that reads as a failure")


def test_a_real_pipeline_argv_is_still_re_parsed(tmp_path):
    """The detection must not throw the CLI case away with the UI one."""
    import json

    from backend import edit_agent as A

    root = tmp_path / "jobs" / "cli_job"
    (root / "mt").mkdir(parents=True)
    (root / "mt" / "demo.bodhan.hi.mt.json").write_text(json.dumps(
        {"segments": [{"id": "a", "start": 0.0, "end": 1.0, "mt_text": "x"}],
         "extra": {}}))
    # A real file: the parser checks `--input` exists, so a bare name would
    # fail for a reason this test is not about.
    src = tmp_path / "demo.mp4"
    src.write_bytes(b"\0" * 64)
    (root / "manifest.json").write_text(json.dumps(
        {"version": 2, "job_id": "cli_job", "status": "ok",
         "input": str(src), "config": {"tts": "bodhan"},
         "argv": ["--input", str(src), "--mt-target-language", "hi"]}))

    ns = A.recover_args(A.open_job("cli_job", data_dir=tmp_path))
    assert ns is not None, "a genuine pipeline command line stopped parsing"
    assert ns.mt_target_language == "hi"
