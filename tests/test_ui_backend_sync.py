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
    keys = [s["key"] for s in plan_stages(presets.plan_config(
        presets.build_kwargs("draft", target_language="hi", tts="bodhan")))]
    assert "9h" not in keys, "a submission planned lip-sync"
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
