"""The register the operator never chose has to be visible.

Stage 9b1 picks the TTS delivery register from the uploader's metadata. It is
the one substantive choice in a run that nobody made and nothing displayed:
`backend/jobs.py::_RESULT_FIELDS` is an allow-list and the field was missing
from it, so a finished run recorded no register at all while the log showed
9b1 choosing `educational lecture` and the style reaching the engine.

Both halves are pinned here -- persisted by the backend, surfaced by the UI --
because either one alone leaves the value invisible.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.jobs import _RESULT_FIELDS
from ui import spec


def _rows(manifest):
    out = spec.build(Path("."), manifest=manifest)
    return {l["key"]: l for l in out["lines"]}


def test_manifest_allowlist_carries_the_register():
    """`PipelineResult` has the fields; this list is what reaches disk."""
    assert "content_register" in _RESULT_FIELDS
    assert "content_register_tier" in _RESULT_FIELDS


def test_a_classified_register_is_shown_with_its_tier():
    row = _rows({
        "config": {"tts": "bodhan", "tts_style": "auto"},
        "result": {"content_register": "educational lecture",
                   "content_register_tier": "categories"},
    })["register"]
    assert row["value"] == "educational lecture"
    assert "YouTube category" in row["hint"]


def test_the_llm_tier_is_marked_as_the_least_certain():
    """A metadata hit and a transcript guess must not read alike."""
    row = _rows({
        "config": {"tts": "bodhan", "tts_style": "auto"},
        "result": {"content_register": "news", "content_register_tier": "llm"},
    })["register"]
    assert "least certain" in row["hint"]


def test_abstention_is_stated_not_hidden():
    """`None` means the neutral reading -- a correct answer, not a failure.

    Showing nothing would leave the operator unable to tell "no register" from
    "the panel does not report registers".
    """
    row = _rows({"config": {"tts": "bodhan", "tts_style": "auto"},
                 "result": {}})["register"]
    assert row["value"] == "Neutral"
    assert "default delivery" in row["hint"]


@pytest.mark.parametrize("cfg", [
    {"tts": "bodhan", "tts_style": "news"},      # operator picked it himself
    {"tts": "indicf5", "tts_style": "auto"},     # engine ignores style anyway
    {},                                          # no TTS at all
])
def test_no_row_when_the_pipeline_did_not_choose(cfg):
    assert "register" not in _rows({"config": cfg, "result": {}})


def test_a_pre_fix_job_degrades_quietly():
    """Jobs finished before this change have neither key; no crash, no row."""
    rows = _rows({"config": {"tts": "bodhan"}, "result": {"tts_model": "bodhan"}})
    assert "register" not in rows


# --- MuseTalk batch size (UI default) -------------------------------------

#: MuseTalk's own default, measured at 15496 MiB of a 16311 MiB card -- 815
#: MiB of headroom, which OOMs as soon as anything else is resident (another
#: tenant, or the ~460 MiB the onnxruntime CUDA EP holds after stage 5).
#: Whatever the UI sets must stay under it.
MUSETALK_OWN_DEFAULT = 20


def test_ui_sets_a_musetalk_batch_size_with_headroom():
    """The UI must pin a batch size, and it must be below MuseTalk's own.

    The exact number is NOT asserted. It was hardcoded as 10 in three places
    across two files, so lowering it to 8 -- a deliberate change, because a
    neighbour holding ~4 GiB pushes a batch-10 render past the ceiling --
    failed three tests that had no opinion about 8 versus 10. A test that
    breaks on a value it does not care about trains people to edit tests.
    """
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi")
    n = kw.get("lipsync_batch_size")
    assert isinstance(n, int) and n > 0, "the UI must pin a batch size"
    assert n < MUSETALK_OWN_DEFAULT, (
        f"batch {n} is not below MuseTalk's own default of "
        f"{MUSETALK_OWN_DEFAULT}, which has no headroom on this card")
    assert "batch_size" not in kw, (
        "must not use the SHARED adapter kwarg -- stage 4 picks that one too")


def test_the_batch_size_reaches_the_lipsync_adapter():
    """`batch_size` is the shared adapter kwarg; prove the hop end to end.

    Asserts the value ARRIVES, not what it is: the point is the hop, and
    pinning the number here is what made a one-line tuning change a
    three-test failure.
    """
    from backend.pipeline import _pick
    from models.lipsync.musetalk_adapter import MuseTalkSync
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi")
    picked = _pick({"batch_size": kw["lipsync_batch_size"]},
                   ("repo_path", "checkpoint", "python_bin", "version",
                    "batch_size", "fps", "bbox_shift"))
    assert MuseTalkSync(**picked).batch_size == kw["lipsync_batch_size"]


def test_sharing_the_kwarg_does_not_break_diarization():
    """Stage 4 picks `batch_size` too; pyannote does not accept it."""
    import warnings
    from models.diarization import get_diarizer
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        get_diarizer("pyannote", batch_size=15)   # must not raise


def test_the_batch_size_is_recorded_even_though_nothing_lipsyncs_at_submit():
    """Inverted on 2026-10-20, and the inversion is the point.

    It used to be set inside `wants_lipsync`, on the reasoning that an
    audio-only dub should not carry a lip-sync knob. Since no submission
    lip-syncs any more, that reasoning would mean NO job ever records one --
    and the render, which happens later, rebuilds its kwargs from this job's
    manifest config. The adapter would then fall back to MuseTalk's own 20.

    So it is recorded on every job, including the ones that never lip-sync.
    An unused key in a manifest costs nothing; a silent OOM costs a render.
    """
    from ui import presets
    for preset in ("dub", "draft", "standard", "best"):
        kw = presets.build_kwargs(preset, target_language="hi")
        assert kw.get("lipsync_batch_size"), preset


def test_lipsync_batch_size_is_a_named_pipeline_parameter():
    """Anything reaching `**adapter_kwargs` fails inside an adapter mid-run
    rather than at the boundary -- the rule `test_ui_backend_sync` enforces."""
    import inspect
    from backend.pipeline import _run_pipeline, run_post_mt_stages
    for fn in (_run_pipeline, run_post_mt_stages):
        params = inspect.signature(fn).parameters
        assert "lipsync_batch_size" in params, fn.__name__
        assert params["lipsync_batch_size"].kind is not inspect.Parameter.VAR_KEYWORD
