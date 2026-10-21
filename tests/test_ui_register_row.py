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

def test_ui_sets_a_musetalk_batch_size_with_headroom():
    """MuseTalk's own default of 20 measured 15496 MiB of a 16311 MiB card.

    815 MiB of headroom OOMs as soon as anything else is resident -- another
    tenant, or the ~460 MiB the onnxruntime CUDA EP holds after stage 5.
    """
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi")
    assert kw.get("lipsync_batch_size") == 10
    assert "batch_size" not in kw, (
        "must not use the SHARED adapter kwarg -- stage 4 picks that one too")


def test_the_batch_size_reaches_the_lipsync_adapter():
    """`batch_size` is the shared adapter kwarg; prove the hop end to end."""
    from backend.pipeline import _pick
    from models.lipsync.musetalk_adapter import MuseTalkSync
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi")
    picked = _pick({"batch_size": kw["lipsync_batch_size"]},
                   ("repo_path", "checkpoint", "python_bin", "version",
                    "batch_size", "fps", "bbox_shift"))
    assert MuseTalkSync(**picked).batch_size == 10


def test_sharing_the_kwarg_does_not_break_diarization():
    """Stage 4 picks `batch_size` too; pyannote does not accept it."""
    import warnings
    from models.diarization import get_diarizer
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        get_diarizer("pyannote", batch_size=15)   # must not raise


def test_no_batch_size_when_lipsync_is_off():
    """It is set inside `wants_lipsync`; an audio-only dub should not carry
    a lip-sync tuning knob into its manifest."""
    from ui import presets
    kw = presets.build_kwargs("standard", target_language="hi",
                              lipsync_model="off")
    assert "lipsync_batch_size" not in kw


def test_lipsync_batch_size_is_a_named_pipeline_parameter():
    """Anything reaching `**adapter_kwargs` fails inside an adapter mid-run
    rather than at the boundary -- the rule `test_ui_backend_sync` enforces."""
    import inspect
    from backend.pipeline import _run_pipeline, run_post_mt_stages
    for fn in (_run_pipeline, run_post_mt_stages):
        params = inspect.signature(fn).parameters
        assert "lipsync_batch_size" in params, fn.__name__
        assert params["lipsync_batch_size"].kind is not inspect.Parameter.VAR_KEYWORD
