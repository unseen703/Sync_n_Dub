"""Lip-sync is a second, explicit submission against a model the operator names.

Changed 2026-10-20. Before this, `preset=standard` meant `lipsync="auto"` and
`ui/runner._probe_and_pick` resolved that from a face-size probe — so the most
expensive stage in the pipeline (musetalk measures 8-21x clip length on this
host against wav2lip's 3-5x) was selected by a number the operator never saw,
and silently fell back to wav2lip whenever the probe failed.

Now: one preset, phase one always stops at the mix, the face size is MEASURED
AND SHOWN, and `POST .../lipsync` refuses without a model rather than
defaulting to musetalk.

The measurement also changed source. The submit probe took the largest face in
sampled frames; the review screen reports the average size of the face ASD
ATTRIBUTED TO A SPEAKER, which is the mouth lip-sync will actually repaint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui import presets as P                                    # noqa: E402
from ui.progress import plan_stages                            # noqa: E402


# ---------------------------------------------------------------------------
# One preset, and the retired ids still resolve
# ---------------------------------------------------------------------------

def test_there_is_one_preset():
    assert P.PRESET_IDS == (P.PRESET_ID,)
    assert set(P.PRESETS) == {P.PRESET_ID}


@pytest.mark.parametrize("old", ["draft", "standard", "best"])
def test_a_retired_preset_id_still_works(old):
    """A browser holding the previous bundle still posts `standard`, and every
    job already on disk names one of these."""
    assert P.resolve_preset(old) == P.PRESET_ID
    kw = P.build_kwargs(old, target_language="hi")
    assert kw.get("lipsync_model") is None


def test_an_unknown_preset_is_still_rejected():
    with pytest.raises(ValueError):
        P.resolve_preset("turbo")


# ---------------------------------------------------------------------------
# Phase one never lip-syncs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["dub", "draft", "standard", "best"])
def test_no_submission_configures_lip_sync(preset):
    kw = P.build_kwargs(preset, target_language="hi")
    assert "lipsync_model" not in kw or kw["lipsync_model"] is None
    keys = [s["key"] for s in plan_stages(P.plan_config(kw))]
    assert "9h" not in keys, "a submission planned lip-sync"
    assert keys[-1] == "9g", f"phase one must end at the mix, ended at {keys[-1]}"


def test_auto_is_gone_from_the_runner():
    """`"auto"` was a UI token that resolved to a model. Reintroducing it is
    how the choice silently leaves the operator again."""
    import ui.runner as R
    # The NAME may survive in the comment that records why it went; what
    # must not survive is the function.
    assert not hasattr(R, "_probe_and_pick")
    assert hasattr(R, "_probe_face"), "the measurement must still be taken"


def test_faces_and_asd_run_even_though_lip_sync_does_not():
    """Their output is what the review screen reads — and gating them on a
    decision that has not been made yet is what made the face size
    invisible."""
    kw = P.build_kwargs("dub", target_language="hi")
    assert kw["face_detector"] == "retinaface"
    assert kw["asd_model"] == "talknet"
    keys = [s["key"] for s in plan_stages(P.plan_config(kw))]
    assert "5" in keys and "6" in keys


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------

def test_the_lipsync_route_refuses_without_a_model():
    src = (REPO_ROOT / "ui" / "routes_edits.py").read_text()
    assert "A lip-sync model is required to proceed" in src
    assert 'model = "musetalk"' not in src, "the musetalk default is back"


def test_readiness_reports_face_size_from_asd_not_the_biggest_face():
    src = (REPO_ROOT / "ui" / "routes_edits.py").read_text()
    assert "/api/jobs/{job_id}/lipsync/readiness" in src
    assert "_speaker_face_tracks" in src, (
        "face size must come from the ASD-attributed track, not a frame probe")
    assert '"model_required": True' in src


def test_readiness_and_the_post_agree_on_the_blockers():
    """`can_start` drives a disabled button; if it and the POST disagree the
    UI offers an action the server then refuses."""
    import ui.routes_edits as RE
    assert hasattr(RE, "_lipsync_blockers")
    src = (REPO_ROOT / "ui" / "routes_edits.py").read_text()
    # Both the GET and the POST must go through it, not restate it. Matched
    # on the ASSIGNMENT so the `def` line does not count, and without the
    # closing paren so the POST's `, model=model` still matches — it passes
    # the model so the compute cap can quote the runtime for the model
    # actually chosen.
    assert src.count("= _lipsync_blockers(view, cfg") == 2
    assert src.count("Apply or discard them first") == 1
