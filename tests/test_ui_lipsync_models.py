"""The registry and what the UI says about it must not drift apart.

`wav2lip_hd` was registered in `models/lipsync/__init__.py` on 2026-10-19 and
reached none of the four UI tables that describe lip-sync models. Nothing
errored: the POST accepted it (it validates against the registry), the
readiness panel listed it (it serves the registry), and only the label, the
runtime estimate and the cost row were missing or wrong. These tests fail on
each half of that.
"""
from __future__ import annotations

import pytest

from models.lipsync import available_models
from ui import cost, lipsync_models, presets, spec


def test_every_registered_model_is_described():
    """A model the registry offers and the UI cannot name.

    It would render in the panel as a bare slug with no speed and no
    estimate — which is exactly how `wav2lip_hd` shipped.
    """
    drift = lipsync_models.unregistered()
    assert drift["missing_facts"] == [], (
        f"registered but undescribed: {drift['missing_facts']}. "
        f"Add them to ui/lipsync_models.py::FACTS.")


def test_no_described_model_has_left_the_registry():
    """The opposite drift: advertising a model the POST would 422 on."""
    drift = lipsync_models.unregistered()
    assert drift["stale_facts"] == [], (
        f"described but unregistered: {drift['stale_facts']}")


def test_hd_is_not_swallowed_by_the_plain_model():
    """`"wav2lip_hd".startswith("wav2lip")` is True.

    Three separate prefix loops tested the short key first, so an HD run was
    relabelled, re-recorded and re-parsed as plain Wav2Lip. `normalise` tests
    the longest key first; this is the regression.
    """
    assert lipsync_models.normalise("wav2lip_hd") == "wav2lip_hd"
    assert lipsync_models.normalise("wav2lip-hd") == "wav2lip_hd"
    assert lipsync_models.normalise("wav2lip") == "wav2lip"
    assert lipsync_models.normalise("Wav2Lip_HD") == "wav2lip_hd"


def test_recorded_variants_still_normalise():
    """The clip runner writes `musetalk-v15`, not `musetalk`."""
    assert lipsync_models.normalise("musetalk-v15") == "musetalk"
    assert lipsync_models.normalise("") is None
    assert lipsync_models.normalise(None) is None
    assert lipsync_models.normalise("sadtalker") is None


def test_summary_line_names_hd_as_hd():
    """`ui/spec.py` said "On — Wav2Lip" for an HD run."""
    word = spec.LIPSYNC_WORDS[lipsync_models.normalise("wav2lip_hd")]
    assert "HD" in word
    assert spec.LIPSYNC_WORDS["wav2lip"] != word


@pytest.mark.parametrize("model", sorted(available_models()))
def test_a_models_cost_is_either_measured_or_declared_unmeasured(model):
    """The guarantee is "no SILENT gap", not "every model has a number".

    A model nobody has timed here is allowed — `latentsync15` is one — but
    it has to say so, by carrying a FACTS entry with `rate: None`. What is
    forbidden is being absent from the table, because then
    `estimate_runtime` falls through and quotes the dub-only band as if it
    were the answer.
    """
    assert model in lipsync_models.FACTS, (
        f"{model} is registered but undescribed in ui/lipsync_models.py")
    measured = lipsync_models.rate_band(model) is not None
    assert (cost._lipsync_rate(model) is not None) is measured
    assert (model in presets.LIPSYNC_FACTORS) is measured
    if not measured:
        assert "not timed" in lipsync_models.speed_text(model)


def test_an_untimed_model_never_quotes_a_bare_number():
    """The silent failure this whole module exists for.

    `estimate_runtime` used to add nothing for a model it did not know and
    return the dub-only band as an ordinary "about X to Y" — so the most
    expensive model on the list advertised as the cheapest.
    """
    est = presets.estimate_runtime(60.0, lipsync_model="definitely_not_a_model")
    assert est["lipsync_measured"] is False
    assert "not timed on this host" in est["text"]

    known = presets.estimate_runtime(60.0, lipsync_model="wav2lip_hd")
    assert known["lipsync_measured"] is True
    assert "not timed" not in known["text"]


def test_hd_is_never_cheaper_than_the_model_it_wraps():
    """It runs that model and then a per-frame GAN pass over the result."""
    plain = presets.estimate_runtime(60.0, lipsync_model="wav2lip")
    hd = presets.estimate_runtime(60.0, lipsync_model="wav2lip_hd")
    assert hd["low_s"] > plain["low_s"]
    assert hd["high_s"] > plain["high_s"]


def test_options_are_built_from_the_registry():
    rows = lipsync_models.options()
    assert [r["id"] for r in rows] == sorted(available_models())
    assert all(r["label"] for r in rows)
