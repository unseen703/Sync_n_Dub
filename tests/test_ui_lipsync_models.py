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


def test_a_longer_id_is_not_swallowed_by_a_shorter_one():
    """`"latentsync15".startswith("latentsync")` is True.

    Three separate prefix loops used to test the short key first, so a run
    was relabelled, re-recorded and re-parsed as the wrong model -- found on
    `wav2lip_hd` vs `wav2lip`, and the same shape applies to the LatentSync
    pair. `normalise` tests the longest key first; this is the regression.
    """
    assert lipsync_models.normalise("latentsync15") == "latentsync15"
    assert lipsync_models.normalise("latentsync-15") == "latentsync15"
    assert lipsync_models.normalise("LatentSync15") == "latentsync15"
    assert lipsync_models.normalise("wav2lip") == "wav2lip"


def test_a_removed_id_resolves_by_prefix_or_not_at_all():
    """The accepted cost of dropping the retired FACTS rows (2026-10-28).

    `wav2lip_hd`, `lipgan` and `latentsync` were display-only rows kept so a
    finished job on disk still read correctly. They were removed by request.
    The consequence, pinned here so it stays deliberate: the three `hd_*`
    job folders now label as plain Wav2Lip and quote its 3-5x band, and a
    `latentsync` (1.6) job -- there are none -- would not resolve at all.
    """
    assert lipsync_models.normalise("wav2lip_hd") == "wav2lip"
    assert lipsync_models.normalise("lipgan") is None
    assert lipsync_models.normalise("latentsync") is None
    assert lipsync_models.RETIRED == frozenset()


def test_recorded_variants_still_normalise():
    """The clip runner writes `musetalk-v15`, not `musetalk`."""
    assert lipsync_models.normalise("musetalk-v15") == "musetalk"
    assert lipsync_models.normalise("") is None
    assert lipsync_models.normalise(None) is None
    assert lipsync_models.normalise("sadtalker") is None


def test_every_described_model_has_a_summary_word():
    """`ui/spec.py` derives its wording from FACTS rather than its own
    prefix loop, which is what used to report an HD run as plain Wav2Lip."""
    for key in lipsync_models.FACTS:
        assert spec.LIPSYNC_WORDS[key].startswith("On — ")
    assert len(set(spec.LIPSYNC_WORDS.values())) == len(lipsync_models.FACTS)


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

    known = presets.estimate_runtime(60.0, lipsync_model="musetalk")
    assert known["lipsync_measured"] is True
    assert "not timed" not in known["text"]


def test_a_slower_model_is_never_quoted_as_cheaper():
    """MuseTalk composites at full resolution; Wav2Lip caps at 1080p and
    renders at 96px. The bands must not cross."""
    fast = presets.estimate_runtime(60.0, lipsync_model="wav2lip")
    slow = presets.estimate_runtime(60.0, lipsync_model="musetalk")
    assert slow["low_s"] > fast["low_s"]
    assert slow["high_s"] > fast["high_s"]


def test_options_are_built_from_the_registry():
    rows = lipsync_models.options()
    assert [r["id"] for r in rows] == sorted(available_models())
    assert all(r["label"] for r in rows)
