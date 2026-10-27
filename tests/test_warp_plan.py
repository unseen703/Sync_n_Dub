"""The `warped` producer: slew-limited planning plus a continuous rate curve.

Three layers, tested at three costs:

  * `backend.tempo_plan`  -- pure arithmetic, no audio. This is where the
    policy lives, so most of the assertions are here.
  * `backend.rate_ramp`   -- the envelope and the warp map. Needs numpy and
    scipy, not a stretcher: building the map is separate from applying it.
  * `backend.warp_plan`   -- end to end, needs a time-map stretcher and skips
    without one.

What each layer can get wrong while still looking right:

  * The planner can satisfy the slew limit by quietly dropping it (the
    `slew_relaxed` ladder). `slew_ok OR flagged` is therefore the invariant,
    never `slew_ok` alone -- a test that asserted only the former would pass
    on a planner that had given up.
  * The envelope can be continuous and still land on the wrong rates at the
    joints, which is exactly the lurch the module exists to remove.
  * The warp can hit the right duration with a map that is not monotone,
    which pyrubberband rejects at render time -- long after planning.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from backend.rate_ramp import (available_backend, build_warp, ramp_windows,
                               rate_envelope)
from backend.tempo_plan import (Cue, CuePlan, TempoConfig, plan_tempo,
                                naive_plan, validate)

CFG = TempoConfig()


def _cue(i, start, end, dur, **kw):
    return Cue(id=f"s{i}", start=start, end=end, tts_duration=dur, **kw)


# ---------------------------------------------------------------------------
# tempo_plan: the policy
# ---------------------------------------------------------------------------

def test_the_default_step_is_the_one_that_was_asked_for():
    """0.20, set 2026-10-25. The JND for speech tempo is ~10%, so this is
    about twice the detection threshold -- audible, not a lurch."""
    assert TempoConfig().max_rel_step == 0.20


def test_a_12_to_18_jump_is_smoothed():
    """The case the whole module exists for.

    Neighbour A needs 1.2x and B needs 1.8x. Afterwards either the step is
    inside the limit, or the planner said out loud that it could not get
    there. Asserting only the first would pass on a planner that silently
    dropped the constraint.
    """
    cues = [_cue(0, 0.0, 5.0, 6.0, speaker="S0"),
            _cue(1, 5.0, 10.0, 9.0, speaker="S0")]
    plans = plan_tempo(cues, CFG)
    rep = validate(cues, plans, CFG)
    assert rep["naive"]["max_rel_step"] > 0.4          # the problem is real
    assert (rep["planned"]["slew_ok"]
            or any("slew_relaxed" in p.flags for p in plans))
    if rep["planned"]["slew_ok"]:
        assert rep["planned"]["max_rel_step"] <= CFG.max_rel_step + 1e-4
        # and the fast cue genuinely came down
        assert plans[1].tempo < 1.8


def test_island_duration_is_conserved():
    """The sync contract. Everything else may move; island edges may not."""
    cues = [_cue(i, i * 4.0, i * 4.0 + 3.6, 3.6 + 0.4 * i, speaker="S0")
            for i in range(4)]
    plans = plan_tempo(cues, CFG)
    rep = validate(cues, plans, CFG)
    assert rep["max_island_dur_err_s"] < 1e-3


def test_a_long_gap_splits_the_chain():
    """Past `island_gap_s` the cues are not neighbours in any useful sense,
    and the second island must start exactly where the source did."""
    cues = [_cue(0, 0.0, 2.0, 2.4, speaker="S0"),
            _cue(1, 9.0, 11.0, 2.4, speaker="S0")]
    plans = plan_tempo(cues, CFG)
    assert plans[0].out_start == pytest.approx(0.0, abs=1e-6)
    assert plans[1].out_start == pytest.approx(9.0, abs=1e-6)


def test_a_speaker_change_splits_the_chain():
    cues = [_cue(0, 0.0, 2.0, 2.4, speaker="S0"),
            _cue(1, 2.0, 4.0, 2.4, speaker="S1")]
    plans = plan_tempo(cues, CFG)
    assert plans[1].out_start == pytest.approx(2.0, abs=1e-4)


def test_a_pinned_cue_keeps_its_onset():
    """`pinned` is how an on-screen cue says its mouth is visible. Everything
    else in the island may slide; this may not."""
    cues = [_cue(0, 0.0, 2.0, 2.6, speaker="S0"),
            _cue(1, 2.0, 5.0, 3.4, speaker="S0", pinned=True),
            _cue(2, 5.0, 7.0, 2.2, speaker="S0")]
    plans = plan_tempo(cues, CFG)
    assert abs(plans[1].out_start - cues[1].start) < 1e-3


def test_a_single_cue_island_is_just_the_naive_fit():
    """PORT (3): it returns before SLSQP. There is no neighbour, so there is
    no choice to make, and solving would discard its own answer."""
    c = _cue(0, 0.0, 2.0, 2.4, speaker="S0")
    plans = plan_tempo([c], CFG)
    assert len(plans) == 1
    assert plans[0].tempo == pytest.approx(min(1.2, CFG.tempo_max), abs=1e-6)
    assert plans[0].out_start == 0.0


def test_the_tempo_is_about_speech_not_the_whole_clip():
    """PORT (2). A 6.5 s clip holding 1.0 s of silence in a 5.0 s slot is
    1.30x as a blob, but only 1.22x once the silence is accounted for -- and
    the planner must be bounding the second number, not the first."""
    c = _cue(0, 0.0, 5.0, 6.5, speaker="S0", speech_s=5.5, fixed_s=0.5)
    assert c.required_tempo == pytest.approx(5.5 / 4.5, abs=1e-6)
    assert c.required_tempo < 6.5 / 5.0


def test_out_slot_stays_consistent_with_the_tempo():
    """PORT (4): the end-pin adjusts the last TEMPO, not just `out_end`.

    `out_slot` must remain `speech_s/tempo + fixed_s`, or the renderer
    produces a clip of one length while the plan claims another -- at a
    residue nothing downstream checks.
    """
    cues = [_cue(i, i * 3.0, i * 3.0 + 2.8, 3.0,
                 speaker="S0", speech_s=2.4, fixed_s=0.3) for i in range(3)]
    for p, c in zip(plan_tempo(cues, CFG), cues):
        assert p.out_slot == pytest.approx(
            c.rate_duration / p.tempo + c.fixed_s, abs=1e-6)


def test_an_impossible_island_flags_rather_than_crashes():
    """20 s of speech into a 2 s slot cannot be done. It must still return a
    usable plan and say what it gave up."""
    cues = [_cue(0, 0.0, 2.0, 20.0, speaker="S0"),
            _cue(1, 2.0, 4.0, 2.0, speaker="S0")]
    plans = plan_tempo(cues, CFG)
    assert len(plans) == 2
    assert all(math.isfinite(p.tempo) and p.tempo > 0 for p in plans)
    assert any(p.flags for p in plans)


def test_naive_plan_leaves_every_slot_where_it_found_it():
    cues = [_cue(i, i * 3.0, i * 3.0 + 2.5, 3.0) for i in range(3)]
    for p, c in zip(naive_plan(cues, CFG), cues):
        assert (p.out_start, p.out_end) == (c.start, c.end)


# ---------------------------------------------------------------------------
# rate_ramp: the curve
# ---------------------------------------------------------------------------

def _plan(tempo=1.0, lead=1.0, trail=1.0, slot=2.0):
    return CuePlan(id="s0", tempo=tempo, out_start=0.0, out_end=slot,
                   lead_tempo=lead, trail_tempo=trail)


def test_the_envelope_lands_on_the_boundary_rates():
    """What makes the chain continuous: cue i's trailing rate IS cue i+1's
    leading rate, so each cue must actually start and end on the numbers the
    planner handed it."""
    r = rate_envelope(_plan(tempo=1.0, lead=1.3, trail=0.8), 2.0, CFG)
    assert float(r(np.array([0.0]))[0]) == pytest.approx(1.3, rel=1e-3)
    assert float(r(np.array([2.0]))[0]) == pytest.approx(0.8, rel=1e-3)
    assert float(r(np.array([1.0]))[0]) == pytest.approx(1.0, rel=1e-3)


def test_the_envelope_runs_fast_inside_a_gap():
    """Pause absorption, expressed as a rate instead of as a cut."""
    r = rate_envelope(_plan(tempo=1.05), 2.0, CFG,
                      gap_rates=[(0.8, 1.2, 3.0)])
    assert float(r(np.array([1.0]))[0]) == pytest.approx(3.0, rel=1e-2)
    assert float(r(np.array([0.4]))[0]) == pytest.approx(1.05, rel=1e-2)


def test_the_envelope_has_no_step_in_it():
    """Continuity is the point. A jump between adjacent grid samples would be
    exactly the lurch this replaces."""
    r = rate_envelope(_plan(tempo=1.05, lead=1.4, trail=0.85), 2.0, CFG,
                      gap_rates=[(0.8, 1.2, 3.0)])
    tau = np.linspace(0.0, 2.0, 4001)
    dlog = np.abs(np.diff(np.log(r(tau))))
    assert dlog.max() < 0.05, f"log-rate step of {dlog.max():.3f} in the envelope"


def test_the_warp_hits_the_target_duration_exactly():
    """`k = I/T` in closed form. If this drifts, every cue lands late."""
    p = _plan(tempo=1.3, lead=1.1, trail=1.25, slot=1.9)
    wr = build_warp(p, 2.4, CFG, 24000, int(2.4 * 24000))
    assert wr.realized_duration == pytest.approx(1.9, abs=1e-6)
    assert wr.target_duration == pytest.approx(1.9, abs=1e-9)


def test_the_time_map_is_strictly_increasing_in_both_columns():
    """Exactly what pyrubberband validates, checked here rather than at
    render time after the planning is already paid for."""
    wr = build_warp(_plan(tempo=1.4, lead=1.0, trail=1.2, slot=1.6),
                    2.2, CFG, 24000, int(2.2 * 24000),
                    gap_rates=[(0.9, 1.3, 2.8)])
    tm = wr.time_map
    assert tm[0][0] == 0 and tm[0][1] == 0
    assert np.all(np.diff(tm[:, 0]) > 0)
    assert np.all(np.diff(tm[:, 1]) > 0)


def test_boundary_drift_is_reported_not_fatal():
    """PORT: the original raises. In a pipeline that kills a dub over one
    awkward cue, and the cue still renders at the right LENGTH -- only the
    joint-continuity promise is weaker than advertised."""
    p = _plan(tempo=1.0, lead=2.0, trail=2.0, slot=0.5)
    wr = build_warp(p, 2.0, CFG, 24000, int(2.0 * 24000),
                    max_boundary_drift=0.001, strict=False)
    assert wr.over_drift is True
    assert wr.realized_duration == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError, match="drift"):
        build_warp(p, 2.0, CFG, 24000, int(2.0 * 24000),
                   max_boundary_drift=0.001, strict=True)


def test_ramp_windows_never_overlap():
    """Width is clamped to 40% of the cue, so lead and trail cannot meet --
    if they did, the two corrections would compose into a rate nobody chose."""
    for d in (0.2, 0.5, 1.0, 3.0):
        a0, b0, a1, b1, w = ramp_windows(d, CFG)
        assert 0.0 <= a0 <= b0 <= a1 <= b1 <= d + 1e-9


def test_a_ramp_slides_into_a_nearby_pause():
    """A rate change inside silence is far less audible than mid-phoneme."""
    a0, b0, _a1, _b1, w = ramp_windows(3.0, CFG, silence_spans=[(0.05, 0.40)])
    assert w > 0
    assert a0 > 0.0, "the lead ramp stayed at the cue edge despite a pause"
    assert a0 >= 0.05 - 1e-9 and b0 <= 0.40 + 1e-9


# ---------------------------------------------------------------------------
# warp_plan: end to end
# ---------------------------------------------------------------------------

def _gapped_wav(path: Path, runs, gaps, *, sr=24000, freq=300.0):
    sf = pytest.importorskip("soundfile")
    parts = []
    for i, r in enumerate(runs):
        t = np.arange(int(r * sr)) / sr
        parts.append((0.5 * np.sin(2 * math.pi * freq * t)).astype("float32"))
        if i < len(gaps):
            parts.append(np.zeros(int(gaps[i] * sr), dtype="float32"))
    y = np.concatenate(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)
    return len(y) / sr


def _job(tmp_path: Path, rows):
    """rows: (start, end, runs, gaps). Returns the merged-JSON path."""
    segs = []
    for i, (a, b, runs, gaps) in enumerate(rows):
        w = tmp_path / "tts" / f"c{i}.wav"
        _gapped_wav(w, runs, gaps, freq=260 + 70 * i)
        segs.append({"id": f"s{i}", "start": a, "end": b, "speaker": "S0",
                     "text": "x", "mt_text": "x", "final_text": "x",
                     "tts_raw_audio": str(w)})
    mt = tmp_path / "mt" / "m.mt.json"
    mt.parent.mkdir(parents=True, exist_ok=True)
    mt.write_text(json.dumps({"segments": segs}, indent=2))
    return mt


def test_the_producer_refuses_up_front_without_a_stretcher(tmp_path, monkeypatch):
    """Preflight. Discovering that nothing can render a time map AFTER every
    cue has been planned is the expensive way to find out."""
    from backend import rate_ramp, warp_plan

    monkeypatch.setattr(rate_ramp, "available_backend", lambda: None)
    mt = _job(tmp_path, [(0.0, 4.0, [1.0, 1.0, 1.0], [0.8, 0.8])])
    with pytest.raises(RuntimeError, match="time-map stretcher"):
        warp_plan.plan_warped(mt)


@pytest.mark.skipif(available_backend() is None,
                    reason="no time-map stretcher (pyrubberband+CLI, or pytsmod)")
def test_the_producer_renders_and_the_plan_needs_no_stretching(tmp_path):
    """The shape that makes the mix pure placement.

    Every Piece comes back at `local_tempo == 1.0` over a file that is
    already its planned length, so `_ffmpeg_mix_plan` builds no stretch
    filter at all.
    """
    from backend.timing import plan

    mt = _job(tmp_path, [(0.0, 4.0, [1.0, 1.0, 1.0], [0.8, 0.8]),
                         (4.5, 8.0, [1.0, 1.0], [0.6])])
    tp = plan("warped", mt)

    assert tp.producer == "warped"
    assert tp.summary["tempo_step_ok"] is True
    assert tp.summary["warp_backend"] in ("rubberband", "pytsmod")
    assert tp.summary["cues_failed"] == 0

    data = json.loads(mt.read_text())
    for seg, sp in zip(data["segments"], tp.segments):
        assert Path(seg["tts_warped_audio"]).exists()
        assert len(sp.pieces) == 1
        assert sp.pieces[0].local_tempo == pytest.approx(1.0)
        # the rendered file is the planned slot, to within a frame
        assert sp.qc["rendered_s"] == pytest.approx(
            sp.qc["planned_slot_s"], abs=0.02)


@pytest.mark.skipif(available_backend() is None,
                    reason="no time-map stretcher (pyrubberband+CLI, or pytsmod)")
def test_the_merged_json_is_versioned_when_the_producer_rewrites_it(tmp_path):
    """It adds `tts_warped_audio` to every segment, so it owes a snapshot."""
    from backend.timing import plan

    mt = _job(tmp_path, [(0.0, 4.0, [1.0, 1.0, 1.0], [0.8, 0.8])])
    plan("warped", mt)
    versions = list((mt.parent / "versions").glob("*.warp.json"))
    assert versions, "no mt/versions/*.warp.json snapshot was written"
