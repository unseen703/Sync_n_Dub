"""Pause absorption and gap borrowing: is the compression landing on silence?

Both features have a failure mode that renders fine and sounds wrong, so
most of what is asserted here is arithmetic rather than "it produced a plan".

  * ABSORPTION's success condition is `speech_tempo == 1.0` on a line whose
    pauses could cover the deficit. A refactor that quietly reverts to the
    blob tempo still emits the right number of pieces, still fills the slot,
    still exits 0 -- the only tell is that the speech rate moved when it did
    not have to. `test_silence_alone_can_cover_the_deficit` is that tell.
  * Its other half is the FLOOR. Absorbing a sentence boundary down to
    nothing welds two sentences together and reads as breathless at 1.0x,
    which is the same complaint the feature exists to answer, relocated.
  * BORROWING must stay off by default and must refuse three cases (nested
    cue, last cue, a slot `set_speed` shortened). The third is the dangerous
    one: borrowing hands back exactly the seconds the edit removed, so the
    edit journals as applied and changes nothing.

The allocator tests need no audio at all; that is deliberate, because they
are the ones that encode the policy. The VAD tests skip without soundfile.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.pause_absorb import (
    DEFAULT_MAX_ABSORB_FRAC,
    PAUSE_TIERS,
    _merge_runs,
    allocate,
    pause_floor,
)
from backend.timing import plan_naive_slot
from conftest import write_tone

BOUNDS = dict(max_speech_tempo=1.8, min_speech_tempo=0.75)


# ---------------------------------------------------------------------------
# the allocator: policy, no audio
# ---------------------------------------------------------------------------

def test_silence_alone_can_cover_the_deficit():
    """The headline case, and the one a regression would hide.

    4.4 s of speech with two 0.8 s gaps in a 5.1 s slot is 0.9 s too long.
    The gaps can spare 0.96 s between them, so the speech must not move at
    all -- a blob tempo would have charged it 6.0/5.1 = 1.18x for a deficit
    the silence could pay by itself.
    """
    a = allocate(4.4, [0.8, 0.8], 5.1, **BOUNDS)
    assert a.speech_tempo == 1.0
    assert a.mode == "absorb"
    assert a.absorbed_s == pytest.approx(0.9, abs=1e-3)
    assert sum(a.kept_gaps) == pytest.approx(0.7, abs=1e-3)
    # and it still lands in the slot
    assert 4.4 + sum(a.kept_gaps) == pytest.approx(5.1, abs=1e-3)


def test_speech_moves_only_for_what_silence_could_not_pay():
    """Deficit 2.0 s, pool 0.96 s -> speech absorbs exactly the remaining 1.04."""
    a = allocate(4.4, [0.8, 0.8], 4.0, **BOUNDS)
    assert a.speech_tempo > 1.0
    assert a.speech_tempo == pytest.approx(4.4 / (4.4 - (2.0 - 0.96)), abs=1e-3)
    # and it is still gentler than the blob tempo it replaces
    assert a.speech_tempo < 6.0 / 4.0


def test_with_no_gaps_it_is_exactly_the_blob_tempo():
    """The degenerate case must not become a second, divergent code path.

    A continuous line has no silence to spend, so absorption has to agree
    with `plan_naive_slot`'s `raw / target` to the last decimal.
    """
    a = allocate(7.0, [], 5.0, **BOUNDS)
    assert a.speech_tempo == pytest.approx(7.0 / 5.0)
    assert a.kept_gaps == []


def test_no_gap_is_absorbed_past_its_floor():
    """A sentence-tier gap keeps its 0.25 s however large the deficit is."""
    a = allocate(4.0, [0.5], 0.1, max_speech_tempo=99.0, min_speech_tempo=0.75,
                 max_absorb_frac=0.99)
    assert a.kept_gaps[0] == pytest.approx(pause_floor(0.5), abs=1e-6)
    assert a.kept_gaps[0] == pytest.approx(0.25, abs=1e-6)


def test_no_gap_gives_up_more_than_max_absorb_frac():
    """The floor is not the only bound: rhythm survives even where it would."""
    a = allocate(4.0, [1.0], 1.0, max_speech_tempo=99.0, min_speech_tempo=0.75,
                 max_absorb_frac=DEFAULT_MAX_ABSORB_FRAC)
    assert a.kept_gaps[0] == pytest.approx(0.4, abs=1e-6)   # 60% taken, not 75%


def test_gaps_give_up_in_proportion_to_what_they_can_spare():
    """Proportional to SPARE, not to length -- every gap ends the same
    fraction of the way to its own floor.

    Not the same fraction of itself: a 0.4 s gap sits nearer its 0.25 s
    sentence floor than a 0.8 s one does, so it has less to give and gives
    less. Draining by length instead would push the short gaps through their
    floors while the long ones still had room, which is how you flatten the
    sentence structure the tiers exist to protect.
    """
    gaps = [0.8, 0.4]
    a = allocate(4.0, gaps, 4.6, **BOUNDS)
    spare = [min(g - pause_floor(g), g * DEFAULT_MAX_ABSORB_FRAC) for g in gaps]
    taken = [g - k for g, k in zip(gaps, a.kept_gaps)]

    assert taken[0] > taken[1]                                   # more seconds
    assert taken[0] / spare[0] == pytest.approx(taken[1] / spare[1], abs=1e-3)
    assert all(k >= pause_floor(g) - 1e-9 for g, k in zip(gaps, a.kept_gaps))


def test_speech_tempo_is_capped_and_says_so():
    a = allocate(4.0, [0.2], 1.0, max_speech_tempo=1.8, min_speech_tempo=0.75)
    assert a.speech_tempo == pytest.approx(1.8)
    assert a.speech_capped is True


# ---------------------------------------------------------------------------
# the allocator, running backwards
# ---------------------------------------------------------------------------

def test_a_short_clip_grows_its_pauses_before_it_drawls():
    """The anti-drawl half. 3.0 s of speech in a 5.0 s slot.

    Today that is a 0.6x stretch over the whole clip, which CLAUDE.md
    describes as audible drawl. Here the 0.5 s gap goes to 1.25 s first and
    the speech only takes what is left.
    """
    a = allocate(3.0, [0.5], 5.0, **BOUNDS)
    assert a.mode == "expand"
    assert a.kept_gaps[0] == pytest.approx(1.25, abs=1e-3)
    assert a.speech_tempo == pytest.approx(0.8, abs=1e-3)   # not 0.6
    assert 3.0 / a.speech_tempo + sum(a.kept_gaps) == pytest.approx(5.0, abs=1e-3)


def test_expansion_with_no_gaps_falls_back_to_the_floor_and_reports_the_hole():
    """No silence to grow, so the floor binds and the leftover is a real pause.

    Same contract as `plan_naive_slot`'s `unfilled_s`: reported, never hidden.
    """
    a = allocate(3.0, [], 5.0, **BOUNDS)
    assert a.speech_tempo == pytest.approx(0.75)
    assert a.speech_capped is True
    assert a.unfilled_s == pytest.approx(1.0, abs=1e-3)


def test_a_clip_that_already_fits_is_left_entirely_alone():
    a = allocate(4.0, [0.5], 4.5, **BOUNDS)
    assert a.mode == "fit"
    assert a.speech_tempo == 1.0
    assert a.kept_gaps == [0.5]


# ---------------------------------------------------------------------------
# tiers and run merging
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gap, expected", [
    (0.60, 0.25),    # sentence boundary
    (0.35, 0.25),    # exactly on the tier edge
    (0.30, 0.12),    # clause
    (0.10, 0.08),    # breath
    (0.05, 0.05),    # already under every floor: clamped to itself
])
def test_pause_floor_tiers(gap, expected):
    assert pause_floor(gap) == pytest.approx(expected)


def test_tiers_are_ordered_longest_first():
    """`pause_floor` returns on the first match, so an out-of-order table
    would silently give every gap the first tier's floor."""
    lengths = [at_least for at_least, _ in PAUSE_TIERS]
    assert lengths == sorted(lengths, reverse=True)


def test_runs_merge_across_the_smallest_gap_first():
    """Losing the 0.1 s breath is cheap; losing the 0.5 s boundary is not."""
    speech = [(0.0, 1.0), (1.1, 2.0), (2.5, 3.0)]
    gaps = [0.1, 0.5]
    out_speech, out_gaps, merged = _merge_runs(speech, gaps, max_runs=2,
                                               min_run_s=0.1)
    assert merged == 1
    assert out_speech == [(0.0, 2.0), (2.5, 3.0)]
    assert out_gaps == [0.5]


def test_runs_shorter_than_the_renderer_floor_are_merged_away():
    """A 20 ms sliver is dropped by the stretch filter and reaches the mix as
    silence (backend.mix._MIN_STRETCH_INPUT_S); merging beats losing it."""
    speech = [(0.0, 1.0), (1.2, 1.22), (1.5, 2.5)]
    out_speech, _, merged = _merge_runs(speech, [0.2, 0.28], max_runs=8,
                                        min_run_s=0.10)
    assert merged >= 1
    assert all(b - a >= 0.10 for a, b in out_speech)


# ---------------------------------------------------------------------------
# gap borrowing, through the real producer
# ---------------------------------------------------------------------------

SEG = dict(safety_margin_s=0.0, max_tempo=1.8, min_tempo=0.75)


def _plan_input(tmp_path: Path, rows) -> Path:
    """rows: (slot_start, slot_end, clip_seconds[, target_duration_s])."""
    segs = []
    for i, row in enumerate(rows):
        a, b, clip = row[0], row[1], row[2]
        w = write_tone(tmp_path / f"c{i}.wav", clip, freq=300 + 120 * i)
        seg = {"id": f"s{i}", "start": a, "end": b, "speaker": "S0",
               "text": "x", "mt_text": "x", "final_text": "x",
               "tts_raw_audio": str(w), "tts_audio": str(w)}
        if len(row) > 3:
            seg["target_duration_s"] = row[3]
        segs.append(seg)
    p = tmp_path / "m.mt.json"
    p.write_text(json.dumps({"segments": segs}, indent=2))
    return p


def test_borrowing_is_off_by_default(tmp_path):
    """Every existing run and every pinned number depends on this."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0), (8.0, 11.0, 2.0)])
    sp = plan_naive_slot(mj, **SEG).segments[0]
    assert sp.pieces[0].local_tempo == pytest.approx(7.0 / 5.0)
    assert "borrowed_s" not in sp.qc


def test_borrowing_extends_the_target_and_lowers_the_tempo(tmp_path):
    """3.0 s gap after the cue, so it may take min(0.4, 1.5) = 0.4 s."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0), (8.0, 11.0, 2.0)])
    sp = plan_naive_slot(mj, borrow_gap_s=0.4, borrow_frac=0.5, **SEG).segments[0]
    assert sp.qc["borrowed_s"] == pytest.approx(0.4)
    assert sp.pieces[0].local_tempo == pytest.approx(7.0 / 5.4, abs=1e-4)
    # the counterfactual, so the trade is legible without re-planning
    assert sp.qc["tempo_unborrowed"] == pytest.approx(7.0 / 5.0, abs=1e-4)


def test_borrowing_is_bounded_by_the_fraction_not_just_the_cap(tmp_path):
    """A 0.4 s gap yields 0.2 s at borrow_frac=0.5, not the full 0.4 cap.

    Half the silence stays silent: the gap is shared, and taking all of it
    leaves the next cue no cushion.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0), (5.4, 9.0, 2.0)])
    sp = plan_naive_slot(mj, borrow_gap_s=0.4, borrow_frac=0.5, **SEG).segments[0]
    assert sp.qc["borrowed_s"] == pytest.approx(0.2, abs=1e-3)


def test_the_last_cue_never_borrows(tmp_path):
    """`_next_slot_start` returns None, so there is no measured gap. Guessing
    one from the source duration is not available here and would be a guess."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 3.0), (8.0, 11.0, 5.0)])
    last = plan_naive_slot(mj, borrow_gap_s=0.4, **SEG).segments[-1]
    assert "borrowed_s" not in last.qc


def test_a_nested_cue_never_borrows(tmp_path):
    """It does not advance the cursor, so two cues would spend the same gap."""
    mj = _plan_input(tmp_path, [(0.0, 8.0, 7.0), (2.0, 2.5, 0.4),
                                (12.0, 14.0, 2.0)])
    plan = plan_naive_slot(mj, borrow_gap_s=0.4, overlap_policy="keep", **SEG)
    nested = plan.segments[1]
    assert nested.qc.get("overlap_kept") is True
    assert "borrowed_s" not in nested.qc


def test_borrowing_refuses_a_slot_that_set_speed_shortened(tmp_path):
    """The silent-no-op this guard exists for.

    `set_speed` works by SHORTENING `target_duration_s`. Borrowing would add
    the seconds straight back, so the edit would journal as applied, re-render
    at exit code 0, and sound exactly as before.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0, 4.0), (8.0, 11.0, 2.0)])
    sp = plan_naive_slot(mj, borrow_gap_s=0.4, use_target_duration=True,
                         **SEG).segments[0]
    assert "borrowed_s" not in sp.qc
    assert sp.qc["borrow_skipped"] == "target_duration_s"
    assert sp.pieces[0].local_tempo == pytest.approx(7.0 / 4.0, abs=1e-4)


def test_the_barrier_wall_moves_out_by_what_was_borrowed(tmp_path):
    """Borrow and barrier are not opposites: the barrier stops a cue running
    on INDEFINITELY, the borrow grants it a measured slice on purpose."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 12.0), (9.0, 12.0, 2.0)])
    sp = plan_naive_slot(mj, borrow_gap_s=0.4, gap_barrier_s=1.0,
                         **SEG).segments[0]
    assert sp.qc["borrowed_s"] == pytest.approx(0.4)
    # overrun is measured against slot.end + borrow, not slot.end
    assert sp.qc["overrun_into_gap_s"] == pytest.approx(
        12.0 / 1.8 - 5.4, abs=1e-3)


def test_the_summary_separates_borrowed_from_overrun(tmp_path):
    """Same seconds of gap, opposite claims: one is a decision, one is damage."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 12.0), (9.0, 12.0, 2.0)])
    s = plan_naive_slot(mj, borrow_gap_s=0.4, gap_barrier_s=1.0, **SEG).summary
    assert s["borrowed_s"] > 0
    assert s["overrun_into_gap_s"] > 0
    assert s["borrowed_segments"] == 1


def test_speech_tempo_is_reported_even_on_the_blob_path(tmp_path):
    """`summarize` must not have to guess which path a segment took."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0)])
    plan = plan_naive_slot(mj, **SEG)
    assert plan.segments[0].qc["speech_tempo"] == pytest.approx(1.4)
    assert plan.summary["speech_tempo_max"] == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# the VAD path, end to end
# ---------------------------------------------------------------------------

def _write_gapped(path: Path, runs, gaps, *, sr: int = 24000, freq: float = 300.0):
    """A wav of alternating tone and digital silence. Returns its duration."""
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    import math

    parts = []
    for i, r in enumerate(runs):
        t = np.arange(int(r * sr)) / sr
        parts.append((0.5 * np.sin(2 * math.pi * freq * t)).astype("float32"))
        if i < len(gaps):
            parts.append(np.zeros(int(gaps[i] * sr), dtype="float32"))
    audio = np.concatenate(parts)
    sf.write(str(path), audio, sr)
    return len(audio) / sr


def test_detect_pauses_finds_the_gaps_that_are_there(tmp_path):
    from backend.pause_absorb import detect_pauses

    dur = _write_gapped(tmp_path / "g.wav", [1.0, 1.0, 1.0], [0.5, 0.5])
    cp = detect_pauses(tmp_path / "g.wav", duration_s=dur, backend="energy")
    if not cp.ok:
        pytest.skip(f"energy VAD unavailable here: {cp.error}")
    assert len(cp.speech) == 3
    assert len(cp.gaps) == 2
    assert all(g == pytest.approx(0.5, abs=0.12) for g in cp.gaps)
    assert cp.speech_s == pytest.approx(3.0, abs=0.25)


def test_absorption_through_the_producer_leaves_the_speech_alone(tmp_path):
    """End to end: 3.0 s of speech + 1.6 s of gaps into a 4.0 s slot.

    The blob tempo would be 4.6/4.0 = 1.15x charged to every syllable. The
    gaps can spare ~0.96 s against a 0.6 s deficit, so the speech must come
    back at exactly 1.0x instead.

    The margin between pool and deficit is deliberately wide: the assertion
    is about the POLICY, and pinning it to a VAD that measures 2.9 s of
    speech on one host and 3.1 s on another would make it a test of the VAD.
    """
    from backend.pause_absorb import detect_pauses

    w = tmp_path / "c0.wav"
    dur = _write_gapped(w, [1.0, 1.0, 1.0], [0.8, 0.8])
    if not detect_pauses(w, duration_s=dur, backend="energy").ok:
        pytest.skip("energy VAD unavailable here")

    mj = tmp_path / "m.mt.json"
    mj.write_text(json.dumps({"segments": [{
        "id": "s0", "start": 0.0, "end": 4.0, "speaker": "S0",
        "text": "x", "mt_text": "x", "final_text": "x",
        "tts_raw_audio": str(w), "tts_audio": str(w)}]}))

    sp = plan_naive_slot(mj, absorb_pauses=True, **SEG).segments[0]
    assert sp.qc["absorb"] == "absorb"
    assert sp.qc["speech_untouched"] is True
    assert sp.qc["speech_tempo"] == pytest.approx(1.0)
    assert sp.qc["pause_absorbed_s"] > 0
    # one piece per speech run, whatever the VAD counted
    assert len(sp.pieces) >= 2
    assert sp.qc["runs"] == len(sp.pieces)
    # pieces stay in order and inside the clip
    assert all(p.src_span.end <= dur + 1e-6 for p in sp.pieces)
    assert [p.dst_start for p in sp.pieces] == sorted(p.dst_start for p in sp.pieces)
    # and the cue still lands in its slot
    assert sp.dst_end == pytest.approx(4.0, abs=0.05)


def test_absorption_falls_back_to_the_blob_when_the_vad_is_useless(tmp_path):
    """A continuous tone has no gaps, so `plan_segment` must decline and the
    producer must render exactly what it always did."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0)])
    sp = plan_naive_slot(mj, absorb_pauses=True, **SEG).segments[0]
    assert len(sp.pieces) == 1
    assert sp.pieces[0].local_tempo == pytest.approx(7.0 / 5.0)
