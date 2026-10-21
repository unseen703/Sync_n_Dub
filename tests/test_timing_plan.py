"""The timing contract: serialisation, non-overlap, and the DP's two degenerate cases.

The degenerate cases are not decoration. `backend/timing_advanced.py`'s own
docstring states them as the specification of the cost function:

    lambda_pause = lambda_speechrate = lambda_phoneme = 0  ->  naive_slot
    lambda_duration = 0, lambda_pause = 1                  ->  pause_aware onsets

If `plan_dp` does not reproduce its target planner under those settings, the
cost model is not what it says it is, and every number the DP arm produces is
measuring something nobody specified. So the assertions here are written to
report WHAT the DP produced next to WHAT the target produced, not merely that
they differ.
"""
from __future__ import annotations

import json

import pytest

from backend import timing as T
from backend import timing_advanced as TA   # noqa: F401  registers dp/word_anchored

TOL = 1e-3          # 1 ms: finer than drift_bucket_s, coarser than float noise


def onsets(plan: T.TimingPlan):
    return {s.id: [round(p.dst_start, 4) for p in s.pieces] for s in plan.segments}


def tempi(plan: T.TimingPlan):
    return {s.id: [round(p.local_tempo, 4) for p in s.pieces] for s in plan.segments}


def spans(plan: T.TimingPlan):
    return {s.id: [(round(p.src_span.start, 4), round(p.src_span.end, 4))
                   for p in s.pieces] for s in plan.segments}


def describe(name: str, plan: T.TimingPlan) -> str:
    out = [f"{name} ({plan.producer}):"]
    for s in plan.segments:
        out.append(f"  {s.id} slot=[{s.slot.start:.3f},{s.slot.end:.3f}] "
                   f"runs={[(round(r.start, 3), round(r.end, 3)) for r in s.src_speech]}")
        for i, p in enumerate(s.pieces):
            out.append(f"    piece {i}: src=[{p.src_span.start:.3f},"
                       f"{p.src_span.end:.3f}] dst_start={p.dst_start:.4f} "
                       f"tempo={p.local_tempo:.4f} dst_end={p.dst_end:.4f}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the fixture itself -- if these are wrong every other test below is theatre
# ---------------------------------------------------------------------------

def test_fixture_keeps_absolute_and_clip_relative_apart(mini_data):
    """`words[]` are ABSOLUTE; `tts_words[]` are CLIP-RELATIVE.

    Mixing the two is the bug `backend/timing.py` names as most likely, so the
    fixture is built to make it detectable: every segment's tts_words[] start
    at 0.0 and end at its clip length, while its words[] live inside its slot.
    """
    import soundfile as sf

    for seg in mini_data["segments"]:
        words = seg["words"]
        assert words[0]["start"] >= seg["start"] - 1e-9
        assert words[-1]["end"] <= seg["end"] + 1e-9

        tw = seg["tts_words"]
        assert tw[0]["start"] == pytest.approx(0.0, abs=1e-9), (
            f"{seg['id']}: tts_words[] must be clip-relative")
        raw = float(sf.info(seg["tts_raw_audio"]).duration)
        assert raw == pytest.approx(seg["tts_duration_s"], abs=1e-3)
        assert tw[-1]["end"] == pytest.approx(raw, abs=1e-3), (
            f"{seg['id']}: tts_words[] must span the whole clip")
        # And the two coordinate systems must actually be distinguishable.
        if seg["start"] > 0:
            assert tw[0]["start"] != pytest.approx(words[0]["start"])


def test_fixture_exercises_both_split_and_single_run(mini_merged):
    counts = []
    for seg in json.loads(mini_merged.read_text())["segments"]:
        speech, pauses = T.source_speech_and_pauses(seg)
        counts.append(len(speech))
    assert counts == [2, 1, 3], f"fixture drifted: run counts {counts}"


# ---------------------------------------------------------------------------
# schema round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("producer", list(T.PRODUCERS))
def test_jsonable_round_trip(producer, mini_merged, tmp_path):
    plan = T.plan(producer, mini_merged)
    again = T.TimingPlan.from_jsonable(json.loads(
        json.dumps(plan.to_jsonable(), ensure_ascii=False)))

    assert again.producer == plan.producer
    assert again.version == plan.version
    assert len(again.segments) == len(plan.segments)
    for a, b in zip(plan.segments, again.segments):
        assert a.id == b.id and a.speaker == b.speaker
        assert (b.slot.start, b.slot.end) == pytest.approx(
            (a.slot.start, a.slot.end), abs=1e-4)
        assert len(a.pieces) == len(b.pieces)
        for pa, pb in zip(a.pieces, b.pieces):
            assert pb.src_span.start == pytest.approx(pa.src_span.start, abs=1e-4)
            assert pb.src_span.end == pytest.approx(pa.src_span.end, abs=1e-4)
            assert pb.dst_start == pytest.approx(pa.dst_start, abs=1e-4)
            assert pb.local_tempo == pytest.approx(pa.local_tempo, abs=1e-5)
            assert pb.anchor == pa.anchor
            assert pb.src_word_range == pa.src_word_range
        assert len(a.inserted_pauses) == len(b.inserted_pauses)

    # and through a real file, which is how the experiment runner moves them
    dest = plan.write(tmp_path / f"{producer}.plan.json")
    reread = T.TimingPlan.read(dest)
    assert onsets(reread) == onsets(plan)
    assert tempi(reread) == tempi(plan)


@pytest.mark.parametrize("producer", list(T.PRODUCERS))
def test_plan_survives_the_merged_json_bus(producer, mini_merged):
    plan = T.plan(producer, mini_merged)
    T.apply_plan_to_merged(plan, mini_merged)
    back = T.load_plan_from_merged(mini_merged)
    assert back is not None
    assert back.producer == plan.producer
    assert onsets(back) == onsets(plan)


# ---------------------------------------------------------------------------
# the one invariant the renderer depends on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("producer", list(T.PRODUCERS))
def test_pieces_within_a_segment_do_not_overlap(producer, mini_merged):
    """Two pieces of ONE clip overlapping means the clip talks over itself."""
    plan = T.plan(producer, mini_merged)
    for sp in plan.segments:
        for i, (a, b) in enumerate(zip(sp.pieces, sp.pieces[1:])):
            assert a.dst_end <= b.dst_start + TOL, (
                f"{producer}/{sp.id}: piece {i} ends at {a.dst_end:.4f} but "
                f"piece {i + 1} starts at {b.dst_start:.4f}\n"
                + describe(producer, plan))


@pytest.mark.parametrize("producer", list(T.PRODUCERS))
def test_pieces_are_monotone_and_cover_the_clip(producer, mini_merged):
    plan = T.plan(producer, mini_merged)
    import soundfile as sf
    raws = {s["id"]: float(sf.info(s["tts_raw_audio"]).duration)
            for s in json.loads(mini_merged.read_text())["segments"]}
    for sp in plan.segments:
        if not sp.pieces:
            continue
        for a, b in zip(sp.pieces, sp.pieces[1:]):
            assert a.src_span.end <= b.src_span.start + TOL, (
                f"{producer}/{sp.id}: src spans are not monotone\n"
                + describe(producer, plan))
        assert sp.pieces[0].src_span.start == pytest.approx(0.0, abs=TOL)
        assert sp.pieces[-1].src_span.end == pytest.approx(raws[sp.id], abs=1e-2), (
            f"{producer}/{sp.id}: the tail of the clip is dropped on the floor")


@pytest.mark.parametrize("producer", list(T.PRODUCERS))
def test_summary_counts_real_overlap(producer, mini_merged):
    plan = T.plan(producer, mini_merged)
    assert plan.summary["segments"] == 3
    assert plan.summary["pieces"] == sum(len(s.pieces) for s in plan.segments)
    assert plan.summary["overlap_s"] >= 0.0


# ---------------------------------------------------------------------------
# DP degenerate case 1: only the duration term -> naive_slot
# ---------------------------------------------------------------------------

DP_OFF = dict(lambda_pause=0.0, lambda_speechrate=0.0, lambda_phoneme=0.0)


def test_dp_with_only_duration_reproduces_naive_slot(mini_merged):
    naive = T.plan_naive_slot(mini_merged, safety_margin_s=0.10,
                              max_tempo=1.10, min_tempo=0.90, shift_forward=True)
    dp = T.plan_dp(
        mini_merged,
        lambda_duration=1.0, **DP_OFF,
        # naive_slot emits ONE piece per segment, so the DP has to be allowed
        # to as well; with allow_split it would be comparing piece counts, not
        # cost models.
        allow_split=False,
        min_tempo=0.90, max_tempo=1.10,
        safety_margin_s=0.10, shift_forward=True,
    )
    report = describe("naive_slot", naive) + "\n" + describe("dp", dp)

    assert [len(s.pieces) for s in dp.segments] == \
           [len(s.pieces) for s in naive.segments], report
    for a, b in zip(naive.segments, dp.segments):
        for i, (pa, pb) in enumerate(zip(a.pieces, b.pieces)):
            assert pb.dst_start == pytest.approx(pa.dst_start, abs=TOL), (
                f"{a.id} piece {i}: naive placed at {pa.dst_start:.4f}, "
                f"dp placed at {pb.dst_start:.4f} "
                f"(delta {pb.dst_start - pa.dst_start:+.4f}s)\n{report}")
            assert pb.local_tempo == pytest.approx(pa.local_tempo, abs=1e-3), (
                f"{a.id} piece {i}: naive tempo {pa.local_tempo:.4f}, "
                f"dp tempo {pb.local_tempo:.4f}\n{report}")


# Was xfail(strict) while `_inner_dp` charged only the second half of
# D_duration. Fixed by seeding the frontier with the placement error and by
# rounding the terminal cost before the argmin -- the two candidates are
# mathematically tied in the undershoot case, and float association order was
# silently picking the source onset over the slot start.
def test_dp_duration_term_charges_the_placement_error(tmp_path):
    """The undershoot case, which is where the dropped onset term shows.

    Built here rather than in `mini.mt.json` because the fixture is tuned to
    the realistic regime (Indic TTS runs LONGER than its slot) and this is the
    opposite one: a 3.0 s clip in a 5.0 s slot.
    """
    import soundfile as sf
    from conftest import write_tone

    wav = write_tone(tmp_path / "short.wav", 3.0)
    doc = {
        "segments": [{
            "id": "SEG_0000", "start": 0.0, "end": 5.0, "speaker": "SPK_00",
            "final_text": "short clip in a long slot",
            "words": [{"text": "a", "start": 0.20, "end": 0.60},
                      {"text": "b", "start": 0.65, "end": 1.10}],
            "tts_raw_audio": str(wav), "tts_audio": str(wav),
        }],
    }
    merged = tmp_path / "short.mt.json"
    merged.write_text(json.dumps(doc))
    assert sf.info(str(wav)).duration == pytest.approx(3.0, abs=1e-3)

    naive = T.plan_naive_slot(merged, safety_margin_s=0.10,
                              max_tempo=1.10, min_tempo=0.90)
    dp = T.plan_dp(merged, lambda_duration=1.0, **DP_OFF, allow_split=False,
                   min_tempo=0.90, max_tempo=1.10, safety_margin_s=0.10)
    assert dp.segments[0].pieces[0].dst_start == pytest.approx(
        naive.segments[0].pieces[0].dst_start, abs=TOL), (
        describe("naive_slot", naive) + "\n" + describe("dp", dp))


# ---------------------------------------------------------------------------
# DP degenerate case 2: only the pause term -> pause_aware onset anchoring
# ---------------------------------------------------------------------------

PAUSE_ONLY = dict(lambda_duration=0.0, lambda_pause=1.0,
                  lambda_speechrate=0.0, lambda_phoneme=0.0)


def test_dp_with_only_pause_lands_on_every_source_onset(mini_merged):
    """This IS the content of degenerate case 2: onset anchoring, exactly.

    `shift_forward=False` because with `lambda_duration = 0` nothing in the
    objective bounds a segment's LENGTH, so the shift-forward floor from an
    over-long previous segment would move the next segment's onset for a
    reason that has nothing to do with the cost model under test.
    """
    dp = T.plan_dp(mini_merged, **PAUSE_ONLY,
                   min_tempo=0.90, max_tempo=1.15, shift_forward=False)
    for sp in dp.segments:
        assert len(sp.pieces) == len(sp.src_speech), (
            f"{sp.id}: D_pause's unmatched-run term is supposed to make "
            f"one-piece-per-source-run optimal; got {len(sp.pieces)} pieces "
            f"for {len(sp.src_speech)} runs\n" + describe("dp", dp))
        for k, (p, run) in enumerate(zip(sp.pieces, sp.src_speech)):
            assert p.dst_start == pytest.approx(run.start, abs=TOL), (
                f"{sp.id} piece {k}: placed at {p.dst_start:.4f}, source onset "
                f"{run.start:.4f}\n" + describe("dp", dp))


def test_dp_with_only_pause_matches_pause_aware_where_the_chunk_fits(mini_merged):
    """Segments `plan_pause_aware` did not have to degrade must agree exactly.

    `plan_pause_aware` documents one legal departure from the source onset:
    when a chunk physically overruns it falls back to the tightest legal pause
    and flags `pauses_floored`. Everywhere else the two must land on the same
    onsets, and that comparison is the only one that tests the cost model
    rather than that fallback.
    """
    pa = T.plan_pause_aware(mini_merged, max_tempo=1.15, min_tempo=0.90,
                            preserve="onset", lead_in_policy="source",
                            shift_forward=False)
    dp = T.plan_dp(mini_merged, **PAUSE_ONLY,
                   min_tempo=0.90, max_tempo=1.15, shift_forward=False)
    report = describe("pause_aware", pa) + "\n" + describe("dp", dp)

    compared = 0
    for a, b in zip(pa.segments, dp.segments):
        if a.qc.get("pauses_floored") or a.qc.get("no_audio"):
            continue
        compared += 1
        for i, (p_pa, p_dp) in enumerate(zip(a.pieces, b.pieces)):
            assert p_dp.dst_start == pytest.approx(p_pa.dst_start, abs=TOL), (
                f"{a.id} piece {i}: pause_aware {p_pa.dst_start:.4f}, dp "
                f"{p_dp.dst_start:.4f} (delta "
                f"{p_dp.dst_start - p_pa.dst_start:+.4f}s)\n{report}")
    assert compared, "no undegraded segment to compare; fixture lost its point"


@pytest.mark.xfail(strict=True, reason=(
    "`plan_pause_aware` cannot reproduce onset anchoring on a segment whose "
    "clip overruns, because it is constrained to ONE tempo for the whole "
    "segment and to cut points proportional to the source run durations. The "
    "DP is free to re-cut, so it hits every onset exactly; pause_aware's "
    "chunk runs past the next onset and it degrades to prev_end + "
    "pause_min_keep_s. Measured on the fixture: SEG_0000 piece 1 is 433 ms "
    "late (2.2330 vs 1.8000) and SEG_0002 piece 2 is 277 ms late (12.6765 vs "
    "12.4000); both segments carry qc.pauses_floored. The deviation is "
    "documented behaviour, so the DEGENERATE CASE as written in "
    "timing_advanced's docstring ('reproduces pause_aware') is only true of "
    "pause_aware's SEMANTICS, which the DP meets; it is not true of "
    "plan_pause_aware's output."))
def test_dp_with_only_pause_reproduces_pause_aware_everywhere(mini_merged):
    pa = T.plan_pause_aware(mini_merged, max_tempo=1.15, min_tempo=0.90,
                            preserve="onset", lead_in_policy="source",
                            shift_forward=False)
    dp = T.plan_dp(mini_merged, **PAUSE_ONLY,
                   min_tempo=0.90, max_tempo=1.15, shift_forward=False)
    report = describe("pause_aware", pa) + "\n" + describe("dp", dp)
    for a, b in zip(pa.segments, dp.segments):
        assert len(a.pieces) == len(b.pieces), report
        for i, (p_pa, p_dp) in enumerate(zip(a.pieces, b.pieces)):
            assert p_dp.dst_start == pytest.approx(p_pa.dst_start, abs=TOL), (
                f"{a.id} piece {i}: pause_aware {p_pa.dst_start:.4f}, dp "
                f"{p_dp.dst_start:.4f}\n{report}")


# ---------------------------------------------------------------------------
# producer-specific behaviour the suite file relies on
# ---------------------------------------------------------------------------

def test_pause_aware_reinserts_the_source_pause(mini_merged):
    plan = T.plan_pause_aware(mini_merged)
    seg0 = next(s for s in plan.segments if s.id == "SEG_0000")
    assert len(seg0.pieces) == 2
    assert len(seg0.inserted_pauses) == 1
    # SEG_0000's source pause is 1.10 -> 1.80, i.e. 0.70 s.
    assert seg0.src_pauses[0].dur == pytest.approx(0.70, abs=1e-6)


def test_word_anchored_uses_tts_words_not_source_words(mini_merged):
    plan = T.plan_word_anchored(mini_merged, require_tts_words=True)
    assert plan.summary["degraded_segments"] == 0, (
        "every fixture segment has tts_words[]; none should degrade")
    import soundfile as sf
    raws = {s["id"]: float(sf.info(s["tts_raw_audio"]).duration)
            for s in json.loads(mini_merged.read_text())["segments"]}
    for sp in plan.segments:
        for p in sp.pieces:
            # src_span is CLIP-RELATIVE: it can never exceed the clip length,
            # which it trivially would if source (absolute) times leaked in.
            assert 0.0 - TOL <= p.src_span.start <= raws[sp.id] + TOL
            assert 0.0 - TOL <= p.src_span.end <= raws[sp.id] + TOL


def test_unknown_producer_is_rejected(mini_merged):
    with pytest.raises(ValueError, match="unknown timing producer"):
        T.plan("magic", mini_merged)
