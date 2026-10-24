"""Does `--speed-mode segment` actually fill a slot, and does `global` stay put?

The two modes answer the same question differently, and each has a failure
that looks like success:

  * `segment` is supposed to stretch a SHORT clip out to cover its slot. If it
    only ever compresses, a short dub line leaves a hole in the middle of its
    own slot and the mix sounds like the speaker stopped talking. The plan
    would still render, still report a tempo, still exit 0.
  * `segment` is the DEFAULT as of 2026-10-22, by request. This line used to
    read "`global` is the default and must stay bit-identical", and that was
    reversed knowingly rather than drifted past: measured on job
    20261021-114009, `global_tempo` met a +4.86 s TTS overrun with
    `global_tempo: 1.0` -- no compression at all -- so 35.50 s of dub was
    laid over a 37 s video and every line landed progressively later than
    the mouth saying it. The timing plan was even BUILT on that run
    (`timing_producer: naive_slot`) and then ignored by the mixer, which is
    the case `pipeline.py:954` warns about.
    Every CLI run's output changes as a result. That is the intent.
  * The slowdown floor has to be REPORTED when it binds. Silently leaving
    1.15 s of silence while claiming the slot is filled is the worse failure,
    because nothing downstream measures it.

Plan-level assertions are exact arithmetic. The one render test needs ffmpeg
and is skipped without a stretch backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import stretch as S
from backend.timing import plan_naive_slot
from conftest import write_tone

SEG_PARAMS = dict(safety_margin_s=0.0, max_tempo=1.5, min_tempo=1.0 / 1.5,
                  use_target_duration=True)


def _plan_input(tmp_path: Path, rows) -> Path:
    """rows: (slot_start, slot_end, clip_seconds) — real wavs behind each."""
    segs = []
    for i, (a, b, clip) in enumerate(rows):
        w = write_tone(tmp_path / f"c{i}.wav", clip, freq=300 + 120 * i)
        segs.append({"id": f"s{i}", "start": a, "end": b, "speaker": "S0",
                     "text": "x", "mt_text": "x", "final_text": "x",
                     "tts_raw_audio": str(w), "tts_audio": str(w)})
    p = tmp_path / "m.mt.json"
    p.write_text(json.dumps({"segments": segs}, indent=2))
    return p


# ---------------------------------------------------------------------------
# the fit itself
# ---------------------------------------------------------------------------

def test_long_clip_is_compressed_into_its_slot(tmp_path):
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0)])
    pc = plan_naive_slot(mj, **SEG_PARAMS).segments[0].pieces[0]
    assert pc.local_tempo == pytest.approx(7.0 / 5.0)
    assert pc.dst_dur == pytest.approx(5.0, abs=1e-3)


def test_short_clip_is_stretched_out_to_fill_its_slot(tmp_path):
    """The half this mode exists for.

    A 3.5 s line in a 5 s slot must come back as 5 s of speech, not 3.5 s of
    speech and 1.5 s of silence.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 3.5)])
    sp = plan_naive_slot(mj, **SEG_PARAMS).segments[0]
    assert sp.pieces[0].local_tempo == pytest.approx(0.7)
    assert sp.pieces[0].dst_dur == pytest.approx(5.0, abs=1e-3)
    assert "unfilled_s" not in sp.qc


def test_a_filled_plan_reports_zero_unfilled_slot(tmp_path):
    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0), (5.0, 10.0, 3.5),
                                (10.0, 15.0, 5.0)])
    plan = plan_naive_slot(mj, **SEG_PARAMS)
    assert plan.summary["unfilled_slot_s"] == 0.0
    assert plan.summary["unfilled_slot_segments"] == 0
    # Everything lands back on the 15 s source timeline.
    assert plan.summary["fit_duration_s"] == pytest.approx(15.0, abs=1e-2)


def test_floor_binding_leaves_silence_and_says_so(tmp_path):
    """With a tight cap the floor binds; the leftover must be reported.

    1.15 s of silence that nothing counts is indistinguishable from a working
    fit until someone listens to the dub.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 3.5)])
    plan = plan_naive_slot(mj, safety_margin_s=0.0, max_tempo=1.1,
                           min_tempo=1 / 1.1, use_target_duration=True)
    sp = plan.segments[0]
    assert sp.qc["tempo_cap_hit"] is True
    assert sp.qc["unfilled_s"] == pytest.approx(1.15, abs=0.01)


# ---------------------------------------------------------------------------
# overlap + gap barrier: keeping speech out of the music
# ---------------------------------------------------------------------------

#: What `--speed-mode segment` and the UI both use.
UI_PARAMS = dict(safety_margin_s=0.0, max_tempo=1.8, min_tempo=0.75,
                 use_target_duration=True, overlap_policy="keep",
                 gap_barrier_s=1.0)


def test_a_nested_cue_is_placed_where_the_source_put_it(tmp_path):
    """Two people talking at once must still be two people talking at once.

    The shift-forward cursor serialises them, which does not merely move the
    line — it moves it OUT of the moment it belongs to, and on a real job
    that moment was 7 s of music.
    """
    mj = _plan_input(tmp_path, [(0.0, 10.0, 10.0), (4.0, 4.5, 0.5)])

    kept = plan_naive_slot(mj, **UI_PARAMS).segments
    assert kept[1].pieces[0].dst_start == pytest.approx(4.0, abs=1e-3)
    assert kept[1].qc["overlap_kept"] is True
    # The cue it talks over still owns the timeline.
    assert kept[0].dst_end == pytest.approx(10.0, abs=1e-2)

    serial = plan_naive_slot(mj, **{**UI_PARAMS, "overlap_policy": "serialize"})
    assert serial.segments[1].pieces[0].dst_start == pytest.approx(10.0, abs=1e-2)


def test_the_overlap_is_reported_rather_than_displaced(tmp_path):
    """`overlap_segments: 0` used to mean "the cursor moved it", not "there
    is none". The plan has to say which."""
    mj = _plan_input(tmp_path, [(0.0, 10.0, 10.0), (4.0, 4.5, 0.5)])
    assert plan_naive_slot(mj, **UI_PARAMS).summary["overlap_segments"] == 1


def test_a_long_gap_is_a_wall_the_cascade_cannot_cross(tmp_path):
    """A cue that has drifted late is sped up to end at its slot, instead of
    running on over whatever fills the gap."""
    rows = [(0.0, 5.0, 12.0),      # overruns at the cap -> 1.67 s of drift
            (5.0, 10.0, 5.0),      # inherits it; 5 s of music follows
            (15.0, 20.0, 5.0)]
    mj = _plan_input(tmp_path, rows)

    walled = plan_naive_slot(mj, **UI_PARAMS).segments
    assert walled[1].qc["gap_after_s"] == pytest.approx(5.0, abs=1e-2)
    assert walled[1].dst_end == pytest.approx(10.0, abs=1e-2)
    assert "overrun_into_gap_s" not in walled[1].qc

    free = plan_naive_slot(mj, **{**UI_PARAMS, "gap_barrier_s": 0.0})
    assert free.segments[1].dst_end == pytest.approx(11.667, abs=1e-2)


def test_what_the_cap_cannot_absorb_is_reported_not_truncated(tmp_path):
    """Losing a line is worse than a short bleed, so the clip is kept and the
    overrun is named."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 12.0), (10.0, 15.0, 5.0)])
    sp = plan_naive_slot(mj, **UI_PARAMS).segments[0]
    assert sp.pieces[0].dst_dur == pytest.approx(12.0 / 1.8, abs=1e-2)
    assert sp.qc["overrun_into_gap_s"] == pytest.approx(1.667, abs=1e-2)


def test_the_music_gap_stays_clean_on_job_20261022_044302(tmp_path):
    """The regression, at the real geometry.

    Seven cues, one of them (26.221-26.575) nested inside 23.926-30.727, and
    7.24 s of music at 30.727-37.966. Serialised, the nested cue landed at
    ~32.2 s — in the middle of the music — and the drift carried the turn
    before it 1.45 s past its slot as well.
    """
    rows = [(0.031, 8.452, 14.848), (8.553, 19.216, 20.992),
            (19.216, 23.673, 9.301), (23.926, 30.727, 6.912),
            (26.221, 26.575, 0.768),            # nested backchannel
            (37.966, 52.752, 22.187), (52.752, 60.038, 9.216)]
    plan = plan_naive_slot(_plan_input(tmp_path, rows), **UI_PARAMS)

    MUSIC = (30.727, 37.966)
    intruders = [s.id for s in plan.segments
                 if s.pieces
                 and s.pieces[0].dst_start < MUSIC[1] - 1e-3
                 and s.dst_end > MUSIC[0] + 1e-3]
    assert not intruders, f"speech rendered over the music: {intruders}"

    # The nested cue speaks at its own moment, over the turn it interrupts.
    assert plan.segments[4].pieces[0].dst_start == pytest.approx(26.221, abs=1e-2)
    # And the dub still lands back on the 60 s source timeline.
    assert plan.summary["fit_duration_s"] == pytest.approx(60.04, abs=0.1)
    assert plan.summary["unfilled_slot_s"] == pytest.approx(1.15, abs=0.01)
    assert plan.summary["unfilled_slot_segments"] == 1


def test_compression_hitting_the_ceiling_is_not_counted_as_unfilled(tmp_path):
    """An overrun is not a pause.

    A clip pinned at the ceiling is too LONG for its slot; the shift-forward
    cursor absorbs it. Counting that as unfilled slot time would make the
    no-pause metric meaningless.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 9.0)])
    plan = plan_naive_slot(mj, safety_margin_s=0.0, max_tempo=1.1,
                           min_tempo=1 / 1.1, use_target_duration=True)
    assert plan.segments[0].qc["tempo_cap_hit"] is True
    assert "unfilled_s" not in plan.segments[0].qc
    assert plan.summary["unfilled_slot_s"] == 0.0


def test_safety_margin_is_itself_a_pause(tmp_path):
    """Why the new mode defaults the margin to 0.0.

    m2 keeps 0.10 s per segment deliberately; that is 0.10 s of silence at the
    end of every slot, which is exactly what --speed-mode segment removes.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 5.0)])
    with_margin = plan_naive_slot(mj, safety_margin_s=0.10, max_tempo=1.5,
                                  min_tempo=1 / 1.5, use_target_duration=True)
    assert with_margin.segments[0].pieces[0].dst_dur == pytest.approx(4.9, abs=1e-3)
    no_margin = plan_naive_slot(mj, **SEG_PARAMS)
    assert no_margin.segments[0].pieces[0].dst_dur == pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------------
# target_duration_s passthrough
# ---------------------------------------------------------------------------

def test_target_duration_overrides_the_slot_when_asked(tmp_path):
    """`backend.edits.set_speed` and stage 9c both write target_duration_s.

    Without `use_target_duration` this producer reads only start/end, so a
    per-segment speed edit is silently ignored in segment mode.
    """
    mj = _plan_input(tmp_path, [(0.0, 5.0, 5.0)])
    data = json.loads(mj.read_text())
    data["segments"][0]["target_duration_s"] = 4.0      # "25% faster"
    mj.write_text(json.dumps(data))
    pc = plan_naive_slot(mj, **SEG_PARAMS).segments[0].pieces[0]
    assert pc.local_tempo == pytest.approx(1.25)
    assert pc.dst_dur == pytest.approx(4.0, abs=1e-3)


def test_control_arm_still_ignores_target_duration(tmp_path):
    """m2's semantics must not shift under it — it is a measured baseline."""
    mj = _plan_input(tmp_path, [(0.0, 5.0, 5.0)])
    data = json.loads(mj.read_text())
    data["segments"][0]["target_duration_s"] = 4.0
    mj.write_text(json.dumps(data))
    pc = plan_naive_slot(mj, safety_margin_s=0.10, max_tempo=1.5,
                         min_tempo=1 / 1.5).segments[0].pieces[0]
    assert pc.local_tempo == pytest.approx(5.0 / 4.9)   # slot, not target


# ---------------------------------------------------------------------------
# the CLI wiring
# ---------------------------------------------------------------------------

def _args(tmp_path, *extra):
    from backend.cli import _parse_args
    v = tmp_path / "v.mp4"
    v.write_bytes(b"\0" * 16)
    return _parse_args(["-i", str(v), *extra])


def test_segment_is_the_default(tmp_path):
    """Changed 2026-10-22 by request. This file's header used to say `global`
    "must stay bit-identical"; that decision was reversed deliberately,
    because `global_tempo` resolved a +4.86 s TTS overrun by applying
    `global_tempo: 1.0` -- i.e. no compression at all -- and the dub drifted
    progressively late against the mouth it was dubbing.

    `--speed-mode` parses to None, not to a literal: the resolver needs to
    tell "nobody said" from "the operator asked", or every --timing-method
    run would collide with a default it never chose.
    """
    a = _args(tmp_path)
    assert a.speed_mode is None
    assert a.speed_cap == 1.8
    assert a.speed_floor == 0.75
    assert a.speed_slot_margin == 0.0

    from backend.cli import resolve_speed_setup
    s = resolve_speed_setup(a)
    assert s.mix_strategy == "timing_plan"
    assert s.timing_producer == "naive_slot"
    assert s.timing_params["max_tempo"] == 1.8
    assert s.timing_params["min_tempo"] == 0.75
    # The companion keys are not optional: leaving the TTS stretch on would
    # fit every clip to its slot twice, once here and once in the mix.
    assert s.tts_apply_stretch is False
    assert s.mix_clip_source == "raw"


def test_global_is_still_reachable(tmp_path):
    """A default, not a lock-in."""
    from backend.cli import resolve_speed_setup
    s = resolve_speed_setup(_args(tmp_path, "--speed-mode", "global"))
    assert s.mix_strategy == "global_tempo"
    assert s.timing_producer is None


def test_a_timing_method_still_works_without_an_explicit_speed_mode(tmp_path):
    """The regression the None default exists to prevent.

    `segment` and `--timing-method` set the same knobs and refuse each
    other. Had the default been the literal "segment", every research-arm
    invocation would have died at startup on a collision with a default the
    operator never typed.
    """
    from backend.cli import resolve_speed_setup
    s = resolve_speed_setup(_args(tmp_path, "--timing-method", "m2"))
    assert s.timing_producer == "naive_slot"
    assert s.mix_strategy == "timing_plan"
    assert s.timing_params["max_tempo"] == 1.50   # the arm's own cap, not 1.8


def test_segment_mode_rejects_a_cap_that_cannot_compress(tmp_path):
    from backend.cli import resolve_speed_setup
    args = _args(tmp_path, "--speed-mode", "segment", "--speed-cap", "1.0")
    with pytest.raises(SystemExit, match="speed-up bound"):
        resolve_speed_setup(args)


def test_segment_mode_rejects_a_floor_above_natural_pace(tmp_path):
    """min_tempo > 1.0 would SPEED UP the clip it is meant to slow down."""
    from backend.cli import resolve_speed_setup
    args = _args(tmp_path, "--speed-mode", "segment", "--speed-floor", "1.2")
    with pytest.raises(SystemExit, match="speed-floor"):
        resolve_speed_setup(args)


def test_segment_and_timing_method_are_mutually_exclusive(tmp_path):
    """They set the same four knobs; silently letting one win is worse.

    Only when `segment` was asked for EXPLICITLY -- see the test above.
    """
    from backend.cli import resolve_speed_setup
    args = _args(tmp_path, "--speed-mode", "segment", "--timing-method", "m2")
    with pytest.raises(SystemExit, match="same"):
        resolve_speed_setup(args)


def test_the_ui_and_the_cli_agree_on_the_bounds(tmp_path):
    """They agree by construction now, and this is what says so.

    `UI_TIMING_DEFAULTS` was a hand-kept COPY of what `resolve_speed_setup`
    builds — the CLI resolver takes an argparse namespace the UI path does
    not have — and this test existed because the two could drift in silence.
    Since 2026-10-23 both call `backend.speed.segment_timing_kwargs`, so the
    assertions below are close to a tautology and are kept deliberately:
    they are what fails if someone reintroduces a second copy, or gives the
    UI its own bounds without routing them through the same function.
    """
    from backend.cli import resolve_speed_setup
    from ui.presets import UI_TIMING_DEFAULTS
    s = resolve_speed_setup(_args(tmp_path))
    for k in ("max_tempo", "min_tempo", "safety_margin_s", "use_target_duration",
              "overlap_policy", "gap_barrier_s"):
        assert UI_TIMING_DEFAULTS["timing_params"][k] == s.timing_params[k], (
            f"UI and CLI disagree on timing_params[{k!r}]")
    assert UI_TIMING_DEFAULTS["mix_strategy"] == s.mix_strategy
    assert UI_TIMING_DEFAULTS["tts_apply_stretch"] == s.tts_apply_stretch
    assert UI_TIMING_DEFAULTS["mix_clip_source"] == s.mix_clip_source


def test_segment_setup_turns_off_the_tts_stretch(tmp_path):
    """Both layers fitting the same clip to the same slot is the
    double-stretch bug, one stage apart."""
    from backend.cli import resolve_speed_setup
    setup = resolve_speed_setup(_args(tmp_path, "--speed-mode", "segment"))
    assert setup.tts_apply_stretch is False
    assert setup.mix_clip_source == "raw"
    assert setup.mix_strategy == "timing_plan"
    assert setup.timing_params["safety_margin_s"] == 0.0
    assert setup.timing_params["min_tempo"] == pytest.approx(1 / 1.5)


def test_global_setup_is_unchanged(tmp_path):
    """The default path must not move. Every existing run depends on it."""
    from backend.cli import resolve_speed_setup
    setup = resolve_speed_setup(_args(tmp_path))
    assert setup.timing_producer is None
    assert setup.timing_params == {}
    assert setup.tts_apply_stretch is True
    assert setup.mix_strategy == "global_tempo"
    assert setup.mix_clip_source == "auto"


# ---------------------------------------------------------------------------
# rendered output
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not S.available_backends(),
                    reason="no time-stretch backend on this host")
def test_rendered_segment_mix_has_no_internal_silence(tmp_path):
    """The plan says the slot is filled; this checks the ffmpeg graph agrees.

    A plan can be perfect and still render a hole -- a stretch filter fed less
    than its analysis window emits nothing and exits 0 (see backend.stretch).
    So: render dialogue only (no M&E to mask a gap) and look for silence.
    """
    import numpy as np
    import soundfile as sf
    from backend.mix import mix_dubbed_audio

    mj = _plan_input(tmp_path, [(0.0, 5.0, 7.0), (5.0, 10.0, 3.5),
                                (10.0, 15.0, 5.0)])
    plan_naive_slot(mj, out_json=tmp_path / "plan.json", **SEG_PARAMS)
    res = mix_dubbed_audio(
        mj, me_stem=None, source_video=tmp_path / "v.mp4",
        out_dir=tmp_path / "mix", sample_rate=48000, mux_video=False,
        strategy="timing_plan", clip_source="raw",
        timing_plan=tmp_path / "plan.json")

    y, sr = sf.read(str(res.audio_wav))
    if y.ndim > 1:
        y = y.mean(axis=1)
    hop = int(0.05 * sr)
    rms = np.array([np.sqrt((y[i:i + hop] ** 2).mean() + 1e-12)
                    for i in range(0, len(y) - hop, hop)])
    # Everything before the 15 s timeline end must be speech. The tail after
    # it is `fit_duration + 0.5` of canvas headroom, trimmed at mux.
    inside = rms[:int(15.0 / 0.05)]
    quiet = (inside < 0.01).sum() * 0.05
    assert quiet < 0.2, f"{quiet:.2f}s of silence inside the dubbed timeline"
