"""Does `--speed-mode segment` actually fill a slot, and does `global` stay put?

The two modes answer the same question differently, and each has a failure
that looks like success:

  * `segment` is supposed to stretch a SHORT clip out to cover its slot. If it
    only ever compresses, a short dub line leaves a hole in the middle of its
    own slot and the mix sounds like the speaker stopped talking. The plan
    would still render, still report a tempo, still exit 0.
  * `global` is the default and must stay bit-identical. A new flag that
    quietly re-routes the default path would change every existing run.
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


def test_global_is_the_default(tmp_path):
    a = _args(tmp_path)
    assert a.speed_mode == "global"
    assert a.mix_strategy == "global_tempo"
    assert a.speed_cap == 1.5
    assert a.speed_slot_margin == 0.0


def test_segment_mode_rejects_a_cap_that_cannot_compress(tmp_path):
    from backend.cli import resolve_speed_setup
    args = _args(tmp_path, "--speed-mode", "segment", "--speed-cap", "1.0")
    with pytest.raises(SystemExit, match="symmetric bound"):
        resolve_speed_setup(args)


def test_segment_and_timing_method_are_mutually_exclusive(tmp_path):
    """They set the same four knobs; silently letting one win is worse."""
    from backend.cli import resolve_speed_setup
    args = _args(tmp_path, "--speed-mode", "segment", "--timing-method", "m2")
    with pytest.raises(SystemExit, match="same"):
        resolve_speed_setup(args)


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
