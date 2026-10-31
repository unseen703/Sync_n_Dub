"""The `--speed-mode` knob values, shared by the CLI and the frontend.

`backend.cli.resolve_speed_setup` builds these from an argparse namespace;
`server.presets` calls the same functions with no arguments and gets the
argparse defaults, so a bare CLI run and a UI run pace identically by
construction.

Deliberately dependency-free: `server/presets.py` is imported when the web
server starts, and routing this through `backend.cli` would drag argparse
and all seven model registries into that import.

The keyword defaults mirror `backend/cli.py`'s `--speed-*` arguments.
"""
from __future__ import annotations

from typing import Any, Dict

#: Default speed-UP bound (`--speed-cap`). Hindi TTS runs 1.21-1.65x longer
#: than the English it covers, so 1.5 sat on top of that range with no
#: headroom.
DEFAULT_SPEED_CAP = 1.8

#: Default slow-DOWN bound (`--speed-floor`). Not 1/1.8: a symmetric floor
#: would slow a short line to 0.56x to fill its slot, which is audible as
#: drawl. What it cannot absorb stays a real pause and is reported as
#: `unfilled_slot_s`.
DEFAULT_SPEED_FLOOR = 0.75

#: Any margin is itself a pause, so the default is none.
DEFAULT_SLOT_MARGIN = 0.0

#: A source gap at least this long is a wall the shift-forward cursor may
#: not cross (`--speed-gap-barrier`).
DEFAULT_GAP_BARRIER = 1.0

#: Seconds of the following gap a cue may take to avoid speeding up
#: (`--speed-borrow-gap`). 0.0 = off, deliberately: a non-zero value moves
#: the tempo of every cue with a gap after it, so every existing run and
#: pinned test number would change. 0.3-0.5 is the range the
#: prosodic-alignment literature uses (Virkar et al. 2022).
DEFAULT_BORROW_GAP = 0.0

#: Of the gap that is there, what fraction a cue may take
#: (`--speed-borrow-frac`). Half, so the next cue keeps a cushion.
DEFAULT_BORROW_FRAC = 0.5

#: Spend a cue's deficit on its own silence before its speech
#: (`--absorb-pauses`). Off by default: it changes what every multi-run cue
#: renders at.
DEFAULT_ABSORB_PAUSES = False

#: Cap on what any single gap may give up, floor or no floor
#: (`--absorb-max-frac`). A dub with every pause floored is relentless.
DEFAULT_ABSORB_MAX_FRAC = 0.60

#: VAD backend for pause detection (`--absorb-vad-backend`). Energy rather
#: than pyannote: a TTS clip is clean audio with near-digital silence in its
#: gaps, and pyannote would hold VRAM into stage 9h.
DEFAULT_ABSORB_VAD_BACKEND = "energy"


def segment_timing_params(
    *,
    speed_cap: float = DEFAULT_SPEED_CAP,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
    slot_margin: float = DEFAULT_SLOT_MARGIN,
    gap_barrier: float = DEFAULT_GAP_BARRIER,
    borrow_gap: float = DEFAULT_BORROW_GAP,
    borrow_frac: float = DEFAULT_BORROW_FRAC,
    absorb_pauses: bool = DEFAULT_ABSORB_PAUSES,
    absorb_max_frac: float = DEFAULT_ABSORB_MAX_FRAC,
    absorb_vad_backend: str = DEFAULT_ABSORB_VAD_BACKEND,
) -> Dict[str, Any]:
    """`timing_params` for the `naive_slot` producer in segment mode."""
    return {
        "max_tempo": speed_cap,
        "min_tempo": speed_floor,
        "safety_margin_s": slot_margin,
        # Both default to "off", so this dict is unchanged for any caller
        # that does not ask.
        "borrow_gap_s": borrow_gap,
        "borrow_frac": borrow_frac,
        "absorb_pauses": absorb_pauses,
        "absorb_max_frac": absorb_max_frac,
        "absorb_vad_backend": absorb_vad_backend,
        # Simultaneous speech stays simultaneous, and the cascade may not
        # walk a line out over the music. Corrections, not preferences, so
        # neither has a flag; the gap threshold does.
        "overlap_policy": "keep",
        "gap_barrier_s": gap_barrier,
        # Honour a per-segment target written by stage 9c or by
        # `backend.edits.set_speed`; without this they are ignored here.
        "use_target_duration": True,
    }


def segment_timing_kwargs(
    *,
    speed_cap: float = DEFAULT_SPEED_CAP,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
    slot_margin: float = DEFAULT_SLOT_MARGIN,
    gap_barrier: float = DEFAULT_GAP_BARRIER,
    borrow_gap: float = DEFAULT_BORROW_GAP,
    borrow_frac: float = DEFAULT_BORROW_FRAC,
    absorb_pauses: bool = DEFAULT_ABSORB_PAUSES,
    absorb_max_frac: float = DEFAULT_ABSORB_MAX_FRAC,
    absorb_vad_backend: str = DEFAULT_ABSORB_VAD_BACKEND,
) -> Dict[str, Any]:
    """Every pipeline kwarg segment mode sets, including the producer.

    `tts_apply_stretch=False` is load-bearing: if the TTS stage stretches
    too, every clip is re-timed twice into the same slot.
    """
    return {
        "timing_producer": "naive_slot",
        "timing_params": segment_timing_params(
            speed_cap=speed_cap, speed_floor=speed_floor,
            slot_margin=slot_margin, gap_barrier=gap_barrier,
            borrow_gap=borrow_gap, borrow_frac=borrow_frac,
            absorb_pauses=absorb_pauses, absorb_max_frac=absorb_max_frac,
            absorb_vad_backend=absorb_vad_backend),
        "tts_apply_stretch": False,
        "mix_strategy": "timing_plan",
        "mix_clip_source": "raw",
    }


#: `--speed-mode warp` defaults. Tighter than segment mode's because this
#: path reports the SPEECH rate rather than a blob rate: past 1.80 the answer
#: is to shorten the text (`--condense-speed-cap`), not push the mixer.
DEFAULT_WARP_STEP = 0.30       # max relative tempo change between adjacent cues
DEFAULT_WARP_TEMPO_MAX = 1.45
DEFAULT_WARP_TEMPO_HARD_MAX = 1.80
DEFAULT_WARP_RAMP_S = 0.18
DEFAULT_WARP_ISLAND_GAP = 1.00
# Keep diarized labels in one island so the planner can smooth across a
# speaker change. True leaves single-cue islands with no neighbour to ramp
# into, which reads as no audible shape at all.
DEFAULT_WARP_BREAK_SPEAKER = False


def warp_timing_kwargs(
    *,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
    step: float = DEFAULT_WARP_STEP,
    tempo_max: float = DEFAULT_WARP_TEMPO_MAX,
    tempo_hard_max: float = DEFAULT_WARP_TEMPO_HARD_MAX,
    # Not `DEFAULT_BORROW_GAP` (0.0): borrowing is opt-in on the segment
    # path, but here the planner borrows as part of the solve, so 0.0 would
    # forbid it outright.
    pause_borrow: float = 0.30,
    island_gap: float = DEFAULT_WARP_ISLAND_GAP,
    ramp_s: float = DEFAULT_WARP_RAMP_S,
    absorb_max_frac: float = DEFAULT_ABSORB_MAX_FRAC,
    vad_backend: str = DEFAULT_ABSORB_VAD_BACKEND,
    warp_backend: str = "auto",
    break_on_speaker: bool = DEFAULT_WARP_BREAK_SPEAKER,
) -> Dict[str, Any]:
    """Every pipeline kwarg `--speed-mode warp` sets.

    `mix_clip_source="warped"` is load-bearing: the plan's Pieces all carry
    `local_tempo = 1.0` because the stretching already happened, so pointing
    the mix at `raw` places unstretched clips at planned positions and the
    dub drifts.

    `tts_apply_stretch=False` for the same reason as segment mode -- the TTS
    stage must not re-time a clip this producer is about to warp.
    """
    return {
        "timing_producer": "warped",
        "timing_params": {
            "max_rel_step": step,
            "tempo_min": speed_floor,
            "tempo_max": tempo_max,
            "tempo_hard_max": tempo_hard_max,
            "pause_borrow_s": pause_borrow,
            "island_gap_s": island_gap,
            "ramp_s": ramp_s,
            "absorb_max_frac": absorb_max_frac,
            "vad_backend": vad_backend,
            "warp_backend": warp_backend,
            "break_on_speaker": break_on_speaker,
        },
        "tts_apply_stretch": False,
        "mix_strategy": "timing_plan",
        "mix_clip_source": "warped",
    }


def validate_warp_bounds(step: float, tempo_max: float,
                         tempo_hard_max: float) -> None:
    """Raise SystemExit on a warp bound that cannot mean what it says."""
    if not 0.0 < step <= 1.0:
        raise SystemExit(
            f"--speed-step must be in (0.0, 1.0] (got {step}); it is the "
            f"largest RELATIVE tempo change allowed between adjacent cues. "
            f"The just-noticeable difference for speech tempo is about 10%, "
            f"so 0.10-0.30 is the useful range.")
    if tempo_max <= 1.0:
        raise SystemExit(
            f"--speed-tempo-max must be > 1.0 (got {tempo_max}).")
    if tempo_hard_max < tempo_max:
        raise SystemExit(
            f"--speed-tempo-hard-max ({tempo_hard_max}) must be >= "
            f"--speed-tempo-max ({tempo_max}); it is the bound a cue may be "
            f"pushed to when nothing else fits, and it is flagged when used.")


def validate_segment_bounds(speed_cap: float, speed_floor: float) -> None:
    """Raise SystemExit on a cap or floor that cannot mean what it says.

    Lives beside the values rather than in the CLI, so the UI gets the same
    refusal if it ever takes these from an operator.
    """
    if speed_cap <= 1.0:
        raise SystemExit(
            f"--speed-cap must be > 1.0 (got {speed_cap}); it is "
            f"the speed-up bound, applied as max_tempo.")
    if not 0.0 < speed_floor <= 1.0:
        raise SystemExit(
            f"--speed-floor must be in (0.0, 1.0] (got "
            f"{speed_floor}); it is the slow-down bound, applied "
            f"as min_tempo.")


def validate_absorb_bounds(
    borrow_gap: float, borrow_frac: float, absorb_max_frac: float,
) -> None:
    """Raise SystemExit on a borrow or absorption bound that cannot hold.

    Separate from `validate_segment_bounds` so the two can be called
    independently.
    """
    if borrow_gap < 0.0:
        raise SystemExit(
            f"--speed-borrow-gap must be >= 0.0 (got {borrow_gap}); it is "
            f"a number of SECONDS a cue may take from the gap after it, "
            f"and 0.0 means take none.")
    if not 0.0 < borrow_frac <= 1.0:
        raise SystemExit(
            f"--speed-borrow-frac must be in (0.0, 1.0] (got {borrow_frac}); "
            f"it is the share of the following gap a cue may take. 1.0 takes "
            f"all of it, which leaves the next cue no cushion.")
    if not 0.0 < absorb_max_frac < 1.0:
        raise SystemExit(
            f"--absorb-max-frac must be in (0.0, 1.0) (got {absorb_max_frac}); "
            f"it is the share of any ONE pause that may be absorbed. 1.0 "
            f"would delete a pause outright, which welds two sentences "
            f"together and reads as breathless even at natural speech rate.")
