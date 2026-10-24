"""The `--speed-mode segment` knobs, in one place.

Two callers set the same four pipeline arguments and used to do it from two
hand-maintained copies: `backend.cli.resolve_speed_setup` (from an argparse
namespace) and `ui.presets.UI_TIMING_DEFAULTS` (from nothing — a literal
dict). The copy was deliberate and documented, because the resolver needs a
namespace the UI path does not have; `tests/test_speed_mode.py` existed to
catch the two drifting apart.

This is the shared half: the values and the reasoning, parameterised by the
four bounds that have flags. The CLI passes what the operator typed, the UI
passes nothing and gets the defaults — which are the argparse defaults, so
"what the UI does" and "what a bare CLI run does" are the same thing by
construction rather than by a test.

Deliberately dependency-free. `ui/presets.py` is imported when the web server
starts and pulls in nothing heavier than `os` and `typing`; routing it
through `backend.cli` would drag argparse and all seven model registries into
that import.

The defaults mirror `backend/cli.py`'s `--speed-*` arguments. They are
repeated as keyword defaults here rather than read back out of the parser
because building a parser to ask it a question is the heavier coupling, and
`tests/test_speed_mode.py` pins the two together.
"""
from __future__ import annotations

from typing import Any, Dict

#: Default speed-UP bound (`--speed-cap`). Raised from 1.5 on 2026-10-22:
#: Hindi TTS runs 1.21–1.65x longer than the English it covers, so 1.5 sat on
#: top of that range with no headroom — job 20261021-114009 overran its slot
#: on 4 of 5 segments at ratios 1.22 / 1.17 / 1.31.
DEFAULT_SPEED_CAP = 1.8

#: Default slow-DOWN bound (`--speed-floor`). NOT 1/1.8. A symmetric floor
#: would slow a short line to 0.56x — nearly double length — to fill its
#: slot, which is audible as drawl. What this floor cannot absorb stays a
#: real pause and is reported as `unfilled_slot_s` rather than hidden in the
#: clip.
DEFAULT_SPEED_FLOOR = 0.75

#: Any margin is itself a pause, so the default is none.
DEFAULT_SLOT_MARGIN = 0.0

#: A source gap at least this long is a wall the shift-forward cursor may not
#: cross (`--speed-gap-barrier`).
DEFAULT_GAP_BARRIER = 1.0


def segment_timing_params(
    *,
    speed_cap: float = DEFAULT_SPEED_CAP,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
    slot_margin: float = DEFAULT_SLOT_MARGIN,
    gap_barrier: float = DEFAULT_GAP_BARRIER,
) -> Dict[str, Any]:
    """`timing_params` for the `naive_slot` producer in segment mode."""
    return {
        "max_tempo": speed_cap,
        "min_tempo": speed_floor,
        "safety_margin_s": slot_margin,
        # Simultaneous speech stays simultaneous, and the cascade is not
        # allowed to walk a line out over the music. Both are corrections
        # rather than preferences, so neither has a flag of its own; the
        # gap threshold does, because what counts as "a gap" is material.
        "overlap_policy": "keep",
        "gap_barrier_s": gap_barrier,
        # Honour a per-segment target written by stage 9c or by
        # `backend.edits.set_speed`; without this they are silently
        # ignored in this mode.
        "use_target_duration": True,
    }


def segment_timing_kwargs(
    *,
    speed_cap: float = DEFAULT_SPEED_CAP,
    speed_floor: float = DEFAULT_SPEED_FLOOR,
    slot_margin: float = DEFAULT_SLOT_MARGIN,
    gap_barrier: float = DEFAULT_GAP_BARRIER,
) -> Dict[str, Any]:
    """Every pipeline kwarg segment mode sets, including the producer.

    `tts_apply_stretch=False` is load-bearing: the TTS stage must NOT stretch
    as well, or every clip is re-timed twice — once into its slot here, once
    into the same slot there.
    """
    return {
        "timing_producer": "naive_slot",
        "timing_params": segment_timing_params(
            speed_cap=speed_cap, speed_floor=speed_floor,
            slot_margin=slot_margin, gap_barrier=gap_barrier),
        "tts_apply_stretch": False,
        "mix_strategy": "timing_plan",
        "mix_clip_source": "raw",
    }


def validate_segment_bounds(speed_cap: float, speed_floor: float) -> None:
    """Raise SystemExit on a cap or floor that cannot mean what it says.

    Lives beside the values rather than in the CLI so the UI gets the same
    refusal if it ever starts taking these from an operator.
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
