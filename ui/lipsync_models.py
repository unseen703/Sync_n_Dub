"""What the UI says about each lip-sync model, in ONE place.

Four tables used to describe the same five models and none of them knew
about the others: `presets.LIPSYNC_FACTORS` (the runtime estimate beside the
radio button), `cost.LIPSYNC_RATES` (the per-stage cost table),
`spec.LIPSYNC_WORDS` (the job summary line) and a hand-written three-row
list in `server.py` (`/api/options`). `wav2lip_hd` was registered in
`models/lipsync/__init__.py` on 2026-10-19 and reached NONE of them, so it
was selectable — `GET .../lipsync/readiness` serves `available_models()`
straight from the registry — while every number and label next to it was
either absent or wrong.

Two faults are worth recording, because both were silent:

1. **A missing rate reads as a fast model, not as a missing rate.**
   `presets.estimate_runtime` adds nothing for a model it does not know and
   returns the dub-only band as an ordinary "about X to Y". So the one model
   here that runs a second per-frame GAN pass advertised as the cheapest
   option on the list. `ui/cost.py` got this right already (it warns by
   name); `presets` did not. `estimate_runtime` now reports
   `lipsync_measured`, and the panel says so rather than quoting a number
   nobody measured.

2. **`"wav2lip_hd".startswith("wav2lip")` is True**, and `wav2lip` came
   first in all three prefix loops (`spec.py`, `server.py::_job_config`,
   `edit_agent._regex_ops`). An HD run therefore displayed as "On —
   Wav2Lip", `_job_config` REWROTE `cfg["lipsync"]` to `"wav2lip"`, and the
   agent's regex fallback parsed "lip-sync it with wav2lip_hd" into the
   plain model — a silent downgrade on the path that runs when no
   `OPENROUTER_API_KEY` is set, i.e. when it is the only parser.
   `normalise()` is the one implementation now, longest key first.

The table is keyed on the REGISTRY, not on a plausible list: `options()`
iterates `available_models()`, and `unregistered()` reports either kind of
drift so `tests/test_ui_lipsync_models.py` fails on a model added to one
side and not the other. That is the tone-picker lesson from CLAUDE.md §2 —
`ui/speakers.py` once shipped an invented list of styles the engine 422s on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.lipsync import available_models


class _Facts(Dict[str, Any]):
    pass


#: Per-model presentation and measured cost.
#:
#: `rate` is wall time as a multiple of clip length for stage 9h alone, on
#: this host's 2x16 GiB Blackwell cards. `None` means NOT MEASURED HERE, and
#: is propagated as "no estimate" rather than quietly omitted from a sum.
FACTS: Dict[str, Dict[str, Any]] = {
    "wav2lip": {
        "label": "Wav2Lip",
        "rate": (3.0, 5.0),
        "note": ("Output is capped at 1080p: a source at or below that "
                 "renders at its own resolution, anything above is halved."),
    },
    "wav2lip_hd": {
        "label": "Wav2Lip-HD",
        # RETIRED 2026-10-23, kept for DISPLAY ONLY. Its `sync()` was
        # `super().sync()` plus the GFPGAN pass, so stage 1 was byte-identical
        # to `wav2lip` -- same model, same wav2lip_gan.pth at img_size=96,
        # same resize policy, same coord cache. With restoration now its own
        # stage it is exactly `--lipsync wav2lip --restore gfpgan`.
        #
        # The row stays because `normalise()` matches longest key first, and
        # `"wav2lip_hd".startswith("wav2lip")` is True: delete it and every
        # one of the 3 HD jobs on disk silently relabels itself "Wav2Lip",
        # which is the exact failure CLAUDE.md S4 records for this pair.
        "retired": True,
        "retired_note": "Use --lipsync wav2lip --restore gfpgan.",
        # Measured 2026-10-19 from the three runs in
        # `data/gender_bench/cuts/hd_runs/logs/` -- stage 9h start to the
        # conform step, --lipsync-num-gpus 2: 40s clip 246s (6.1x), 40s clip
        # 550s (13.7x), 100s clip 2292s (22.9x). The spread is real and is
        # not noise: stage 2 is per-PIXEL, so a 1080p source costs several
        # times a 540p one at the same frame count. Banded wide on purpose.
        "rate": (6.0, 24.0),
        "note": ("Wav2Lip, then Real-ESRGAN + GFPGAN over every frame, "
                 "restored to the 1080p cap. A second pass, not a cheaper "
                 "one — cost scales with frame size, not just length."),
    },
    "musetalk": {
        "label": "MuseTalk",
        "rate": (8.0, 21.0),
        "note": "Composites at full resolution.",
    },
    "latentsync": {
        "label": "LatentSync",
        "rate": (75.0, 90.0),         # measured "~83x"
        # Stated rather than discovered. `latentsync_adapter.py` defaults to
        # `configs/unet/stage2_512.yaml`, and CLAUDE.md §1 measures 512x512
        # at ~18 GiB against a 16311 MiB card. An OOM here surfaces as
        # "produced no mp4", so the warning belongs in front of the click.
        "note": ("Highest fidelity, and the slowest by far. Defaults to the "
                 "512x512 UNet, measured at ~18 GiB — above this host's "
                 "16 GiB cards. Set LATENTSYNC_UNET_CONFIG to "
                 "configs/unet/stage2.yaml for the 256 variant."),
    },
    "latentsync15": {
        "label": "LatentSync 1.5",
        # NOT measured on this host. 1.6 at 512 ran 72-83x clip length; 1.5
        # is the same code at 256, so it is certainly faster, but "certainly
        # faster" is not a number and a guessed band in a consent dialog is
        # worse than a blank -- the user plans around it. `options()` renders
        # this as "not timed on this host".
        "rate": None,
        "note": ("Same model at 256x256, ~8 GB — the one LatentSync that "
                 "fits these 16 GiB cards with room to spare. Blurrier "
                 "teeth and lips than 1.6; that is why 1.6 exists. Needs "
                 "its own checkpoint (ByteDance/LatentSync-1.5)."),
    },
    "lipgan": {
        "label": "LipGAN",
        "rate": (5.0, 15.0),          # torch port; least characterised
        "note": "2019 baseline, in-env PyTorch port.",
        # RETIRED 2026-10-23. Display only, for jobs recorded before then.
        "retired": True,
        "retired_note": "Removed; use wav2lip or musetalk.",
    },
}

#: Registered once, now display-only. They are deliberately still in `FACTS`
#: and deliberately NOT in `models/lipsync`: a job folder on disk names the
#: model it ran, and a reader that cannot resolve the name either shows a
#: bare slug or -- worse, for `wav2lip_hd` -- silently resolves it to a
#: DIFFERENT model via the prefix match. `options()` iterates the registry,
#: so nothing retired is ever offered.
RETIRED: frozenset = frozenset(
    k for k, v in FACTS.items() if v.get("retired"))

#: Longest first, so `wav2lip_hd` is tested before `wav2lip`.
_KEYS_BY_LENGTH: Tuple[str, ...] = tuple(
    sorted(FACTS, key=len, reverse=True))


def normalise(name: Optional[str]) -> Optional[str]:
    """`musetalk-v15` -> `musetalk`, `wav2lip-hd` -> `wav2lip_hd`.

    The clip runner records a variant (`musetalk-v15`), so label lookups and
    registry calls need the base id. Returns None for an empty or unknown
    value — the caller decides whether that means "off" or "unknown", and
    those are different answers.
    """
    s = (name or "").strip().lower().replace("-", "_")
    if not s:
        return None
    for key in _KEYS_BY_LENGTH:
        if s == key or s.startswith(key):
            return key
    return None


def facts(name: Optional[str]) -> Dict[str, Any]:
    """Presentation + cost for a model id, or `{}` if it is not one of ours."""
    key = normalise(name)
    return dict(FACTS[key]) if key else {}


def rate_band(name: Optional[str]) -> Optional[Tuple[float, float]]:
    """`(low, high)` multiples of clip length, or None when unmeasured."""
    return facts(name).get("rate")


def label(name: Optional[str]) -> Optional[str]:
    return facts(name).get("label")


def speed_text(name: Optional[str]) -> str:
    """"3–5× clip length", or an explicit admission that nobody timed it."""
    band = rate_band(name)
    if not band:
        return "not timed on this host"
    lo, hi = band
    return f"{lo:g}–{hi:g}× clip length"


def options() -> List[Dict[str, Any]]:
    """One row per REGISTERED model, for `/api/options` and readiness.

    Driven by `available_models()` so a newly registered adapter appears
    here the moment it is registered. A model with no `FACTS` entry is still
    listed — hiding it would make it unselectable through the UI while the
    POST happily accepts it — but it is listed honestly, with its own id as
    the label and no speed claim.
    """
    rows = []
    for m in available_models():
        f = FACTS.get(m, {})
        rows.append({
            "id": m,
            "label": f.get("label") or m,
            "speed": speed_text(m) if f else "not timed on this host",
            "note": f.get("note"),
            "measured": bool(f.get("rate")),
        })
    return rows


def unregistered() -> Dict[str, List[str]]:
    """Drift in both directions, for the guard test.

    `missing_facts` is a model the registry offers that this module cannot
    describe — it will render as a bare slug with no estimate.
    `stale_facts` is the opposite: a row here for an adapter that no longer
    exists, which would advertise a model the POST 422s on.

    `RETIRED` rows are exempt from `stale_facts` BY DESIGN. They describe
    models that ran before and are named in job folders on disk; they are
    never offered, because `options()` iterates the registry rather than
    this table. Treating them as drift would force a choice between a
    failing guard test and losing the labels for finished work.
    """
    registered = set(available_models())
    return {
        "missing_facts": sorted(registered - set(FACTS)),
        "stale_facts": sorted(set(FACTS) - registered - RETIRED),
    }


__all__ = ["FACTS", "normalise", "facts", "rate_band", "label",
           "speed_text", "options", "unregistered"]
