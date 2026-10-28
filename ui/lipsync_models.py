"""What the UI says about each lip-sync model, in one place.

`presets.LIPSYNC_FACTORS`, `cost.LIPSYNC_RATES` and `spec.LIPSYNC_WORDS` all
derive from the `FACTS` table below, and `options()` iterates the registry
(`models.lipsync.available_models()`) rather than a hand-written list, so a
newly registered adapter cannot be missing from the UI.

Two things here are load-bearing:

1. A missing `rate` must read as "not measured", never as a fast model --
   `estimate_runtime` reports `lipsync_measured` so the panel can say so.
2. `normalise()` matches the longest key first, because
   `"wav2lip_hd".startswith("wav2lip")` is True and a prefix match silently
   downgrades an HD job to the plain model.

`unregistered()` reports drift in both directions for
`tests/test_ui_lipsync_models.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models.lipsync import available_models


class _Facts(Dict[str, Any]):
    pass


#: Per-model presentation and measured cost.
#:
#: `rate` is stage-9h wall time as a multiple of clip length on this host's
#: 2x16 GiB Blackwell cards. `None` means not measured, and is propagated as
#: "no estimate" rather than omitted from a sum.
FACTS: Dict[str, Dict[str, Any]] = {
    "wav2lip": {
        "label": "Wav2Lip",
        "rate": (3.0, 5.0),
        "note": ("Output is capped at 1080p: a source at or below that "
                 "renders at its own resolution, anything above is halved."),
    },
    "wav2lip_hd": {
        "label": "Wav2Lip-HD",
        # RETIRED 2026-10-23, display only. It was `wav2lip` plus a GFPGAN
        # pass, which is now `--lipsync wav2lip --restore gfpgan`.
        #
        # Do not delete this row: `normalise()` matches longest key first,
        # and `"wav2lip_hd".startswith("wav2lip")` is True, so without it
        # every HD job on disk relabels itself "Wav2Lip".
        "retired": True,
        "retired_note": "Use --lipsync wav2lip --restore gfpgan.",
        # Measured 2026-10-19 over three runs: 6.1x, 13.7x, 22.9x. The
        # spread is real -- stage 2 is per-pixel, so a 1080p source costs
        # several times a 540p one at the same frame count.
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
        # Defaults to `configs/unet/stage2_512.yaml`, which needs ~18 GiB
        # against a 16311 MiB card. An OOM surfaces as "produced no mp4",
        # so the warning goes in front of the click.
        "note": ("Highest fidelity, and the slowest by far. Defaults to the "
                 "512x512 UNet, measured at ~18 GiB — above this host's "
                 "16 GiB cards. Set LATENTSYNC_UNET_CONFIG to "
                 "configs/unet/stage2.yaml for the 256 variant."),
    },
    "latentsync15": {
        "label": "LatentSync 1.5",
        # Not timed on this host. 1.6 at 512 ran 72-83x; 1.5 is the same
        # code at 256, so faster, but no measured band exists. `options()`
        # renders this as "not timed on this host".
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

#: Registered once, now display-only. Kept in `FACTS` so a job folder naming
#: one still renders a real label; absent from `models/lipsync` so nothing
#: retired is ever offered.
RETIRED: frozenset = frozenset(
    k for k, v in FACTS.items() if v.get("retired"))

#: Longest first, so `wav2lip_hd` is tested before `wav2lip`.
_KEYS_BY_LENGTH: Tuple[str, ...] = tuple(
    sorted(FACTS, key=len, reverse=True))


def normalise(name: Optional[str]) -> Optional[str]:
    """`musetalk-v15` -> `musetalk`, `wav2lip-hd` -> `wav2lip_hd`.

    The clip runner records variants, so label lookups and registry calls
    need the base id. Matches longest key first, so `wav2lip_hd` does not
    resolve to `wav2lip`. Returns None for an empty or unknown value.
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
    """One row per registered model, for `/api/options` and readiness.

    Driven by `available_models()`, so a newly registered adapter appears
    here at once. A model with no `FACTS` entry is still listed, with its id
    as the label and no speed claim.
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

    `missing_facts`: a registered model this table cannot describe (renders
    as a bare slug with no estimate). `stale_facts`: a row here for an
    adapter that no longer exists.

    `RETIRED` rows are exempt from `stale_facts` -- they label finished jobs
    on disk and are never offered, since `options()` iterates the registry.
    """
    registered = set(available_models())
    return {
        "missing_facts": sorted(registered - set(FACTS)),
        "stale_facts": sorted(set(FACTS) - registered - RETIRED),
    }


__all__ = ["FACTS", "normalise", "facts", "rate_band", "label",
           "speed_text", "options", "unregistered"]
