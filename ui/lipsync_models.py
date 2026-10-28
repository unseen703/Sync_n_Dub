"""What the UI says about each lip-sync model, in one place.

`presets.LIPSYNC_FACTORS`, `cost.LIPSYNC_RATES` and `spec.LIPSYNC_WORDS` all
derive from the `FACTS` table below, and `options()` iterates the registry
(`models.lipsync.available_models()`) rather than a hand-written list, so a
newly registered adapter cannot be missing from the UI.

Two things here are load-bearing:

1. A missing `rate` must read as "not measured", never as a fast model --
   `estimate_runtime` reports `lipsync_measured` so the panel can say so.
2. `normalise()` matches the longest key first, because a shorter id can be
   a prefix of a longer one (`wav2lip` of `wav2lip_hd`, `latentsync` of
   `latentsync15`) and an unordered match silently downgrades one to the
   other.

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
    "musetalk": {
        "label": "MuseTalk",
        "rate": (8.0, 21.0),
        "note": "Composites at full resolution.",
    },
    "latentsync15": {
        "label": "LatentSync 1.5",
        # Not timed on this host. The removed 1.6 at 512 ran 72-83x; this
        # is the same code at 256, so faster, but no measured band exists.
        # `options()` renders this as "not timed on this host".
        "rate": None,
        "note": ("Audio-conditioned latent diffusion at 256x256, ~8 GB. "
                 "Highest fidelity of the three and the slowest by far. "
                 "Needs its own checkpoint (ByteDance/LatentSync-1.5)."),
    },
}

#: Registered once, now display-only. EMPTY since 2026-10-28, when
#: `wav2lip_hd`, `lipgan` and `latentsync` (1.6) were dropped from `FACTS`
#: outright rather than kept as retired rows. The mechanism stays: it is how
#: a job folder naming a removed model keeps a real label, and the next
#: removal that has jobs behind it should use it.
RETIRED: frozenset = frozenset(
    k for k, v in FACTS.items() if v.get("retired"))

#: Longest first. No key is currently a prefix of another, but `wav2lip_hd`
#: was (of `wav2lip`) and `latentsync` is (of `latentsync15`) -- so this
#: ordering is what a re-added row would rely on.
_KEYS_BY_LENGTH: Tuple[str, ...] = tuple(
    sorted(FACTS, key=len, reverse=True))


def normalise(name: Optional[str]) -> Optional[str]:
    """`musetalk-v15` -> `musetalk`, `latentsync15-x` -> `latentsync15`.

    The clip runner records variants, so label lookups and registry calls
    need the base id. Matches longest key first, which is what keeps a
    longer id from resolving to a shorter one it happens to start with.
    Returns None for an empty or unknown value.

    Since the three retired rows were dropped on 2026-10-28, `wav2lip_hd`
    resolves here to `wav2lip` and `latentsync` to None. The three
    `hd_*` job folders on disk are the former case.
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
