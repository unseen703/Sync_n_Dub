"""What the UI says about each lip-sync model, in one place.

`presets.LIPSYNC_FACTORS`, `cost.LIPSYNC_RATES` and `spec.LIPSYNC_WORDS` all
derive from `FACTS`, and `options()` iterates the registry rather than a
hand-written list, so a newly registered adapter cannot be missing from the UI.

Two rules are load-bearing. A missing `rate` must read as "not measured",
never as a fast model, which is what `estimate_runtime`'s `lipsync_measured`
flag is for. And `normalise()` matches the longest key FIRST, since a shorter
id can be a prefix of a longer one and an unordered match silently downgrades
one to the other.
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
        "note": (""),
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
        "note": (""),
    },
}

#: Display-only, and currently empty. The mechanism is how a job folder
#: naming a removed model keeps a real label, so a removal with jobs behind
#: it should mark the row retired rather than delete it.
RETIRED: frozenset = frozenset(
    k for k, v in FACTS.items() if v.get("retired"))

#: Longest first. No key is currently a prefix of another, but `latentsync`
#: would be of `latentsync15`, so a re-added row relies on this ordering.
_KEYS_BY_LENGTH: Tuple[str, ...] = tuple(
    sorted(FACTS, key=len, reverse=True))


def normalise(name: Optional[str]) -> Optional[str]:
    """`musetalk-v15` -> `musetalk`, `latentsync15-x` -> `latentsync15`.

    The clip runner records variants, so label lookups and registry calls need
    the base id. Matches the longest key first, which keeps a longer id from
    resolving to a shorter one it happens to start with. Returns None for an
    empty or unknown value.
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
    return (3,5)


def speed_text(name: Optional[str]) -> str:
    """"3–5× clip length"""
    band = rate_band(name)

    lo, hi = 3,5 
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
            "speed": speed_text(m),
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
