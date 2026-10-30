"""System-wide memory checks for the lip-sync stages.

The vendored models load every frame into RAM, so resident memory is
`frames x width x height x 3` and grows without bound in clip length — a
few minutes of 1080p exceeds this host's 62 GiB. Segmenting bounds that
normally; these checks are the backstop for what it does not cover.

Reads `MemAvailable`, not `MemFree`: these renders generate gigabytes of
reclaimable page cache, so `MemFree` would read a healthy host as full.

The in-render per-batch check lives in the vendored inference script, which
runs in its own env and cannot import this. Both read
`DUBBING_MEM_GUARD_PCT` so the two agree.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

#: Treat the host as out of memory at this share in use. Override with
#: `DUBBING_MEM_GUARD_PCT`; 0 disables. The vendored inference script reads
#: the same variable.
DEFAULT_LIMIT_PCT = float(os.getenv("DUBBING_MEM_GUARD_PCT", "95") or 95)

_MEMINFO = Path("/proc/meminfo")


def memory_snapshot() -> dict:
    """`{total_gib, available_gib, used_pct}`, or an empty dict.

    Returns {} rather than raising without /proc/meminfo — a memory check must
    never itself be why a render fails.
    """
    try:
        fields = {}
        for line in _MEMINFO.read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                fields[key] = float(rest.strip().split()[0])  # kB
            if len(fields) == 2:
                break
        total, avail = fields["MemTotal"], fields["MemAvailable"]
        if total <= 0:
            return {}
        return {"total_gib": total / 1048576.0,
                "available_gib": avail / 1048576.0,
                "used_pct": 100.0 * (total - avail) / total}
    except Exception:                                    # noqa: BLE001
        return {}


def used_pct() -> Optional[float]:
    return memory_snapshot().get("used_pct")


def breach(limit_pct: Optional[float] = None) -> Optional[str]:
    """A human sentence if memory is already past the limit, else None."""
    limit = DEFAULT_LIMIT_PCT if limit_pct is None else limit_pct
    if limit <= 0:
        return None
    snap = memory_snapshot()
    pct = snap.get("used_pct")
    if pct is None or pct < limit:
        return None
    return (f"system memory is {pct:.1f}% used "
            f"({snap['available_gib']:.1f} GiB free of "
            f"{snap['total_gib']:.1f} GiB), at or past the {limit:.0f}% "
            f"guard")
