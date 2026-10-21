"""System-wide memory checks for the lip-sync stages.

Both vendored models load every frame of their input into RAM --
MuseTalk's `read_imgs()`, Wav2Lip's `full_frames.append(frame)` -- so
resident memory is `frames x width x height x 3` and grows without bound
in clip length. Measured on this host: MuseTalk is 5.87 GiB + 6.14
MiB/frame at 1080p, putting a 7-minute clip at ~70 GiB against 62 GiB of
system RAM.

Segmenting bounds that in the normal case. These checks are the backstop
for what it does not cover: a source that slipped past the resolution cap,
a neighbour's job growing underneath ours, a wrong duration probe.

**`MemAvailable`, not `MemFree`.** Page cache is reclaimable, and these
renders generate a lot of it -- MuseTalk writes every frame to disk as PNG
before reading it back, ~10 GB for a 2-minute 1080p segment. Measuring
with `MemFree` would read a healthy host as nearly full and refuse good
runs. `MemAvailable` is the kernel's own estimate of what a new allocation
could actually get, which is the question being asked.

The in-render, per-batch check is NOT here: MuseTalk runs in its own conda
env with its own `sys.path`, so it cannot import this module. That check
is a self-contained patch in `third_party/MuseTalk/scripts/inference.py`
and reads `/proc/meminfo` the same way. The two must agree on
`DUBBING_MEM_GUARD_PCT`, which is why both read the same variable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

#: Treat the host as out of memory at this share in use. 95 by request
#: (2026-10-20). Override with `DUBBING_MEM_GUARD_PCT`; 0 disables.
#: `third_party/MuseTalk/scripts/inference.py` reads the same variable.
DEFAULT_LIMIT_PCT = float(os.getenv("DUBBING_MEM_GUARD_PCT", "95") or 95)

_MEMINFO = Path("/proc/meminfo")


def memory_snapshot() -> dict:
    """`{total_gib, available_gib, used_pct}`, or an empty dict.

    Returns {} rather than raising on a host without /proc/meminfo: a
    memory check must never itself be the reason a render fails.
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
