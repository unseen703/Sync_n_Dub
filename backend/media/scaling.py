"""VRAM/RAM-aware defaults for lip-sync batch sizes and segment length.

The pipeline was first tuned on 2 x 16311 MiB Blackwell cards and ~62 GiB
of system RAM, which fixed four constants:

  * MuseTalk `batch_size`              = 8     (15058 MiB peak at 20)
  * Wav2Lip `wav2lip_batch_size`       = 16
  * Wav2Lip `face_det_batch_size`      = 4
  * `backend.lipsync.run_lipsync` `max_segment_s` = 60.0 s

These scale almost linearly with VRAM, so a 24 GiB card should run ~12/24/6
and a 48 GiB card ~24/48/12. The segment length is bounded by BOTH VRAM and
system RAM, because every segment holds decoded frames in memory as well as
GPU latents; a VRAM-rich / RAM-poor box must not race to 300 s and swap.

The baselines live here, every call site that used to carry a literal reads
them from this module instead, and two env vars override the result when a
specific run wants to pin a value:

  * `DUBBING_LIPSYNC_BATCH`     overrides every batch default (MuseTalk +
                                 Wav2Lip + `lipsync_batch_size` UI preset)
  * `DUBBING_LIPSYNC_SEGMENT_S` overrides `max_segment_s`

Deliberately dependency-free beyond `torch` and `psutil` (both already pulled
in by the main env). The helpers never raise -- a missing probe falls back
to the literal 16 GiB-tuned value, so unit tests and CPU-only hosts keep
exactly the numbers they had before this module existed.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# baselines -- the numbers measured on the 16 GiB / ~62 GiB reference host
# ---------------------------------------------------------------------------

BASELINE_VRAM_GIB = 16.0
#: System RAM assumed sufficient for one 60 s segment. Not measured per-run;
#: picked so today's 62 GiB host stays VRAM-bound (RAM arm doesn't bite),
#: and a RAM-starved deploy clamps instead of swapping.
RAM_REF_GIB = 36.0

MUSETALK_BATCH_BASE = 8
MUSETALK_BATCH_CAP = 48

WAV2LIP_BATCH_BASE = 16
WAV2LIP_BATCH_CAP = 64

WAV2LIP_FACE_DET_BASE = 4
WAV2LIP_FACE_DET_CAP = 16

SEGMENT_S_BASE = 60.0
SEGMENT_S_FLOOR = 30.0
SEGMENT_S_CAP = 300.0

ENV_BATCH = "DUBBING_LIPSYNC_BATCH"
ENV_SEGMENT_S = "DUBBING_LIPSYNC_SEGMENT_S"


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def min_visible_vram_gib() -> Optional[float]:
    """Smallest total VRAM across visible CUDA devices, in GiB.

    `min` because a heterogeneous rig has to pace by its weakest card -- the
    scheduler fans jobs out to all of them. Returns None when CUDA is
    unavailable (CI, CPU-only hosts); callers then fall back to the literal
    baseline.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        n = torch.cuda.device_count()
        if n <= 0:
            return None
        totals = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            totals.append(props.total_memory / (1024 ** 3))
        return float(min(totals)) if totals else None
    except Exception as e:                                # noqa: BLE001
        log.debug("scaling: VRAM probe failed: %s", e)
        return None


def total_ram_gib() -> Optional[float]:
    """Total system RAM in GiB, read from psutil then /proc/meminfo.

    Total, not "available" -- "available" fluctuates with the OS page cache
    and would make the segment length jitter between otherwise identical
    runs.
    """
    try:
        import psutil
        return float(psutil.virtual_memory().total) / (1024 ** 3)
    except Exception:                                     # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:                                     # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# scalers
# ---------------------------------------------------------------------------

def _scale_int(base: int, cap: int) -> int:
    """Scale `base` by VRAM ratio, clamp to [1, cap]."""
    vram = min_visible_vram_gib()
    if vram is None:
        return base
    scaled = round(base * vram / BASELINE_VRAM_GIB)
    return max(1, min(cap, int(scaled)))


def musetalk_batch_default() -> int:
    override = os.getenv(ENV_BATCH, "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            log.warning("scaling: ignoring non-integer %s=%r", ENV_BATCH, override)
    return _scale_int(MUSETALK_BATCH_BASE, MUSETALK_BATCH_CAP)


def wav2lip_batch_default() -> int:
    override = os.getenv(ENV_BATCH, "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            log.warning("scaling: ignoring non-integer %s=%r", ENV_BATCH, override)
    return _scale_int(WAV2LIP_BATCH_BASE, WAV2LIP_BATCH_CAP)


def wav2lip_face_det_default() -> int:
    # Not overridden by DUBBING_LIPSYNC_BATCH -- that var is about model
    # throughput; the face detector is a separate stage and the adapter
    # already halves on OOM.
    return _scale_int(WAV2LIP_FACE_DET_BASE, WAV2LIP_FACE_DET_CAP)


def lipsync_segment_s_default() -> float:
    """Seconds per lip-sync chunk, bounded by whichever of VRAM/RAM is tighter.

    `base * min(vram / 16, ram / RAM_REF)`, floored at 30 s so a tiny box
    cannot pay a fixed per-segment overhead on every second of input, capped
    at 300 s so a dropped segment doesn't lose five minutes of work.
    """
    override = os.getenv(ENV_SEGMENT_S, "").strip()
    if override:
        try:
            return max(SEGMENT_S_FLOOR, float(override))
        except ValueError:
            log.warning("scaling: ignoring non-numeric %s=%r",
                        ENV_SEGMENT_S, override)
    vram = min_visible_vram_gib()
    ram = total_ram_gib()
    if vram is None and ram is None:
        return SEGMENT_S_BASE
    arms = []
    if vram is not None:
        arms.append(vram / BASELINE_VRAM_GIB)
    if ram is not None:
        arms.append(ram / RAM_REF_GIB)
    ratio = min(arms)
    scaled = SEGMENT_S_BASE * ratio
    return float(max(SEGMENT_S_FLOOR, min(SEGMENT_S_CAP, scaled)))
