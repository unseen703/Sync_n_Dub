"""System-memory checks for the lip-sync stages.

Two halves that must agree. `backend/memguard.py` checks before a render
and between segments; a self-contained patch in
`third_party/MuseTalk/scripts/inference.py` checks before each BATCH,
because that is the only place that can see memory move during a render.

There is deliberately no watchdog process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend import memguard  # noqa: E402

MUSETALK_INFERENCE = REPO / "third_party" / "MuseTalk" / "scripts" / "inference.py"


# --- the measurement -------------------------------------------------------

def test_snapshot_reads_real_memory():
    snap = memguard.memory_snapshot()
    if not snap:
        pytest.skip("no /proc/meminfo on this host")
    assert snap["total_gib"] > 0.5
    assert 0.0 <= snap["used_pct"] <= 100.0
    assert snap["available_gib"] <= snap["total_gib"]


def test_availability_not_free_space():
    """`MemAvailable`, not `MemFree`, on BOTH sides.

    MuseTalk writes every frame to disk as PNG and reads it back, leaving
    ~10 GB of page cache on a 2-minute 1080p segment. Page cache is
    reclaimable, so counting it as used would read a healthy host as full
    and stall every run at the first batch.
    """
    for path in (REPO / "backend" / "memguard.py", MUSETALK_INFERENCE):
        src = path.read_text()
        assert "MemAvailable" in src, f"{path.name} does not read MemAvailable"
        assert 'key == "MemFree"' not in src and "'MemFree'" not in src, (
            f"{path.name} measures MemFree; that ignores reclaimable cache")


def test_breach_is_quiet_on_a_healthy_host():
    assert memguard.breach(limit_pct=100.0) is None


def test_breach_fires_and_names_the_numbers():
    why = memguard.breach(limit_pct=0.001)
    assert why, "a 0.001% limit must register as breached"
    for token in ("%", "GiB", "guard"):
        assert token in why, f"the message should say {token!r}"


def test_zero_disables_it():
    """`DUBBING_MEM_GUARD_PCT=0` is the documented escape hatch."""
    assert memguard.breach(limit_pct=0) is None


# --- the in-render, per-batch check ---------------------------------------

def test_the_batch_check_runs_before_the_batch_work():
    """Before, not after: the allocation that tips the host over happens
    inside the iteration being guarded."""
    src = MUSETALK_INFERENCE.read_text()
    loop = src[src.index("for i, (whisper_batch, latent_batch)"):]
    guard = loop.index("_wait_for_memory")
    work = loop.index("pe(whisper_batch)")
    assert guard < work, "the memory check runs after the batch is processed"


def test_the_batch_check_is_self_contained():
    """MuseTalk runs in its own conda env with its own sys.path, so it
    cannot import `backend.memguard`. A patch that tried would raise
    ImportError at the top of every render."""
    src = MUSETALK_INFERENCE.read_text()
    assert "_wait_for_memory" in src and "_system_mem_used_pct" in src
    assert "from backend" not in src and "import backend" not in src, (
        "the vendored patch imports the repo; that env cannot see it")
    assert "import time" in src, "_wait_for_memory sleeps; time must be imported"


def test_both_halves_read_the_same_env_var():
    """Two thresholds that can disagree is worse than one that is wrong."""
    for path in (REPO / "backend" / "memguard.py", MUSETALK_INFERENCE):
        assert "DUBBING_MEM_GUARD_PCT" in path.read_text(), path.name


def test_the_pause_is_bounded_and_then_raises():
    """Pausing only helps if something ELSE frees memory. When this render
    is the consumer, waiting cannot help, so an unbounded wait would hang a
    GPU job forever instead of failing it."""
    src = MUSETALK_INFERENCE.read_text()
    fn = src[src.index("def _wait_for_memory"):]
    fn = fn[:fn.index("\n@") if "\n@" in fn[:4000] else 4000]
    assert "_MEM_WAIT_S" in fn, "the wait is not bounded"
    assert "raise MemoryError" in fn, "a timed-out wait must fail, not return"


# --- what must NOT be there -----------------------------------------------

def test_there_is_no_watchdog():
    """Removed by request. The adapters shell out with a plain
    `subprocess.run`; nothing polls memory and kills the child."""
    assert not hasattr(memguard, "run_guarded")
    for name in ("musetalk_adapter", "wav2lip_adapter"):
        src = (REPO / "models" / "lipsync" / f"{name}.py").read_text()
        assert "run_guarded" not in src, f"{name} still uses a watchdog"
        assert "subprocess.run(cmd" in src, f"{name} lost its plain run"


# --- the cheap checks around a render -------------------------------------

def test_the_segment_loop_rechecks_between_segments():
    """A batch check covers the segment that is running. Memory already
    gone when the NEXT segment starts is a different case, and starting
    anyway pays for a model load that then has nowhere to go."""
    src = (REPO / "backend" / "lipsync.py").read_text()
    fn = src[src.index("def _run_lipsync_segmented"):
             src.index("def _run_lipsync_sharded")]
    assert "_mem_breach()" in fn


def test_run_lipsync_refuses_an_already_full_host():
    src = (REPO / "backend" / "lipsync.py").read_text()
    head = src[src.index("def run_lipsync("):src.index("def _render_one")]
    assert "not starting lip-sync" in head
