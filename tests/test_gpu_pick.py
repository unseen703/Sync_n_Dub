"""Device selection: use every FREE gpu, refuse when none is.

Policy set 2026-10-19 by request:
  both cards free -> use both;  one free -> use it;  none free -> refuse.

"Free" means no OTHER process on the card -- deliberately not a free-VRAM
threshold. A threshold was tried and removed: our own pipeline holds a CUDA
context on whichever device it touched earlier (measured 1.9-4.7 GiB for
`ui.runner` after face detection), so a memory floor reads the job's own
context as a foreign tenant and refuses the card it is already using.

Refusing returns [] and the caller must stop. [] must never be read as "use
the default device": an unpinned worker lands on whichever card the driver
chooses, which is the collision this exists to avoid.
"""
from __future__ import annotations

import pytest

from backend import gpu as G


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DUBBING_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_both_free_uses_both(monkeypatch):
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "gpu_busy_pids", lambda: {})
    assert G.pick_devices(None) == [0, 1]


def test_one_free_uses_that_one(monkeypatch):
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "gpu_busy_pids", lambda: {0: [4242]})
    assert G.pick_devices(None) == [1]


def test_none_free_refuses(monkeypatch):
    """[] is a refusal. The caller stops and tells the operator."""
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "gpu_busy_pids", lambda: {0: [1], 1: [2]})
    assert G.pick_devices(None) == []
    assert G.pick_devices(1) == []


def test_n_caps_the_count(monkeypatch):
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "gpu_busy_pids", lambda: {})
    assert G.pick_devices(1) == [0]


def test_no_gpus_returns_empty(monkeypatch):
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [])
    assert G.pick_devices(None) == []


def test_our_own_context_does_not_make_a_card_look_busy(monkeypatch):
    """The reason the VRAM floor was removed.

    `run_lipsync` executes inside the process that already loaded face
    detection, so that process holds a context on one of the cards.
    """
    import os
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "_own_pids", lambda: {os.getpid()})
    import subprocess as sp
    outs = iter([
        sp.CompletedProcess([], 0, stdout=f"GPU-aaa, {os.getpid()}\n", stderr=""),
        sp.CompletedProcess([], 0, stdout="0, GPU-aaa\n1, GPU-bbb\n", stderr=""),
    ])
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: next(outs))
    assert G.gpu_busy_pids() == {}, "our own pid must not count as a tenant"


def test_a_foreign_pid_does_make_a_card_busy(monkeypatch):
    import os
    import subprocess as sp
    monkeypatch.setattr(G, "_own_pids", lambda: {os.getpid()})
    outs = iter([
        sp.CompletedProcess([], 0, stdout="GPU-aaa, 999999\n", stderr=""),
        sp.CompletedProcess([], 0, stdout="0, GPU-aaa\n1, GPU-bbb\n", stderr=""),
    ])
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: next(outs))
    assert G.gpu_busy_pids() == {0: [999999]}


def test_nvidia_smi_failure_leaves_cards_usable(monkeypatch):
    """No information must not become a refusal."""
    import subprocess as sp
    monkeypatch.setattr(G.subprocess, "run",
                        lambda *a, **k: sp.CompletedProcess([], 1, "", "boom"))
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    assert G.gpu_busy_pids() == {}
    assert G.pick_devices(None) == [0, 1]


def test_free_memory_is_remapped_under_cuda_visible_devices(monkeypatch):
    """Kept for reporting. nvidia-smi gives PHYSICAL ids; with
    CUDA_VISIBLE_DEVICES=2,5 the process sees 0 and 1."""
    import subprocess as sp
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(
        G.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(
            [], 0, stdout="0, 100\n1, 200\n2, 7000\n5, 9000\n", stderr=""))
    assert G.gpu_free_memory() == {0: 7000, 1: 9000}


# --- the refusal reaches the operator ------------------------------------

def test_preflight_refuses_when_all_cards_are_taken(monkeypatch):
    from ui import runner as R
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "free_devices", lambda: [])
    monkeypatch.setattr(G, "gpu_busy_pids", lambda: {0: [11], 1: [22]})
    with pytest.raises(RuntimeError, match="no GPU is free"):
        R._preflight_gpu()


def test_preflight_passes_with_one_free_card(monkeypatch):
    from ui import runner as R
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(G, "free_devices", lambda: [1])
    R._preflight_gpu()


def test_preflight_is_silent_on_a_cpu_only_host(monkeypatch):
    from ui import runner as R
    monkeypatch.setattr(G, "visible_cuda_devices", lambda: [])
    R._preflight_gpu()


def test_preflight_runs_before_the_download():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "ui" / "runner.py").read_text()
    assert src.index("_preflight_gpu()") < src.index('writer.begin("fetch"')


def test_the_refusal_is_a_recognised_ui_error():
    from ui.errors import explain
    r = explain("no GPU is free — gpu0 busy (pid 123). Lip-sync needs a card "
                "to itself; waiting is safer than sharing one.")
    assert r["recognised"] is True
    assert "busy" in r["title"].lower()
