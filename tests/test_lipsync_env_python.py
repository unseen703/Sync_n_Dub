"""MuseTalk and LatentSync must find their own interpreter, from any launcher.

Both have upstream pins incompatible with the `dubbing` env, which is why they
have separate conda envs. The adapters used to resolve
`<MODEL>_PYTHON or sys.executable`, and an empty value is the NORMAL case:
neither variable was in `.env.example`, and `LIPGAN_PYTHON` / `WAV2LIP_PYTHON`
are legitimately blank (those two do run in `dubbing`), so a blank looks safe.

Measured 2026-10-17: a full 40 s CLI run executed all 15 stages, then stage 9h
shelled MuseTalk into `dubbing`, `import diffusers` raised, and the adapter
logged `lipsync[musetalk]: done -> <path>` with `rc=1` in a parenthetical. Exit
code 0, manifest recording a render that did not exist.

Resolution happens in the adapter rather than in each launcher, so the CLI, the
UI runner (`scripts/ui_run_job.sh`), the edit agent and `experiment.py` are all
covered by one change.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from models.lipsync.base import resolve_env_python


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("MUSETALK_PYTHON", "/from/env/python")
    assert resolve_env_python("musetalk", "musetalk", "/explicit/python") \
        == "/explicit/python"


def test_env_var_beats_the_sibling_env(monkeypatch):
    """An operator pointing at a custom build must not be second-guessed."""
    monkeypatch.setenv("MUSETALK_PYTHON", "/custom/python")
    assert resolve_env_python("musetalk", "musetalk") == "/custom/python"


def test_blank_env_var_is_treated_as_unset(monkeypatch):
    """`.env` ships `LIPGAN_PYTHON=` style blanks; a blank must not win."""
    monkeypatch.setenv("MUSETALK_PYTHON", "   ")
    got = resolve_env_python("musetalk", "musetalk")
    assert got.strip(), "a blank value must fall through, not be returned"
    assert got != "   "


def test_sibling_conda_env_is_found_without_any_variable(monkeypatch, tmp_path):
    """The fix: `<prefix>/envs/<name>/bin/python` beside the running one."""
    envs = tmp_path / "envs"
    (envs / "dubbing" / "bin").mkdir(parents=True)
    (envs / "musetalk" / "bin").mkdir(parents=True)
    running = envs / "dubbing" / "bin" / "python"
    running.write_text("#!/bin/sh\n")
    sibling = envs / "musetalk" / "bin" / "python"
    sibling.write_text("#!/bin/sh\n")

    monkeypatch.delenv("MUSETALK_PYTHON", raising=False)
    monkeypatch.setattr(sys, "executable", str(running))
    assert resolve_env_python("musetalk", "musetalk") == str(sibling)


def test_no_sibling_falls_back_and_warns(monkeypatch, tmp_path, caplog):
    """Silence here is what made the original failure cost a whole run."""
    envs = tmp_path / "envs"
    (envs / "dubbing" / "bin").mkdir(parents=True)
    running = envs / "dubbing" / "bin" / "python"
    running.write_text("#!/bin/sh\n")

    monkeypatch.delenv("MUSETALK_PYTHON", raising=False)
    monkeypatch.setattr(sys, "executable", str(running))
    with caplog.at_level("WARNING"):
        got = resolve_env_python("musetalk", "musetalk")
    assert got == str(running)
    assert any("MUSETALK_PYTHON" in r.getMessage() for r in caplog.records)


# --- the two adapters actually use it -------------------------------------

@pytest.mark.parametrize("module,cls_name,env_name", [
    ("musetalk", "MuseTalkSync", "musetalk"),
    ("latentsync", "LatentSyncSync", "latentsync"),
])
def test_adapter_resolves_without_the_env_var(monkeypatch, module, cls_name, env_name):
    import importlib
    monkeypatch.delenv(f"{module.upper()}_PYTHON", raising=False)
    mod = importlib.import_module(f"models.lipsync.{module}_adapter")
    adapter = getattr(mod, cls_name)()
    assert adapter.python_bin, "no interpreter resolved at all"
    # On this host the sibling env exists, so it must NOT be `dubbing`.
    sibling = Path(sys.executable).resolve().parents[2] / env_name / "bin" / "python"
    if sibling.is_file():
        assert Path(adapter.python_bin) == sibling, (
            f"{module} fell back to the dubbing env while {sibling} exists")


def test_wav2lip_is_deliberately_not_changed():
    """Wav2Lip DOES run in `dubbing`; rerouting it would be a regression.

    `lipgan_adapter` was the other half of this pair and was removed on
    2026-10-23 along with `wav2lip_hd`. The vestigial `lipgan` conda env
    (Python 3.7 / TF 1.15) was already unused -- the adapter had been ported
    to in-env PyTorch -- so nothing here depends on it any more.
    """
    src = (Path(__file__).resolve().parent.parent / "models" / "lipsync"
           / "wav2lip_adapter.py").read_text()
    assert "resolve_env_python" not in src, (
        "wav2lip_adapter must keep its sys.executable default")


def test_the_removed_adapters_are_gone_from_both_sides():
    """Registry and files together, so a half-removal cannot linger.

    `wav2lip_hd`'s `sync()` was `super().sync()` plus the GFPGAN pass -- its
    stage 1 was byte-identical to `wav2lip`. With restoration now stage 9i
    it IS `--lipsync wav2lip --restore gfpgan`, so an adapter for it would be
    a second way to say the same thing.
    """
    from models.lipsync import available_models

    root = Path(__file__).resolve().parent.parent
    for name in ("lipgan", "wav2lip_hd"):
        assert name not in available_models(), f"{name} is still registered"
    for f in ("lipgan_adapter.py", "wav2lip_hd_adapter.py"):
        assert not (root / "models" / "lipsync" / f).exists(), f"{f} remains"


def test_retired_models_still_resolve_for_old_jobs():
    """Display-only rows, and they are load-bearing.

    `"wav2lip_hd".startswith("wav2lip")` is True and `normalise()` matches on
    prefix, so dropping the FACTS row would resolve three finished HD jobs on
    disk to plain `wav2lip` -- the silent downgrade CLAUDE.md S4 records for
    exactly this pair. `options()` reads the registry, so nothing retired is
    ever offered.
    """
    import ui.lipsync_models as L

    for name in ("wav2lip_hd", "lipgan"):
        assert L.normalise(name) == name, f"{name} no longer resolves"
        assert name in L.RETIRED
        assert name not in {r["id"] for r in L.options()}, (
            f"{name} is retired but still offered")
    assert L.unregistered() == {"missing_facts": [], "stale_facts": []}


def test_env_example_documents_both_interpreters():
    """A blank here is not harmless, unlike LIPGAN/WAV2LIP."""
    src = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    for var in ("MUSETALK_PYTHON", "LATENTSYNC_PYTHON",
                "MUSETALK_REPO", "LATENTSYNC_REPO"):
        assert var in src, f"{var} missing from .env.example"


# --- wav2lip resize_factor ------------------------------------------------

def test_wav2lip_resize_factor_defaults_to_auto():
    """The default is now the POLICY, not a number.

    It was a flat 2, which cost every run half its resolution to protect
    against one case: a 4K source, where `inference.py:118` raises "Image too
    big to run face detection on GPU" at stage 9h, after ASR, MT, TTS and the
    mix have been paid for (measured 2026-10-18 — one shard of a 2-GPU split
    died there while the other succeeded). That error is an OOM fallback
    rather than a pixel limit, so the answer is a ceiling, not a constant.
    """
    import inspect
    from models.lipsync.wav2lip_adapter import Wav2LipSync
    assert inspect.signature(Wav2LipSync).parameters["resize_factor"].default is None
    assert Wav2LipSync().resize_factor is None


def test_resize_factor_is_still_overridable():
    """An explicit factor has to stay reachable and must win untouched --
    it is the only escape hatch, since there is no CLI flag for it."""
    from models.lipsync.wav2lip_adapter import Wav2LipSync, resolve_resize_factor
    assert Wav2LipSync(resize_factor=1).resize_factor == 1
    factor, why = resolve_resize_factor(Path("/nonexistent.mp4"), 1)
    assert factor == 1
    assert "explicit" in why


@pytest.mark.parametrize("dims,expected", [
    ((1920, 1080), 1),      # exactly the ceiling -- native
    ((1280, 720), 1),       # under it -- native
    ((640, 360), 1),
    ((3840, 2160), 2),      # 4K -- halves to exactly 1920x1080
    ((2560, 1440), 2),      # above the ceiling, so halved
    ((1920, 1082), 2),      # two scanlines over is still over
])
def test_the_ceiling_decides_the_factor(monkeypatch, dims, expected):
    """At or below 1080p render native; above it, halve. Nothing else."""
    import models.lipsync.wav2lip_adapter as W
    import backend.lipsync as BL
    monkeypatch.setattr(BL, "_video_dims", lambda p: dims)
    factor, _why = W.resolve_resize_factor(Path("/clip.mp4"))
    assert factor == expected


def test_an_unprobeable_source_keeps_the_safe_factor(monkeypatch):
    """None is a silent answer (CLAUDE.md §1), and the two guesses are not
    symmetric: guessing native risks an OOM at stage 9h after the whole dub
    has been paid for, guessing halved costs resolution on one file."""
    import models.lipsync.wav2lip_adapter as W
    import backend.lipsync as BL
    monkeypatch.setattr(BL, "_video_dims", lambda p: None)
    factor, why = W.resolve_resize_factor(Path("/clip.mp4"))
    assert factor == 2
    assert "could not probe" in why


@pytest.mark.parametrize("dims,expected", [
    ((3840, 2160), (1920, 1080)),
    ((1920, 1080), (1920, 1080)),
    ((1280, 720), (1280, 720)),     # a ceiling, never an upscale
    ((1440, 1440), (1080, 1080)),
    ((1079, 1081), (1078, 1080)),   # both sides forced even
])
def test_cap_dims_is_a_ceiling_with_even_sides(dims, expected):
    """`wav2lip_hd` restores to this, so it decides the HD output size too.
    Odd dimensions matter: libx264 with yuv420p refuses them."""
    from models.lipsync.wav2lip_adapter import cap_dims
    assert cap_dims(dims) == expected


def test_the_resolved_factor_reaches_argv_and_the_coord_key():
    """Both, or the cache poisons itself.

    Keying the coord cache on `self.resize_factor` would file every
    auto-resolved run under the same `None`, so a 4K render's half-size boxes
    would load for a 720p clip and be painted at the wrong scale -- at exit
    code 0, since upstream only checks the box COUNT against the frame count.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "models" / "lipsync"
           / "wav2lip_adapter.py").read_text()
    assert '"--resize_factor", str(resize_factor)' in src
    assert '"resize_factor": resize_factor,' in src
    assert "str(self.resize_factor)" not in src
