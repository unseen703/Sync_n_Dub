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


def test_lipgan_and_wav2lip_are_deliberately_not_changed():
    """Those two DO run in `dubbing`; rerouting them would be a regression."""
    for name in ("lipgan_adapter", "wav2lip_adapter"):
        src = (Path(__file__).resolve().parent.parent / "models" / "lipsync"
               / f"{name}.py").read_text()
        assert "resolve_env_python" not in src, (
            f"{name} must keep its sys.executable default")


def test_env_example_documents_both_interpreters():
    """A blank here is not harmless, unlike LIPGAN/WAV2LIP."""
    src = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    for var in ("MUSETALK_PYTHON", "LATENTSYNC_PYTHON",
                "MUSETALK_REPO", "LATENTSYNC_REPO"):
        assert var in src, f"{var} missing from .env.example"


# --- wav2lip resize_factor ------------------------------------------------

def test_wav2lip_defaults_to_resize_factor_2():
    """1 makes S3FD refuse a large frame outright.

    `third_party/Wav2Lip/inference.py:81` raises "Image too big to run face
    detection on GPU" rather than downscaling, and it does so at stage 9h --
    after ASR, MT, TTS and the mix have been paid for. Measured 2026-10-18 on
    a 4K source: one shard died there while the other shard of the same run
    succeeded, so it is frame size, not the model.
    """
    import inspect
    from models.lipsync.wav2lip_adapter import Wav2LipSync
    assert inspect.signature(Wav2LipSync).parameters["resize_factor"].default == 2
    assert Wav2LipSync().resize_factor == 2


def test_resize_factor_is_still_overridable():
    """Full resolution has to stay reachable: the factor halves the whole
    output (1080p -> 960x540), which is a real cost on a small-frame source."""
    from models.lipsync.wav2lip_adapter import Wav2LipSync
    assert Wav2LipSync(resize_factor=1).resize_factor == 1


def test_the_factor_reaches_the_subprocess_argv():
    """It is passed as `--resize_factor`; a default that never ships is no
    default at all."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "models" / "lipsync"
           / "wav2lip_adapter.py").read_text()
    assert '"--resize_factor", str(self.resize_factor)' in src
