"""Registry behaviour for the face-restoration stage.

These exercise the parts that are pure logic — name resolution, kwarg
filtering, and the no-face warning — without touching a GPU, a checkpoint
or a third_party clone. The adapter's actual inference is covered by
running it, not by a unit test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.restore import (  # noqa: E402
    BaseRestore,
    RestoreRequest,
    RestoreResult,
    available_models,
    get_restore,
)


def test_codeformer_is_registered():
    assert "codeformer" in available_models()


def test_unknown_model_names_the_alternatives():
    with pytest.raises(ValueError) as e:
        get_restore("gfpgan")
    assert "codeformer" in str(e.value)


@pytest.mark.parametrize("name", ["CodeFormer", " codeformer ", "CODEFORMER"])
def test_name_lookup_is_case_and_space_insensitive(name):
    assert get_restore(name).name == "codeformer"


def test_known_kwargs_reach_the_adapter():
    a = get_restore("codeformer", fidelity=0.3, upscale=2, only_center_face=True)
    assert (a.fidelity, a.upscale, a.only_center_face) == (0.3, 2, True)


def test_upscale_defaults_to_1():
    """A restoration pass must not quietly also be an upscale.

    Upstream's default is 2, which would change the frame's resolution and
    make a restored render incomparable with the raw one beside it.
    """
    assert get_restore("codeformer").upscale == 1


def test_bg_upsampler_defaults_to_off():
    """Upstream's notebook enables Real-ESRGAN; we do not.

    It rewrites the whole frame and costs VRAM the lip-sync model has
    usually just spent — see models/restore/codeformer_adapter.py.
    """
    assert get_restore("codeformer").bg_upsampler == "None"


def test_unknown_kwargs_warn_and_are_dropped():
    with pytest.warns(UserWarning, match="nonsense_knob"):
        a = get_restore("codeformer", nonsense_knob=1, fidelity=0.9)
    assert not hasattr(a, "nonsense_knob")
    assert a.fidelity == 0.9


def test_cuda_device_pins_the_subprocess():
    env = get_restore("codeformer", cuda_device=1)._subprocess_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_subprocess_env_preloads_conda_libstdcxx():
    """Face detection hits the cv2/libtbb CXXABI trap without this.

    See CLAUDE.md §1 — whichever libstdc++ loads first decides whether
    the import works at all.
    """
    assert "libstdc++" in get_restore("codeformer")._subprocess_env()["LD_PRELOAD"]


def test_missing_repo_is_caught_before_any_work():
    a = get_restore("codeformer", repo_path="/nonexistent/CodeFormer")
    with pytest.raises(RuntimeError, match="repo not found"):
        a._ensure_loaded()


class _NoFaceRestore(BaseRestore):
    """An adapter that succeeds while finding nothing to restore."""
    name = "noface"

    def _ensure_loaded(self) -> None:
        self._loaded = True

    def restore(self, request: RestoreRequest) -> RestoreResult:
        return RestoreResult(model_name=self.name, out_video=request.out_path,
                             duration_s=1.0, fps=25.0,
                             frames=25, frames_with_faces=0)


def test_zero_faces_is_reported_rather_than_passing_as_success(monkeypatch, tmp_path):
    """rc=0 with no face anywhere means the stage re-encoded and nothing else.

    The voice-assignment stage once ran green for several dubs while its
    gender matching was dead code; this is the same failure shape, so the
    stage has to say so itself.
    """
    import backend.lipsync as L
    import models.restore as R

    monkeypatch.setitem(R._REGISTRY_SPEC, "noface", ("", ""))
    monkeypatch.setattr(R, "_load", lambda key: _NoFaceRestore)

    info = L._run_restore(
        "noface", video=tmp_path / "in.mp4", audio=tmp_path / "in.wav",
        out_path=tmp_path / "out.mp4", kwargs={}, gpus=[],
    )
    assert info["error"] is None
    assert "no face" in info["warning"]


def test_restore_failure_does_not_discard_the_lipsync_render(tmp_path):
    """A restoration fault reports; it must not raise past the caller.

    The lip-sync render is already a usable result at that point.
    """
    import backend.lipsync as L

    info = L._run_restore(
        "no_such_model", video=tmp_path / "in.mp4", audio=tmp_path / "in.wav",
        out_path=tmp_path / "out.mp4", kwargs={}, gpus=[],
    )
    assert "Unknown restoration model" in info["error"]


def test_relative_out_path_is_resolved_before_use(tmp_path, monkeypatch):
    """A relative out_path must not be read against the repo's cwd.

    The adapter shells out with cwd=third_party/CodeFormer, so an
    unresolved `--input_path data/jobs/<id>/...` resolves against that
    clone instead and upstream reports "No input image/video is found".
    Callers pass relative out dirs routinely, so this is the normal case.
    """
    import models.restore.codeformer_adapter as C

    seen = {}

    def fake_run(cmd, **kw):
        # The frame-extract call is the first one; record where it writes.
        seen.setdefault("cmd", cmd)
        raise RuntimeError("stop after the first subprocess call")

    monkeypatch.setattr(C.subprocess, "run", fake_run)
    monkeypatch.setattr(C, "_probe_video", lambda p: (25.0, 1.0))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src.mp4").write_bytes(b"")

    a = get_restore("codeformer", repo_path=str(tmp_path))
    a._loaded = True
    with pytest.raises(RuntimeError, match="stop after"):
        a.restore(RestoreRequest(video=tmp_path / "src.mp4",
                                 out_path=Path("out/rel.mp4")))

    frame_target = seen["cmd"][-1]
    assert Path(frame_target).is_absolute(), frame_target
