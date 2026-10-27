"""The Liao-family ASD adapters, and the shared MFCC+crop half.

Covers `light_asd` (CVPR 2023) and `lr_asd` (IJCV 2025), which share one
implementation because their upstreams share an API exactly.

The grep guards at the bottom are the point of this file. `ts_talknet`
carried its own copy of the cropping and the int16 rescale, missed both
halves of the 2026-10-11 TalkNet fix, and sat broken until a cleanup pass
found it -- it would have scored near-zero had it loaded at all. These are
the third and fourth models to need that code, so the tests assert it is
IMPORTED rather than re-typed, in both directions.
"""
from __future__ import annotations

import ast
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from models.asd import _mfcc_av, available_models
from models.asd.lightasd_adapter import LRASD, LightASD

REPO = Path(__file__).resolve().parents[1]
ASD_DIR = REPO / "models" / "asd"

#: (class, vendor dir, env prefix) for every model sharing this adapter.
LIAO = [
    pytest.param(LightASD, "lightasd", "LIGHTASD", id="light_asd"),
    pytest.param(LRASD, "lrasd", "LRASD", id="lr_asd"),
]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_both_liao_models_are_registered():
    assert "light_asd" in available_models()
    assert "lr_asd" in available_models()


def test_registry_spec_is_a_dict_literal():
    """`scripts/repo_checkup.py` reads it with `ast.parse`, never importing
    the module -- so building it dynamically silently reports zero adapters
    for this stage in the five-hourly README regeneration."""
    src = (ASD_DIR / "__init__.py").read_text()
    assert re.search(r"^_REGISTRY_SPEC.*=\s*\{", src, re.M)
    assert '"light_asd": (".lightasd_adapter", "LightASD")' in src
    assert '"lr_asd": (".lightasd_adapter", "LRASD")' in src


def _docstring_ids(tree):
    out = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(id(first.value))
    return out


def test_no_third_party_paths_left_in_asd():
    """The 2026-10-23 move put every ASD upstream under models/asd/vendor/.

    The legacy fallback that `_vendor.resolve_vendor_repo` consumes is the
    ONE place a `third_party/` path is still legitimate -- it is what keeps
    a checkout where the move never ran working. It arrives either as a
    `legacy=` argument or as a `legacy_path` class attribute.

    This parses rather than grepping lines. A line-based version failed on
    the registry docstring, which *describes* the fallback in prose -- and
    the fix is not to widen the allowance until nothing trips it, because
    then it stops catching a real path too.
    """
    for py in sorted(ASD_DIR.glob("*.py")):
        tree = ast.parse(py.read_text())
        allowed = set(_docstring_ids(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "legacy_path":
                        allowed.add(id(node.value))
            elif isinstance(node, ast.keyword) and node.arg == "legacy":
                allowed.add(id(node.value))

        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            if "third_party" not in node.value or id(node) in allowed:
                continue
            pytest.fail(f"{py.name}:{node.lineno} uses a third_party path "
                        f"outside the legacy fallback: {node.value!r}")


# ---------------------------------------------------------------------------
# the shared half
# ---------------------------------------------------------------------------

def test_window_bounds_covers_every_frame_exactly_once():
    bounds = _mfcc_av.window_bounds(1000, 150)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 1000
    for (_, prev_end), (nxt_start, _) in zip(bounds, bounds[1:]):
        assert prev_end == nxt_start


def test_window_bounds_single_pass_when_disabled_or_short():
    assert _mfcc_av.window_bounds(1000, 0) == [(0, 1000)]
    assert _mfcc_av.window_bounds(1000, -1) == [(0, 1000)]
    assert _mfcc_av.window_bounds(100, 150) == [(0, 100)]


def test_window_bounds_merges_a_stub_tail():
    """A tail shorter than the 3D stem's context window is scored almost
    entirely on padding, so it is given to the window before it."""
    bounds = _mfcc_av.window_bounds(153, 150)
    assert bounds == [(0, 153)]
    # ...but a tail with real content keeps its own window.
    bounds = _mfcc_av.window_bounds(170, 150)
    assert bounds == [(0, 150), (150, 170)]


def test_load_audio_int16_rescales(tmp_path):
    """The whole reason `_mfcc_av` exists. Floats in [-1, 1] shift every
    log-energy cepstral term and pin every score near zero, silently."""
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "a.wav"
    sig = np.full(1600, 0.5, dtype=np.float32)
    sf.write(str(wav), sig, 16000)

    out = _mfcc_av.load_audio_int16(wav)
    assert out.max() == pytest.approx(0.5 * 32768.0, rel=1e-3)


def test_load_audio_int16_refuses_wrong_sample_rate(tmp_path):
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "b.wav"
    sf.write(str(wav), np.zeros(8000, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="16 kHz"):
        _mfcc_av.load_audio_int16(wav, model_name="Light-ASD")


def test_mfcc_window_scaling_is_a_no_op_at_25fps():
    """TalkNet passes no fps and Light-ASD passes the real one. At 25 they
    must agree, or the two adapters disagree on identical input."""
    pytest.importorskip("python_speech_features")
    audio = (np.random.default_rng(0).standard_normal(16000) * 1000.0)
    unscaled = _mfcc_av.mfcc_features(audio)
    at_25 = _mfcc_av.mfcc_features(audio, fps=25.0)
    assert unscaled.shape == at_25.shape
    assert np.allclose(unscaled, at_25)


def test_mfcc_window_scales_away_from_25fps():
    pytest.importorskip("python_speech_features")
    audio = (np.random.default_rng(0).standard_normal(16000) * 1000.0)
    # Half the step means roughly twice the rows.
    assert (_mfcc_av.mfcc_features(audio, fps=50.0).shape[0]
            > 1.8 * _mfcc_av.mfcc_features(audio, fps=25.0).shape[0])


def test_align_pair_trims_to_common_length():
    torch = pytest.importorskip("torch")
    a = torch.zeros(1, 17, 128)
    v = torch.zeros(1, 16, 128)
    a2, v2 = _mfcc_av.align_pair(a, v)
    assert a2.shape[1] == v2.shape[1] == 16
    # Light-ASD's backend is `x1 + x2`, so this has to actually make the
    # addition legal rather than merely look tidy.
    assert (a2 + v2).shape == (1, 16, 128)


def test_crop_one_is_square_and_grayscale():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    patch = _mfcc_av.crop_one(frame, (100, 100, 180, 180), 112, cv2)
    assert patch.shape == (112, 112)
    assert patch.dtype == np.uint8


def test_crop_one_survives_a_box_off_frame():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    patch = _mfcc_av.crop_one(frame, (-50, -50, -10, -10), 112, cv2)
    assert patch.shape == (112, 112)


# ---------------------------------------------------------------------------
# adapter wiring, without the weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,vendor,prefix", LIAO)
def test_adapter_resolves_into_the_vendor_root(cls, vendor, prefix,
                                               monkeypatch):
    monkeypatch.delenv(f"{prefix}_REPO", raising=False)
    monkeypatch.delenv(f"{prefix}_CHECKPOINT", raising=False)

    a = cls(device="cpu")
    assert a.repo_path.as_posix().endswith(f"models/asd/vendor/{vendor}")
    assert a.checkpoint.name == "finetuning_TalkSet.model"


@pytest.mark.parametrize("cls,vendor,prefix", LIAO)
def test_relative_checkpoint_resolves_inside_the_clone(cls, vendor, prefix,
                                                       monkeypatch):
    """`.env` naming `weight/pretrain_AVA.model` has to keep meaning the
    one in the clone after a vendor move."""
    monkeypatch.delenv(f"{prefix}_REPO", raising=False)
    monkeypatch.setenv(f"{prefix}_CHECKPOINT", "weight/pretrain_AVA.model")

    a = cls(device="cpu")
    assert a.checkpoint.is_absolute()
    assert a.checkpoint.as_posix().endswith(
        f"models/asd/vendor/{vendor}/weight/pretrain_AVA.model")


@pytest.mark.parametrize("cls,vendor,prefix", LIAO)
def test_window_frames_blank_env_means_the_default(cls, vendor, prefix,
                                                   monkeypatch):
    """A blank `.env` line must not read as 0 -- that would silently restore
    the single-pass behaviour the window exists to end."""
    monkeypatch.setenv(f"{prefix}_WINDOW_FRAMES", "")
    assert cls(device="cpu").window_frames == 150

    monkeypatch.setenv(f"{prefix}_WINDOW_FRAMES", "0")
    assert cls(device="cpu").window_frames == 0


@pytest.mark.parametrize("cls,vendor,prefix", LIAO)
def test_missing_repo_names_the_clone_command(cls, vendor, prefix,
                                              tmp_path):
    """The repo is made missing directly, NOT by pinning `*_REPO` at a
    bogus path -- that does not reach this error. `resolve_vendor_repo`
    treats a pin whose target does not exist as a stale pin and falls back
    to the vendored clone, which is present here, so the adapter loads
    normally. That is the documented behaviour; an earlier version of this
    test asserted the opposite and passed only while nothing was cloned.
    """
    a = cls(device="cpu")
    a.repo_path = tmp_path / "nope"
    with pytest.raises(RuntimeError, match="git clone"):
        a._ensure_loaded()


@pytest.mark.parametrize("cls,vendor,prefix", LIAO)
def test_stale_repo_pin_falls_back_to_the_vendor_clone(cls, vendor, prefix,
                                                       monkeypatch, tmp_path):
    """A `.env` pin that outlives its target must not fail the run."""
    if not (REPO / "models" / "asd" / "vendor" / vendor).exists():
        pytest.skip(f"{vendor} is not cloned on this host")
    monkeypatch.setenv(f"{prefix}_REPO", str(tmp_path / "gone"))
    a = cls(device="cpu")
    assert a.repo_path.as_posix().endswith(f"models/asd/vendor/{vendor}")


@pytest.mark.parametrize("cls,vendor,prefix", LIAO)
def test_env_prefixes_do_not_collide(cls, vendor, prefix, monkeypatch):
    """Each model reads only its OWN variables. `LIGHTASD` is not a prefix
    of `LRASD` and vice versa, but the lookup is built by f-string, so an
    attribute left unoverridden on the subclass would silently make both
    models read one set of pins."""
    for p in ("LIGHTASD", "LRASD"):
        monkeypatch.delenv(f"{p}_REPO", raising=False)
    other = "LRASD" if prefix == "LIGHTASD" else "LIGHTASD"
    monkeypatch.setenv(f"{other}_REPO", "/tmp/should-not-be-read")

    a = cls(device="cpu")
    assert "should-not-be-read" not in a.repo_path.as_posix()


def test_lrasd_overrides_every_constant_that_differs():
    """A constant left inherited points LR-ASD at Light-ASD's clone, which
    loads and scores with no error and the wrong weights."""
    for attr in ("name", "vendor_name", "env_prefix",
                 "legacy_path", "clone_url"):
        assert getattr(LRASD, attr) != getattr(LightASD, attr), attr
    # ...and the one that is SUPPOSED to be shared stays shared.
    assert LRASD.default_checkpoint == LightASD.default_checkpoint


def test_lrasd_reuses_the_scoring_path():
    """The whole reason it is a subclass. If someone re-implements these on
    LRASD, the two models can drift the way ts_talknet drifted."""
    assert issubclass(LRASD, LightASD)
    for method in ("score", "_score_track", "_score_window",
                   "_ensure_loaded"):
        assert method not in vars(LRASD), (
            f"LRASD overrides {method}; it should inherit it")


def _fake_asd_module(clone: Path):
    mod = types.ModuleType("ASD")
    mod.__file__ = str(clone / "ASD.py")
    return mod


def test_sibling_clone_already_imported_is_refused(monkeypatch, tmp_path):
    """Both clones ship a top-level `ASD` module. Python caches by NAME, so
    the second model loaded in one process would score with the first one's
    network at no error."""
    monkeypatch.setitem(sys.modules, "ASD",
                        _fake_asd_module(tmp_path / "lightasd"))
    a = LRASD(device="cpu")
    a.repo_path = tmp_path / "lrasd"
    with pytest.raises(RuntimeError, match="already imported"):
        a._refuse_sibling_on_path()


def test_guard_allows_reloading_the_same_clone(monkeypatch, tmp_path):
    """Re-loading the model already imported is legal and must stay so."""
    monkeypatch.setitem(sys.modules, "ASD",
                        _fake_asd_module(tmp_path / "lrasd"))
    a = LRASD(device="cpu")
    a.repo_path = tmp_path / "lrasd"
    a._refuse_sibling_on_path()                     # must not raise


def test_guard_ignores_a_stale_sys_path_entry(monkeypatch, tmp_path):
    """The bug this guard shipped with: a `sys.path` entry outlives the
    load that added it, so scanning the path refused a model that nothing
    had actually imported. With no `ASD` module loaded there is no
    conflict, whatever is on the path."""
    monkeypatch.delitem(sys.modules, "ASD", raising=False)
    monkeypatch.setattr(sys, "path", ["/somewhere/lightasd", *sys.path])
    a = LRASD(device="cpu")
    a.repo_path = tmp_path / "lrasd"
    a._refuse_sibling_on_path()                     # must not raise


# ---------------------------------------------------------------------------
# drift guards — the ts_talknet lesson
# ---------------------------------------------------------------------------

ADAPTERS = ["talknet_adapter.py", "lightasd_adapter.py"]


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_imports_the_shared_half(name):
    src = (ASD_DIR / name).read_text()
    assert "from ._mfcc_av import" in src, (
        f"{name} must import the shared MFCC+crop code, not re-type it")


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_does_not_rescale_audio_itself(name):
    """One home for the int16 factor. A second one drifts."""
    body = (ASD_DIR / name).read_text()
    code = "\n".join(l for l in body.splitlines()
                     if not l.lstrip().startswith("#"))
    assert "32768" not in code, (
        f"{name} rescales audio itself; call _mfcc_av.load_audio_int16")


@pytest.mark.parametrize("name", ADAPTERS)
def test_adapter_does_not_call_mfcc_directly(name):
    code = "\n".join(l for l in (ASD_DIR / name).read_text().splitlines()
                     if not l.lstrip().startswith("#"))
    assert "python_speech_features" not in code, (
        f"{name} builds MFCC itself; call _mfcc_av.mfcc_features")


def test_shared_module_keeps_the_int16_rationale():
    """Delete the measurement and the next person deletes the rescale."""
    src = (ASD_DIR / "_mfcc_av.py").read_text()
    assert "32768" in src
    assert "int16" in src.lower()
