"""LatentSync 1.5 and 1.6 share a codebase, so the pairing is the risk.

Upstream's `docs/changelog_v1.6.md`: "we did not make any changes to the
model structure or training strategy... To switch between versions, you only
need to load the corresponding checkpoint and modify the `resolution`
parameter in the U-Net config file."

Identical architecture means a 1.5 checkpoint loads without complaint under
a 512 config and simply generates badly -- no exception, no warning, a
finished mp4 with a soft mouth, after ~80x the clip length of GPU time.
These tests pin the guard against that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from models.lipsync import available_models, get_lipsync          # noqa: E402
from models.lipsync import latentsync_adapter as LS               # noqa: E402


def test_only_1_5_is_registered():
    """1.6 was removed on 2026-10-28 -- it needed ~18 GB against 16 GiB
    cards, so it was the one registered model that could not run here.

    The id stays `latentsync15` rather than reverting to `latentsync`,
    because the id names the (checkpoint, resolution) pair. Reverting it
    would make a future 1.6 a silent prefix alias for this one.
    """
    assert "latentsync15" in available_models()
    assert "latentsync" not in available_models()
    assert sorted(LS.VERSIONS) == ["1.5"]


@pytest.mark.parametrize("model_id,version,resolution", [
    ("latentsync15", "1.5", 256),
])
def test_each_id_resolves_its_own_pair(monkeypatch, model_id, version, resolution):
    monkeypatch.delenv("LATENTSYNC_VERSION", raising=False)
    monkeypatch.delenv("LATENTSYNC_UNET_CONFIG", raising=False)
    monkeypatch.delenv("LATENTSYNC_UNET_CKPT", raising=False)
    a = get_lipsync(model_id)
    assert a.version == version
    assert a.expected_resolution == resolution
    assert a.unet_config_path == LS.VERSIONS[version][0]
    assert a.unet_ckpt_path == LS.VERSIONS[version][1]


def test_the_checkpoint_keeps_its_disambiguated_name():
    """Upstream shipped 1.5 and 1.6 both as `latentsync_unet.pt`. 1.6 is
    gone, but its checkpoint may still sit in the vendor checkout under the
    bare name -- so 1.5 keeps the suffixed one rather than reclaiming it."""
    assert LS.VERSIONS["1.5"][1].endswith("latentsync_unet_1.5.pt")


def test_the_reported_model_name_follows_the_version():
    """It was hardcoded `latentsync-1.6`, so a 1.5 render would have been
    recorded in the manifest as 1.6."""
    src = (REPO / "models" / "lipsync" / "latentsync_adapter.py").read_text()
    assert '"latentsync-1.6"' not in src
    assert 'f"latentsync-{self.version}"' in src


# --- the mismatch guard ----------------------------------------------------

def _fake_repo(tmp_path: Path, cfg_name: str, resolution: int,
               ckpt_name: str) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "inference.py").write_text("")
    cfg = tmp_path / cfg_name
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(f"data:\n  resolution: {resolution}\n")
    ckpt = tmp_path / ckpt_name
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not-really-weights")
    return tmp_path


def test_a_version_config_mismatch_is_refused(tmp_path):
    """1.5's checkpoint under a 512 config. The whole point.

    Still reachable with 1.6 gone: `LATENTSYNC_UNET_CONFIG` can point at a
    hand-edited 512 yaml, and the architecture is identical, so the
    mismatch loads without error and renders a soft mouth."""
    cfg, ckpt, _res = LS.VERSIONS["1.5"]
    root = _fake_repo(tmp_path, "configs/unet/stage2_512.yaml", 512, ckpt)
    a = LS.LatentSyncSync(repo_path=str(root),
                            unet_config_path="configs/unet/stage2_512.yaml",
                            unet_ckpt_path=ckpt, python_bin=sys.executable)
    with pytest.raises(RuntimeError, match="resolution"):
        a._ensure_loaded()


def test_a_matching_pair_loads(tmp_path):
    cfg, ckpt, res = LS.VERSIONS["1.5"]
    root = _fake_repo(tmp_path, cfg, res, ckpt)
    a = LS.LatentSyncSync(repo_path=str(root), python_bin=sys.executable)
    a._ensure_loaded()
    assert a._loaded


def test_an_unreadable_resolution_warns_rather_than_refusing(tmp_path, caplog):
    """A config using anchors or a syntax the regex does not expect must
    degrade to "cannot tell", not block a render that would have worked."""
    cfg, ckpt, _res = LS.VERSIONS["1.5"]
    root = _fake_repo(tmp_path, cfg, 256, ckpt)
    (root / cfg).write_text("data:\n  something_else: 1\n")
    a = LS.LatentSyncSync(repo_path=str(root), python_bin=sys.executable)
    a._ensure_loaded()
    assert a._loaded


def test_the_missing_checkpoint_error_names_the_right_hf_repo(tmp_path):
    """1.5 and 1.6 shipped the same filename from different repos, so the
    error names the repo, not just the file."""
    cfg, ckpt, res = LS.VERSIONS["1.5"]
    root = _fake_repo(tmp_path, cfg, res, ckpt)
    (root / ckpt).unlink()
    a = LS.LatentSyncSync(repo_path=str(root), python_bin=sys.executable)
    with pytest.raises(RuntimeError, match="LatentSync-1.5"):
        a._ensure_loaded()


def test_an_unknown_version_is_refused():
    with pytest.raises(ValueError, match="Unknown LatentSync version"):
        LS.LatentSyncSync(version="2.0", python_bin=sys.executable)


# --- the vendor move (2026-10-23) ------------------------------------------

def test_the_resolver_prefers_the_new_home_but_accepts_the_old(tmp_path,
                                                               monkeypatch):
    """~21 GB has to move for the new layout to be real, and that move is a
    shell command the operator runs. Between the code change and the move --
    and on any checkout where it never happened -- the old path is still the
    correct answer, so the resolver picks by EXISTENCE, not preference."""
    from models.lipsync.base import resolve_vendor_repo
    monkeypatch.delenv("LATENTSYNC_REPO", raising=False)

    legacy = tmp_path / "third_party" / "LatentSync"
    legacy.mkdir(parents=True)
    got = resolve_vendor_repo("latentsync", "LATENTSYNC_REPO",
                              legacy=str(legacy))
    assert got == legacy, "the old location must still work before the move"


def test_an_explicit_path_and_the_env_still_win(tmp_path, monkeypatch):
    from models.lipsync.base import resolve_vendor_repo
    assert resolve_vendor_repo("latentsync", "LATENTSYNC_REPO",
                               str(tmp_path / "explicit")) \
        == tmp_path / "explicit"
    monkeypatch.setenv("LATENTSYNC_REPO", str(tmp_path / "from_env"))
    assert resolve_vendor_repo("latentsync", "LATENTSYNC_REPO") \
        == tmp_path / "from_env"


def test_a_missing_repo_names_the_NEW_location(monkeypatch):
    """So the adapter's "repo not found" error points at where it should be,
    not where it used to be."""
    from models.lipsync.base import resolve_vendor_repo, VENDOR_ROOT
    monkeypatch.delenv("NOPE_REPO", raising=False)
    got = resolve_vendor_repo("nope", "NOPE_REPO", legacy="third_party/Nope")
    assert VENDOR_ROOT in str(got)


def test_wav2lips_checkpoint_follows_its_repo(monkeypatch):
    """It was a second hardcoded `third_party/...` path, so after the move
    the repo would resolve to its new home while the weights pointed at the
    old one -- and that raises at stage 9h, after the whole dub is paid for."""
    from models.lipsync.wav2lip_adapter import Wav2LipSync
    monkeypatch.delenv("WAV2LIP_CHECKPOINT", raising=False)
    a = Wav2LipSync(repo_path="/tmp/somewhere/wav2lip", python_bin=sys.executable)
    assert str(a.checkpoint).startswith("/tmp/somewhere/wav2lip")


def test_the_vendor_dir_is_gitignored():
    """21 GB is one `git add -A` away from history. LatentSync's UNet is a
    4.7 GB `.pt`, and `*.pt` was NOT ignored before this move -- it escaped
    only because `third_party/` was ignored wholesale."""
    import subprocess
    for p in ("models/lipsync/vendor/x",
              "models/lipsync/vendor/latentsync/checkpoints/latentsync_unet.pt"):
        rc = subprocess.run(["git", "check-ignore", "-q", p],
                            cwd=REPO).returncode
        assert rc == 0, f"{p} is not gitignored"
