"""scripts/fetch_assets.py — manifest shape, feature matching, status checks.

No test here touches the network or the real ~16 GB of vendored weights;
everything downloadable is exercised against small synthetic files instead.
The real manifest (`models/assets.lock.json`) is checked for internal
consistency only — every dest unique, every required field present, sizes
and hashes well-formed — not against what's actually on disk, since a CI
checkout will not have the weights.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_assets as FA  # noqa: E402


@pytest.fixture
def manifest():
    return FA.load_manifest()


def test_manifest_loads_and_has_both_sections(manifest):
    assert manifest["vendor_repos"]
    assert manifest["weights"]


def test_every_weight_has_required_fields(manifest):
    for w in manifest["weights"]:
        for field in ("id", "feature", "dest", "bytes", "sha256", "source"):
            assert field in w, f"{w.get('id', '?')} missing {field!r}"
        assert len(w["sha256"]) == 64, f"{w['id']}: sha256 is not 64 hex chars"
        int(w["sha256"], 16)  # raises if not valid hex
        assert w["bytes"] > 0
        assert w["source"] in ("hf", "url", "gdrive")
        if w["source"] == "hf":
            assert w.get("repo") and w.get("filename")
        if w["source"] == "url":
            assert w.get("url", "").startswith("https://")
        if w["source"] == "gdrive":
            assert w.get("gdrive_id")


def test_every_vendor_repo_has_required_fields(manifest):
    for v in manifest["vendor_repos"]:
        for field in ("name", "feature", "url", "commit", "dest"):
            assert field in v, f"{v.get('name', '?')} missing {field!r}"
        assert len(v["commit"]) == 40, f"{v['name']}: commit is not a full 40-char sha"


def test_no_duplicate_ids_or_dests(manifest):
    ids = [w["id"] for w in manifest["weights"]]
    assert len(ids) == len(set(ids)), "duplicate weight id"
    dests = [w["dest"] for w in manifest["weights"]]
    assert len(dests) == len(set(dests)), "two weights share a destination"
    names = [v["name"] for v in manifest["vendor_repos"]]
    assert len(names) == len(set(names)), "duplicate vendor repo name"


def test_patch_file_exists_for_every_vendor_repo_that_names_one(manifest):
    for v in manifest["vendor_repos"]:
        if v.get("patch"):
            assert (ROOT / v["patch"]).is_file(), f"{v['name']}: {v['patch']} missing"


# ---------------------------------------------------------------------------
# --feature matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_feature,wanted,expected", [
    ("lipsync:wav2lip", ["all"], True),
    ("lipsync:wav2lip", ["lipsync:wav2lip"], True),
    ("lipsync:wav2lip", ["lipsync"], True),
    ("lipsync:wav2lip", ["lipsync:musetalk"], False),
    ("lipsync:wav2lip", ["restore"], False),
    ("restore:gfpgan", ["restore"], True),
    ("asd:talknet", ["asd:talknet"], True),
])
def test_feature_matching(item_feature, wanted, expected):
    assert FA._matches_feature(item_feature, wanted) is expected


# ---------------------------------------------------------------------------
# status / check against real files (synthetic, not the real weights)
# ---------------------------------------------------------------------------

def _fake_weight(path: Path, content: bytes) -> dict:
    path.write_bytes(content)
    return {
        "id": "fake", "feature": "test:fake", "dest": str(path),
        "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        "source": "url", "url": "https://example.invalid/fake.bin",
    }


def test_status_weight_present(tmp_path):
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    assert FA.status_weight(w) == "present"


def test_status_weight_missing(tmp_path):
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    (tmp_path / "f.bin").unlink()
    assert FA.status_weight(w) == "missing"


def test_status_weight_size_mismatch(tmp_path):
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    (tmp_path / "f.bin").write_bytes(b"short")
    assert FA.status_weight(w) == "size-mismatch"


def test_check_weight_ok(tmp_path):
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    assert FA.check_weight(w) == "ok"


def test_check_weight_hash_mismatch_even_at_right_size(tmp_path):
    """The case a size-only check (status_weight) cannot catch -- corruption
    that happens to land on the same byte count."""
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    (tmp_path / "f.bin").write_bytes(b"HELLO WORLD")  # same length, wrong bytes
    assert FA.status_weight(w) == "present"           # size alone says fine
    assert FA.check_weight(w) == "hash-mismatch"       # hash catches it


def test_fetch_weight_skips_when_already_correct(tmp_path, monkeypatch):
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    called = []
    monkeypatch.setattr(FA, "_download_url", lambda *a: called.append(True))
    assert FA.fetch_weight(w, dry_run=False, force=False) is True
    assert not called, "should not re-download a file already at the right size"


def test_fetch_weight_force_redownloads(tmp_path, monkeypatch):
    w = _fake_weight(tmp_path / "f.bin", b"hello world")
    called = []

    def _redownload(w, dest):
        called.append(True)
        dest.write_bytes(b"hello world")

    monkeypatch.setattr(FA, "_download_url", _redownload)
    assert FA.fetch_weight(w, dry_run=False, force=True) is True
    assert called, "force=True must re-download even when the file already looks right"


def test_fetch_weight_rejects_and_removes_a_bad_download(tmp_path, monkeypatch):
    w = {
        "id": "fake", "feature": "test:fake", "dest": str(tmp_path / "f.bin"),
        "bytes": 11, "sha256": hashlib.sha256(b"hello world").hexdigest(),
        "source": "url", "url": "https://example.invalid/fake.bin",
    }

    def _wrong_download(w, dest):
        dest.write_bytes(b"wrong wrong")  # wrong content -> wrong hash

    monkeypatch.setattr(FA, "_download_url", _wrong_download)
    ok = FA.fetch_weight(w, dry_run=False, force=False)
    assert ok is False
    assert not (tmp_path / "f.bin").exists(), "a hash-mismatched download must not be left on disk"


def test_dry_run_never_writes(tmp_path, monkeypatch):
    w = {
        "id": "fake", "feature": "test:fake", "dest": str(tmp_path / "f.bin"),
        "bytes": 11, "sha256": "0" * 64,
        "source": "url", "url": "https://example.invalid/fake.bin",
    }
    monkeypatch.setattr(FA, "_download_url",
                         lambda *a: (_ for _ in ()).throw(AssertionError("dry-run must not download")))
    assert FA.fetch_weight(w, dry_run=True, force=False) is True
    assert not (tmp_path / "f.bin").exists()


def test_resolve_dest_handles_torch_home_placeholder(monkeypatch):
    monkeypatch.setenv("TORCH_HOME", "/tmp/fake-torch-home")
    got = FA._resolve_dest("<TORCH_HOME>/hub/checkpoints/x.pth")
    assert str(got) == "/tmp/fake-torch-home/hub/checkpoints/x.pth"


def test_resolve_dest_relative_path_is_repo_relative():
    got = FA._resolve_dest("models/foo/bar.pth")
    assert got == FA.REPO_ROOT / "models" / "foo" / "bar.pth"
