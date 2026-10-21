"""Per-frame face boxes are cached and reused across runs.

Face detection is the expensive preprocessing step in both adapters:
MuseTalk runs DWPose + S3FD over every frame, Wav2Lip runs S3FD over every
frame, and both recomputed it on every run. Re-rendering the same clip --
which is what changing a voice, a tone or a batch size actually does -- paid
for the whole pass again.

The dangerous part is the KEY. Every sharded run writes `part00.mp4` and
`part01.mp4`, so a filename key would serve one job's boxes to another job's
frames: plausible boxes, a successful render, mouths in the wrong places.
That is the voice-reference trap from CLAUDE.md in a new costume, so these
tests weigh correctness of the key far more than the hit rate.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from models.lipsync import coords


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("DUBBING_COORD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DUBBING_COORD_CACHE", raising=False)


def _vid(tmp_path, name="part00.mp4", data=b"\0" * 4096):
    p = Path(tmp_path) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_round_trip(tmp_path):
    v = _vid(tmp_path)
    coords.save(v, "musetalk", {"bbox_shift": 0}, [(1, 2, 3, 4)], frames=1)
    assert coords.load(v, "musetalk", {"bbox_shift": 0}) == [(1, 2, 3, 4)]


def test_same_basename_different_file_is_a_MISS(tmp_path):
    """The whole point. Both shards of every job are called part00.mp4."""
    a = _vid(tmp_path / "jobA", data=b"a" * 4096)
    b = _vid(tmp_path / "jobB", data=b"b" * 8192)
    coords.save(a, "musetalk", {}, ["A-boxes"], frames=1)
    assert coords.load(b, "musetalk", {}) is None, \
        "one job's boxes must never be served to another job's frames"


def test_a_changed_parameter_is_a_miss(tmp_path):
    """bbox_shift moves the box, so its coords are not interchangeable."""
    v = _vid(tmp_path)
    coords.save(v, "musetalk", {"bbox_shift": 0}, ["zero"], frames=1)
    assert coords.load(v, "musetalk", {"bbox_shift": 10}) is None


def test_the_two_models_do_not_share_entries(tmp_path):
    """MuseTalk's boxes are landmark-derived, Wav2Lip's are S3FD + pads."""
    v = _vid(tmp_path)
    coords.save(v, "musetalk", {}, ["mt"], frames=1)
    assert coords.load(v, "wav2lip", {}) is None


def test_a_frame_count_mismatch_is_rejected(tmp_path):
    """The backstop for a file rewritten at identical size and mtime."""
    v = _vid(tmp_path)
    coords.save(v, "musetalk", {}, ["x"] * 100, frames=100)
    assert coords.load(v, "musetalk", {}, expect_frames=100) is not None
    assert coords.load(v, "musetalk", {}, expect_frames=250) is None


def test_a_corrupt_entry_recomputes_rather_than_raising(tmp_path):
    v = _vid(tmp_path)
    coords.save(v, "musetalk", {}, ["x"], frames=1)
    pkl, _ = coords._paths(v, "musetalk", {})
    pkl.write_bytes(b"not a pickle")
    assert coords.load(v, "musetalk", {}) is None


def test_the_cache_can_be_switched_off(tmp_path, monkeypatch):
    v = _vid(tmp_path)
    monkeypatch.setenv("DUBBING_COORD_CACHE", "0")
    assert coords.save(v, "musetalk", {}, ["x"], frames=1) is None
    assert coords.load(v, "musetalk", {}) is None


def test_a_write_failure_never_fails_the_render(tmp_path, monkeypatch):
    v = _vid(tmp_path)
    monkeypatch.setenv("DUBBING_COORD_CACHE_DIR", "/proc/nope/cannot-write")
    assert coords.save(v, "musetalk", {}, ["x"], frames=1) is None   # no raise


# --- both adapters opt in by default -------------------------------------

@pytest.mark.parametrize("module,cls", [
    ("musetalk", "MuseTalkSync"), ("wav2lip", "Wav2LipSync")])
def test_adapters_enable_the_cache_by_default(module, cls):
    import importlib
    m = importlib.import_module(f"models.lipsync.{module}_adapter")
    assert getattr(m, cls)().use_coord_cache is True


def test_wav2lip_upstream_gained_the_flags():
    """Wav2Lip had no caching; the flags are a vendored patch and a
    re-clone reverts them."""
    src = (Path(__file__).resolve().parent.parent / "third_party" / "Wav2Lip"
           / "inference.py").read_text()
    for tok in ("--use_saved_coord", "--saved_coord", "--coord_path",
                "Using saved face coordinates"):
        assert tok in src, f"{tok} missing from the vendored Wav2Lip"


def test_wav2lip_rejects_boxes_of_the_wrong_length():
    """A length mismatch must recompute, not draw one clip's mouths on
    another's frames."""
    src = (Path(__file__).resolve().parent.parent / "third_party" / "Wav2Lip"
           / "inference.py").read_text()
    assert "len(boxes) == len(images)" in src
    assert "Saved coordinates do not match this clip" in src


def test_wav2lip_caches_boxes_not_crops():
    """Crops are slices of the frames; caching them would store the whole
    video a second time."""
    src = (Path(__file__).resolve().parent.parent / "third_party" / "Wav2Lip"
           / "inference.py").read_text()
    assert "{'boxes': boxes, 'mask': face_mask}" in src


def test_musetalk_pickle_name_matches_upstreams_rule():
    """MuseTalk uses `basename.split('.')[0]` (inference.py:99), which
    truncates at the FIRST dot. Our inputs are `<stem>.dubbed.mp4`, so
    `Path.stem` yields "x.dubbed" where MuseTalk writes "x" -- and the
    cache became a silent no-op in both directions: nothing stored, and a
    staged hit never read.
    """
    import os
    from pathlib import Path as P
    name = "L3oOldViIgY_0to60.dubbed.mp4"
    assert os.path.basename(name).split(".")[0] == "L3oOldViIgY_0to60"
    assert P(name).stem == "L3oOldViIgY_0to60.dubbed"        # the trap
    src = (P(__file__).resolve().parent.parent / "models" / "lipsync"
           / "musetalk_adapter.py").read_text()
    assert 'video_abs.name.split(".")[0] + ".pkl"' in src, \
        "the staged pickle name no longer matches MuseTalk's own rule"
    assert 'staged = tdp / f"{video_abs.stem}.pkl"' not in src


def test_the_adapter_passes_saved_coord(tmp_path, monkeypatch):
    """Without it, inference.py deletes the pickle before we can copy it."""
    import subprocess as sp
    from models.lipsync import musetalk_adapter as MA
    from models.lipsync.base import LipSyncRequest
    seen = {}

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return sp.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(MA.subprocess, "run", _fake_run)
    a = MA.MuseTalkSync()
    monkeypatch.setattr(a, "_ensure_loaded", lambda: None)
    a.sync(LipSyncRequest(video=tmp_path / "v.dubbed.mp4",
                          audio=tmp_path / "a.wav",
                          out_path=tmp_path / "o.mp4"))
    assert "--saved_coord" in seen["cmd"]


# --- the guard that was missing -------------------------------------------

@pytest.mark.parametrize("rel", [
    "third_party/Wav2Lip/inference.py",
    "third_party/MuseTalk/scripts/inference.py",
])
def test_patched_vendored_files_still_compile(rel):
    """We patch both of these; a patch that does not parse kills every run.

    Measured 2026-10-19: the coord-cache patch added `global
    _face_present_mask` at the top of `face_detect`, which made the
    pre-existing `global` later in the same function illegal ("assigned to
    before global declaration"). Wav2Lip then failed at IMPORT -- every
    shard, every clip, rc=1 in under two seconds -- and the string-matching
    tests written alongside the patch all still passed, because they only
    grepped for text and never parsed the file.
    """
    from pathlib import Path as P
    f = P(__file__).resolve().parent.parent / rel
    src = f.read_text()
    try:
        compile(src, str(f), "exec")
    except SyntaxError as e:
        raise AssertionError(f"{rel} does not parse: line {e.lineno}: {e.msg}")


def test_face_present_mask_is_declared_global_exactly_once():
    """Two `global` statements either side of an assignment is the exact
    shape that broke it."""
    from pathlib import Path as P
    src = (P(__file__).resolve().parent.parent / "third_party" / "Wav2Lip"
           / "inference.py").read_text()
    assert src.count("global _face_present_mask") == 1
