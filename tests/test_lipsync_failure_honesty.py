"""A lip-sync that produced nothing must not report success.

Two defects, both from one measured run (2026-10-17, MuseTalk shelled into the
wrong interpreter):

  1. `run_lipsync` logged `lipsync[musetalk]: done -> <path>` at INFO with
     `(err=... rc=1)` as a trailing parenthetical, after a wall of upstream
     TensorFlow banner noise. The failure read as a success.
  2. The pipeline recorded `result.lipsync_mp4 = ls.out_video` regardless --
     the path the adapter was ASKED to write -- so the manifest named an mp4
     that was never created. `edit_agent` and anything else reading the
     manifest would believe in it.

A third shape is covered here too: exit 0 with no file. That is not
hypothetical -- Wav2Lip's `inference.py` muxes with `subprocess.call` and
discards the return code -- so the FILE is the arbiter, not the error field.

No GPU and no model: the adapter is stubbed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import backend.lipsync as L
from conftest import write_tone


class _StubAdapter:
    """Pretends to be a lip-sync adapter; writes a file, or doesn't."""

    def __init__(self, *, write: bool, error=None):
        self._write, self._error = write, error
        self.model_name = "stub"

    def sync(self, request):
        if self._write:
            Path(request.out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(request.out_path).write_bytes(b"\0" * 2048)
        return type("R", (), {
            "model_name": "stub", "duration_s": 1.0, "fps": 25.0,
            "error": self._error, "extra": {},
        })()


@pytest.fixture
def job(tmp_path):
    src = tmp_path / "src.mp4"; src.write_bytes(b"\0" * 1024)
    wav = write_tone(tmp_path / "drive.wav", 1.0, sample_rate=16000)
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"segments": [], "extra": {}}))
    return src, wav, merged, tmp_path / "out"


def _run(monkeypatch, job, *, write, error=None):
    src, wav, merged, out = job
    monkeypatch.setattr(L, "get_lipsync", lambda m, **k: _StubAdapter(write=write, error=error))
    monkeypatch.setattr(L, "available_models", lambda: ["stub"])
    monkeypatch.setattr(L, "pick_devices", lambda n: [])
    # [] now means "no card clears the 12 GiB floor"; these stubs
    # are CPU-path tests, so say there are no GPUs at all.
    monkeypatch.setattr(L, "visible_cuda_devices", lambda: [])
    monkeypatch.setattr(L, "_needs_remux", lambda p: False)
    return L.run_lipsync(merged, model="stub", source_video=src,
                         driving_audio=wav, out_dir=out)


def test_a_real_render_still_says_done(monkeypatch, job, caplog):
    with caplog.at_level(logging.INFO, logger="backend.lipsync"):
        out = _run(monkeypatch, job, write=True)
    assert out.error is None
    assert Path(out.out_video).is_file()
    assert any("done →" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_an_adapter_error_is_logged_as_an_error_not_done(monkeypatch, job, caplog):
    with caplog.at_level(logging.INFO, logger="backend.lipsync"):
        out = _run(monkeypatch, job, write=False, error="stub rc=1: boom")
    assert out.error
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a failed render logged nothing at ERROR"
    assert any("FAILED" in m for m in errors)
    assert not any("done →" in r.getMessage() for r in caplog.records), \
        "'done' must never appear for a failed render"


def test_exit_zero_with_no_file_becomes_an_error(monkeypatch, job, caplog):
    """The Wav2Lip shape: rc=0, no output, hours of GPU work lost."""
    with caplog.at_level(logging.INFO, logger="backend.lipsync"):
        out = _run(monkeypatch, job, write=False, error=None)
    assert out.error, "silent no-output must be promoted to an error"
    assert "wrote no output" in out.error
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_the_merged_json_records_whether_a_file_exists(monkeypatch, job):
    src, wav, merged, out = job
    _run(monkeypatch, job, write=False, error="boom")
    extra = json.loads(merged.read_text())["extra"]["lipsync"]
    assert extra["produced"] is False
    assert extra["error"]


def test_produced_flag_rides_on_the_output(monkeypatch, job):
    out = _run(monkeypatch, job, write=True)
    assert out.extra.get("produced") is True


# --- the pipeline half ----------------------------------------------------

def test_pipeline_gates_lipsync_mp4_on_the_file_existing():
    """`ls.out_video` is the requested path, not proof of a render."""
    src = (Path(__file__).resolve().parent.parent
           / "backend" / "pipeline.py").read_text()
    assert "if ls.out_video and Path(ls.out_video).is_file():" in src, \
        "the existence gate before out.lipsync_mp4 is gone"
    gate = src.index("if ls.out_video and Path(ls.out_video).is_file():")
    assign = src.index("out.lipsync_mp4 = ls.out_video")
    assert gate < assign, "the assignment is no longer guarded by the gate"


# --- frame parity: a lip-sync must not change the frame count -------------

def test_frame_count_parses_the_csv_trailing_comma(tmp_path, monkeypatch):
    """`-of csv=p=0` emits "960,"; int() on that raises.

    The except would then answer 0, which reads as "cannot probe" and skips
    the parity check silently -- the check would be dead code.
    """
    import backend.lipsync as L
    import subprocess as sp
    monkeypatch.setattr(L, "_ffprobe_bin", lambda: "/bin/true", raising=False)
    import backend.media_io as M
    monkeypatch.setattr(M, "_ffprobe_bin", lambda: "/bin/true")
    monkeypatch.setattr(
        L.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(a, 0, stdout="960,\n", stderr=""))
    assert L._frame_count(tmp_path / "x.mp4") == 960


def test_unprobeable_video_skips_the_check_rather_than_failing(tmp_path, monkeypatch):
    """No ffprobe must not turn into a false "frames lost" verdict."""
    import backend.lipsync as L
    import backend.media_io as M
    monkeypatch.setattr(M, "_ffprobe_bin", lambda: None)
    assert L._frame_count(tmp_path / "x.mp4") == 0


def test_a_dropped_frame_render_is_rejected(monkeypatch, job, caplog):
    """The real defect: MuseTalk deletes frames it finds no face in.

    Measured 2026-10-18: source 960 frames / 40.040 s, output 950 / 40.499 s,
    with 12-20 s byte-identical frame to frame while the source moves. The
    file played and nothing errored, so only counting frames catches it.
    """
    import logging
    import backend.lipsync as L
    counts = iter([960, 950])            # source, then output
    monkeypatch.setattr(L, "_frame_count", lambda p: next(counts))
    # This test is about the PARITY verdict, not the conform that now runs
    # just before it; stub the conform so it does not consume the iterator.
    monkeypatch.setattr(L, "_conform_to_source", lambda *a, **k: 0)
    with caplog.at_level(logging.INFO, logger="backend.lipsync"):
        out = _run(monkeypatch, job, write=True)
    assert out.error and "frames" in out.error
    assert any("FRAME PARITY FAILED" in r.getMessage() for r in caplog.records)


def test_equal_frame_counts_pass(monkeypatch, job, caplog):
    import logging
    import backend.lipsync as L
    counts = iter([960, 960])
    monkeypatch.setattr(L, "_frame_count", lambda p: next(counts))
    monkeypatch.setattr(L, "_conform_to_source", lambda *a, **k: 0)
    with caplog.at_level(logging.INFO, logger="backend.lipsync"):
        out = _run(monkeypatch, job, write=True)
    assert out.error is None
    assert any("frame parity ok" in r.getMessage() for r in caplog.records)
