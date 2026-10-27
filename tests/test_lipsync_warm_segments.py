"""The warm-model path: one loaded model per GPU, pieces cycled through it.

A long clip is cut into pieces to bound host RAM. The old loop paid a full
model load per piece; now each GPU's whole run of pieces goes to ONE
`sync_many` call that loads the model once and loops them, and the pieces are
sized `seg_s / num_gpus` so running one per GPU at a time does not raise peak
RAM above the single-segment bound.

These tests drive `run_lipsync` with a recording stub -- no GPU, no ffmpeg --
to pin the distribution and ordering. The RAM-sizing invariant itself is
asserted against the source in test_musetalk_frame_parity.
"""
import json
import logging
from pathlib import Path

import pytest

import backend.lipsync as L
from conftest import write_tiny_mp4, write_tone


class _RecordingStub:
    """A lip-sync adapter that records each warm batch it is handed."""

    instances: list = []

    def __init__(self, *, cuda_device=None, **_kw):
        self.cuda_device = cuda_device
        self.batches: list = []
        _RecordingStub.instances.append(self)

    def sync(self, request):
        return self.sync_many([request])[0]

    def sync_many(self, requests):
        self.batches.append([Path(r.out_path).name for r in requests])
        out = []
        for r in requests:
            Path(r.out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(r.out_path).write_bytes(b"\0" * 2048)
            out.append(type("R", (), {
                "model_name": "stub", "duration_s": 1.0, "fps": 25.0,
                "error": None, "extra": {}})())
        return out


@pytest.fixture
def job(tmp_path):
    src = write_tiny_mp4(tmp_path / "src.mp4")
    wav = write_tone(tmp_path / "drive.wav", 1.0, sample_rate=16000)
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"segments": [], "extra": {}}))
    return src, wav, merged, tmp_path / "out"


def _run_segmented(monkeypatch, job, *, gpus):
    """Force the segmented path with `gpus` cards and a recording stub."""
    src, wav, merged, out_dir = job
    _RecordingStub.instances = []
    split_calls = {}
    concat_calls = {}

    def _fake_split(video, audio, n, out_dir, *, min_chunk_s=15.0):
        split_calls["n"] = n
        return [(Path(out_dir) / f"part{i:02d}.mp4",
                 Path(out_dir) / f"part{i:02d}.wav") for i in range(n)]

    def _fake_concat(parts, out, *, expected, audio=None):
        concat_calls["order"] = [Path(p).name for p in parts]
        concat_calls["audio"] = audio
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"\0" * 4096)
        return Path(out)

    monkeypatch.setattr(L, "get_lipsync", lambda m, **k: _RecordingStub(**k))
    monkeypatch.setattr(L, "available_models", lambda: ["stub"])
    monkeypatch.setattr(L, "pick_devices", lambda n: list(gpus))
    monkeypatch.setattr(L, "visible_cuda_devices", lambda: list(gpus))
    monkeypatch.setattr(L, "probe_duration", lambda p: 180.0)
    monkeypatch.setattr(L, "split_video_audio", _fake_split)
    monkeypatch.setattr(L, "_conform_to_source", lambda *a, **k: 0)
    monkeypatch.setattr(L, "_frame_count", lambda p: 100)
    monkeypatch.setattr(L, "_stream_durations", lambda p: (180.0, 180.0))
    monkeypatch.setattr(L, "_probe_video_fps_dur", lambda p: (25.0, 180.0))
    monkeypatch.setattr(L, "_concat_shards_verified", _fake_concat)
    monkeypatch.setattr(L, "_needs_remux", lambda p: False)

    out = L.run_lipsync(merged, model="stub", source_video=src,
                        driving_audio=wav, out_dir=out_dir)
    return out, split_calls, concat_calls


def test_pieces_shrink_with_gpu_count(monkeypatch, job):
    """`piece_s = seg_s / g`, so two GPUs cut twice as many pieces.

    seg_s defaults to 60; a 180 s clip is 3 pieces on one GPU (60 s each) and
    6 on two (30 s each) -- which is what keeps two pieces in flight summing
    to one seg_s of frames rather than two.
    """
    _out, one_gpu, _c = _run_segmented(monkeypatch, job, gpus=[0])
    assert one_gpu["n"] == 3
    _out, two_gpu, _c = _run_segmented(monkeypatch, job, gpus=[0, 1])
    assert two_gpu["n"] == 6


def test_model_loads_once_per_gpu_not_once_per_piece(monkeypatch, job):
    """One warm adapter per GPU, each handed a batch of pieces."""
    _out, _s, _c = _run_segmented(monkeypatch, job, gpus=[0, 1])
    # One adapter instance per GPU -- NOT one per piece.
    assert len(_RecordingStub.instances) == 2
    devices = sorted(s.cuda_device for s in _RecordingStub.instances)
    assert devices == [0, 1]
    # Each GPU's adapter got ONE warm call (sync_many), not three.
    for s in _RecordingStub.instances:
        assert len(s.batches) == 1
        assert len(s.batches[0]) == 3            # 6 pieces / 2 GPUs


def test_pieces_are_distributed_round_robin_and_reassembled_in_order(
        monkeypatch, job):
    """GPU i takes pieces i, i+g, i+2g...; the concat sees global order."""
    _out, _s, concat = _run_segmented(monkeypatch, job, gpus=[0, 1])
    by_device = {s.cuda_device: s.batches[0] for s in _RecordingStub.instances}
    assert by_device[0] == ["seg00.lipsync.mp4", "seg02.lipsync.mp4",
                            "seg04.lipsync.mp4"]
    assert by_device[1] == ["seg01.lipsync.mp4", "seg03.lipsync.mp4",
                            "seg05.lipsync.mp4"]
    # The join is in global time order regardless of which GPU rendered each.
    assert concat["order"] == [f"seg{i:02d}.lipsync.mp4" for i in range(6)]
    # The dub is handed to the concat so it joins video-only and remuxes the
    # known-good audio -- never decoding the pieces' own (upstream) AAC.
    assert concat["audio"] is not None


def test_a_failed_piece_stops_the_join(monkeypatch, job):
    """One lost piece means the clip would have a gap; fail rather than ship."""
    src, wav, merged, out_dir = job

    class _Flaky(_RecordingStub):
        def sync_many(self, requests):
            out = []
            for r in requests:
                name = Path(r.out_path).name
                if name == "seg03.lipsync.mp4":
                    out.append(type("R", (), {
                        "model_name": "stub", "duration_s": 0.0, "fps": 0.0,
                        "error": "boom", "extra": {}})())
                    continue
                Path(r.out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(r.out_path).write_bytes(b"\0" * 2048)
                out.append(type("R", (), {
                    "model_name": "stub", "duration_s": 1.0, "fps": 25.0,
                    "error": None, "extra": {}})())
            return out

    monkeypatch.setattr(L, "get_lipsync", lambda m, **k: _Flaky(**k))
    monkeypatch.setattr(L, "available_models", lambda: ["stub"])
    monkeypatch.setattr(L, "pick_devices", lambda n: [0, 1])
    monkeypatch.setattr(L, "visible_cuda_devices", lambda: [0, 1])
    monkeypatch.setattr(L, "probe_duration", lambda p: 180.0)
    monkeypatch.setattr(
        L, "split_video_audio",
        lambda v, a, n, od, *, min_chunk_s=15.0: [
            (Path(od) / f"part{i:02d}.mp4", Path(od) / f"part{i:02d}.wav")
            for i in range(n)])
    monkeypatch.setattr(L, "_conform_to_source", lambda *a, **k: 0)
    monkeypatch.setattr(L, "_frame_count", lambda p: 100)
    monkeypatch.setattr(L, "_needs_remux", lambda p: False)

    out = L.run_lipsync(merged, model="stub", source_video=src,
                        driving_audio=wav, out_dir=out_dir)
    assert out.error and "failed" in out.error


# --- the base default --------------------------------------------------------

def test_base_sync_many_maps_sync_in_order():
    """An adapter with no warm override still works: one sync() per request,
    results in input order."""
    from models.lipsync.base import BaseLipSync, LipSyncRequest, LipSyncResult

    class _Seq(BaseLipSync):
        name = "seq"

        def __init__(self):
            super().__init__(device="cpu")

        def _ensure_loaded(self):
            self._loaded = True

        def sync(self, request):
            return LipSyncResult(model_name="seq",
                                 out_video=Path(request.out_path),
                                 duration_s=0.0, fps=0.0)

    reqs = [LipSyncRequest(video=Path(f"{i}.mp4"), audio=Path(f"{i}.wav"),
                           out_path=Path(f"o{i}.mp4")) for i in range(3)]
    got = _Seq().sync_many(reqs)
    assert [r.out_video.name for r in got] == ["o0.mp4", "o1.mp4", "o2.mp4"]
