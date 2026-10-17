"""Does the player get handed the right file?

Both failures this guards against were found by running the resolver over the
real `data/jobs/` folders, and neither would have raised anything:

  * `video.lipsync` resolved to `lipsync/spliced.noaudio.mp4` — a SILENT
    intermediate — while the finished `final.hi.musetalk.mp4` sat at the job
    root. The user gets a video with no sound and no error.
  * `audio.original` resolved to `clip.dub.48k.stereo.wav`, which is the
    DUB. The "original audio" button plays the dubbed track. This is the one
    failure here that actively misleads rather than merely disappoints, and
    it is invisible unless you know what the clip is supposed to sound like.

So the tests below are mostly about what must NOT be chosen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ui import media as M

REPO = Path(__file__).resolve().parent.parent


def _job(tmp_path: Path, files, manifest=None) -> Path:
    """Build a job folder from a list of relative paths."""
    root = tmp_path / "jobs" / "J"
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0" * 64)
    root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _resolve(root: Path):
    return M.resolve(root, data_dir=root.parent.parent)


# ---------------------------------------------------------------------------
# what must not be chosen
# ---------------------------------------------------------------------------

def test_silent_intermediate_is_never_the_render(tmp_path):
    root = _job(tmp_path, [
        "lipsync/spliced.noaudio.mp4",
        "final.hi.musetalk.asdgated.mp4",
    ])
    got = _resolve(root)["video"]["lipsync"]
    assert got["name"] == "final.hi.musetalk.asdgated.mp4"


def test_per_shot_parts_are_never_the_render(tmp_path):
    root = _job(tmp_path, [
        "lipsync/shot0.musetalk.mp4", "lipsync/shot1.musetalk.mp4",
        "x.musetalk.lipsync.mp4",
    ])
    assert _resolve(root)["video"]["lipsync"]["name"] == "x.musetalk.lipsync.mp4"


def test_gpu_shards_are_never_the_render(tmp_path):
    root = _job(tmp_path, [
        "lipsync/out.shards/part00.lipsync.mp4",
        "lipsync/out.shards/part01.lipsync.mp4",
        "lipsync/x.musetalk.lipsync.mp4",
    ])
    assert _resolve(root)["video"]["lipsync"]["name"] == "x.musetalk.lipsync.mp4"


def test_original_audio_is_never_the_dub(tmp_path):
    """`clip.dub.48k.stereo.wav` matches a naive `*.48k.stereo.wav` glob."""
    root = _job(tmp_path, [
        "audio/clip.dub.48k.stereo.wav",
        "audio/clip.en.16k.mono.wav",
    ])
    got = _resolve(root)["audio"]["original"]
    assert got["name"] == "clip.en.16k.mono.wav"
    assert got["full_quality"] is False


def test_original_audio_is_never_a_revoiced_track(tmp_path):
    root = _job(tmp_path, [
        "audio/clip.dub.48k.stereo.revoiced.wav",
        "audio/clip.en.16k.mono.wav",
    ])
    assert _resolve(root)["audio"]["original"]["name"] == "clip.en.16k.mono.wav"


# ---------------------------------------------------------------------------
# the ranking
# ---------------------------------------------------------------------------

def test_restored_render_outranks_the_bare_one(tmp_path):
    root = _job(tmp_path, [
        "lipsync/x.wav2lip.lipsync.mp4",
        "lipsync/x.wav2lip.codeformer.mp4",
    ])
    assert _resolve(root)["video"]["lipsync"]["name"] == "x.wav2lip.codeformer.mp4"


def test_remuxed_render_outranks_the_bare_one(tmp_path):
    root = _job(tmp_path, [
        "lipsync/x.wav2lip.lipsync.mp4",
        "lipsync/x.wav2lip.lipsync.remux.mp4",
    ])
    assert _resolve(root)["video"]["lipsync"]["name"] == "x.wav2lip.lipsync.remux.mp4"


def test_the_manifest_wins_over_a_glob(tmp_path):
    """The record beats a guess — that is the whole point of recording it."""
    root = _job(tmp_path,
                ["lipsync/x.wav2lip.codeformer.mp4", "lipsync/x.wav2lip.lipsync.mp4"],
                manifest={"result": {"lipsync_mp4": "lipsync/x.wav2lip.lipsync.mp4"}})
    assert _resolve(root)["video"]["lipsync"]["name"] == "x.wav2lip.lipsync.mp4"


def test_merged_json_out_video_is_rerooted_when_the_job_moved(tmp_path):
    """`extra.lipsync.out_video` is absolute; a copied job folder must still
    resolve rather than silently reporting no render."""
    root = _job(tmp_path, ["lipsync/x.musetalk.lipsync.mp4"])
    merged = root / "mt" / "x.bodhan.hi.mt.json"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text(json.dumps({"segments": [], "extra": {"lipsync": {
        "out_video": "/somewhere/else/entirely/x.musetalk.lipsync.mp4"}}}))
    got = M.resolve(root, data_dir=root.parent.parent, merged_json=merged)
    assert got["video"]["lipsync"]["name"] == "x.musetalk.lipsync.mp4"


def test_original_video_is_the_largest_in_videos(tmp_path):
    root = tmp_path / "jobs" / "J"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "preview.mp4").write_bytes(b"\0" * 16)
    (root / "videos" / "clip.mp4").write_bytes(b"\0" * 4096)
    assert _resolve(root)["video"]["original"]["name"] == "clip.mp4"


def test_full_quality_stereo_is_preferred_and_flagged(tmp_path):
    root = _job(tmp_path, [
        "audio/x.44k.stereo.wav", "audio/x.16k.mono.wav",
    ])
    got = _resolve(root)["audio"]["original"]
    assert got["name"] == "x.44k.stereo.wav"
    assert got["full_quality"] is True and got["note"] is None


def test_mono_fallback_says_why_it_is_not_full_quality(tmp_path):
    root = _job(tmp_path, ["audio/x.16k.mono.wav"])
    got = _resolve(root)["audio"]["original"]
    assert got["full_quality"] is False
    assert "16 kHz mono" in got["note"]


def test_missing_artifacts_are_none_not_an_exception(tmp_path):
    root = tmp_path / "jobs" / "J"
    root.mkdir(parents=True)
    got = _resolve(root)
    assert got["video"]["lipsync"] is None
    assert got["audio"]["dubbed"] is None


def test_urls_are_files_urls(tmp_path):
    root = _job(tmp_path, ["mix/x.dubbed.wav"])
    got = _resolve(root)["audio"]["dubbed"]
    assert got["url"] == "/files/jobs/J/mix/x.dubbed.wav"
    assert got["download_url"].endswith("?download=1")


# ---------------------------------------------------------------------------
# against the real folders
# ---------------------------------------------------------------------------

JOBS = REPO / "data" / "jobs"


@pytest.mark.skipif(not JOBS.is_dir(), reason="no data/jobs on this host")
def test_no_real_job_resolves_original_audio_to_a_dub():
    folders = [d for d in JOBS.iterdir() if d.is_dir()]
    if not folders:
        pytest.skip("data/jobs is empty")
    bad = []
    for d in folders:
        got = M.resolve(d, data_dir=REPO / "data")["audio"]["original"]
        if got and ("dub" in got["name"].lower()
                    or "revoiced" in got["name"].lower()):
            bad.append((d.name, got["name"]))
    assert not bad, f"original-audio button would play the dub: {bad[:3]}"


@pytest.mark.skipif(not JOBS.is_dir(), reason="no data/jobs on this host")
def test_no_real_job_resolves_the_render_to_an_intermediate():
    folders = [d for d in JOBS.iterdir() if d.is_dir()]
    if not folders:
        pytest.skip("data/jobs is empty")
    bad = []
    for d in folders:
        got = M.resolve(d, data_dir=REPO / "data")["video"]["lipsync"]
        if got and ("noaudio" in got["name"].lower()
                    or got["name"].lower().startswith(("shot", "part"))):
            bad.append((d.name, got["name"]))
    assert not bad, f"player would be handed an intermediate: {bad[:3]}"
