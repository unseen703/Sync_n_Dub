"""A frame MuseTalk cannot lip-sync must be copied through, not deleted.

Measured 2026-10-18 on `data/jobs/20261018-005205...`, a 40 s clip:

    part00  480 frames in -> 255 out   video stream 10.64 s, audio 20.02 s
    part01  480 frames in -> 468 out   video stream 19.52 s, audio 20.35 s
    final   960 frames in -> 950 out   40.499 s against a 40.040 s source

One faceless frame truncated the rest of the shard, because
`ffmpeg -f image2 -i %08d.png` stops at the first gap in the numbering. The
container then reported the longer AUDIO duration, so the file looked
full-length while the picture froze on its last decoded frame for 9.4 s.

A second, silent defect rode along: upstream skipped placeholder frames when
building `input_latent_list`, leaving it 2(N-k) long against 2N for the frame
and coord cycles, so every mouth after the first missing face was composited
onto a shifted frame.

These tests pin the patch (indices aligned, originals written) and the two
guards that would have caught the render.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# Vendor tree moved from third_party/MuseTalk to models/lipsync/vendor/musetalk
# on 2026-10-23 (CLAUDE.md section Lipsync upstreams). Both are legal; resolve
# by existence so the test tracks whichever is live on this checkout.
_CAND = [REPO / "models" / "lipsync" / "vendor" / "musetalk" / "scripts" / "inference.py",
         REPO / "third_party" / "MuseTalk" / "scripts" / "inference.py"]
INFERENCE = next((p for p in _CAND if p.exists()), _CAND[0])


# --- the vendored patch ---------------------------------------------------

def test_placeholder_frames_are_written_not_skipped():
    """The fix itself. A re-clone of MuseTalk reverts this file."""
    src = INFERENCE.read_text()
    assert "is_placeholder_cycle[i % len(is_placeholder_cycle)]" in src, \
        "the blend loop no longer writes the original frame for a placeholder"
    assert "cv2.imwrite(f\"{result_img_save_path}/{str(i).zfill(8)}.png\",\n" \
           "                                ori_frame)" in src, \
        "the untouched original frame is no longer written"


def test_the_latent_list_keeps_one_entry_per_frame():
    """Index alignment across the four cycled lists.

    Without this `input_latent_list_cycle` is 2(N-k) while the frame and
    coord cycles are 2N, and the mouths drift onto wrong frames silently.
    """
    src = INFERENCE.read_text()
    assert "is_placeholder = []" in src
    assert "is_placeholder_cycle = is_placeholder + is_placeholder[::-1]" in src
    # The old unconditional skip must be gone.
    assert "if bbox == coord_placeholder:\n                    continue" not in src, \
        "the latent loop still drops placeholder frames"


def test_the_bare_except_is_gone():
    """A resize failure on a frame that DOES have a face is a real error.

    Swallowing it produced a truncated video with no error at all.
    """
    src = INFERENCE.read_text()
    blend = src[src.index("# Pad generated images to original video size"):]
    assert "except:\n                    continue" not in blend, \
        "the blend loop still swallows resize failures silently"


# --- the guards -----------------------------------------------------------

def test_stream_duration_helper_reads_each_stream_separately(tmp_path, monkeypatch):
    """The container reports the LONGEST stream, which is what hid this."""
    import backend.lipsync as L
    import backend.media_io as M
    import subprocess as sp
    monkeypatch.setattr(M, "_ffprobe_bin", lambda: "/bin/true")
    answers = iter(["10.635625\n", "20.020000\n"])
    monkeypatch.setattr(
        L.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(a, 0, stdout=next(answers), stderr=""))
    v, a = L._stream_durations(tmp_path / "x.mp4")
    assert (round(v, 2), round(a, 2)) == (10.64, 20.02)


def test_a_truncated_video_stream_is_rejected(monkeypatch, tmp_path):
    """video 10.64 s under audio 20.02 s = 9.4 s of held picture."""
    import logging
    import backend.lipsync as L
    from conftest import write_tiny_mp4, write_tone
    from models.tts.base import TTSResult  # noqa: F401  (import parity)

    # Reuse the stub harness from the sibling module.
    from test_lipsync_failure_honesty import _StubAdapter
    src = write_tiny_mp4(tmp_path / "src.mp4")
    wav = write_tone(tmp_path / "d.wav", 1.0, sample_rate=16000)
    merged = tmp_path / "m.json"; merged.write_text('{"segments": [], "extra": {}}')

    monkeypatch.setattr(L, "get_lipsync", lambda m, **k: _StubAdapter(write=True))
    monkeypatch.setattr(L, "available_models", lambda: ["stub"])
    monkeypatch.setattr(L, "pick_devices", lambda n: [])
    # [] now means "no card clears the 12 GiB floor"; these stubs
    # are CPU-path tests, so say there are no GPUs at all.
    monkeypatch.setattr(L, "visible_cuda_devices", lambda: [])
    monkeypatch.setattr(L, "_needs_remux", lambda p: False)
    monkeypatch.setattr(L, "_frame_count", lambda p: 100)      # parity ok
    # (rendered video, audio) then (source video, source audio)
    durs = iter([(10.64, 20.02), (20.02, 20.02)])
    monkeypatch.setattr(L, "_stream_durations", lambda p: next(durs))

    with pytest.MonkeyPatch.context():
        out = L.run_lipsync(merged, model="stub", source_video=src,
                            driving_audio=wav, out_dir=tmp_path / "o")
    assert out.error and "freeze" in out.error and "short" in out.error


def test_matching_streams_pass(monkeypatch, tmp_path):
    import backend.lipsync as L
    from conftest import write_tiny_mp4, write_tone
    from test_lipsync_failure_honesty import _StubAdapter
    src = write_tiny_mp4(tmp_path / "src.mp4")
    wav = write_tone(tmp_path / "d.wav", 1.0, sample_rate=16000)
    merged = tmp_path / "m.json"; merged.write_text('{"segments": [], "extra": {}}')
    monkeypatch.setattr(L, "get_lipsync", lambda m, **k: _StubAdapter(write=True))
    monkeypatch.setattr(L, "available_models", lambda: ["stub"])
    monkeypatch.setattr(L, "pick_devices", lambda n: [])
    # [] now means "no card clears the 12 GiB floor"; these stubs
    # are CPU-path tests, so say there are no GPUs at all.
    monkeypatch.setattr(L, "visible_cuda_devices", lambda: [])
    monkeypatch.setattr(L, "_needs_remux", lambda p: False)
    monkeypatch.setattr(L, "_frame_count", lambda p: 100)
    durs = iter([(20.0, 20.02), (20.0, 20.02)])
    monkeypatch.setattr(L, "_stream_durations", lambda p: next(durs))
    out = L.run_lipsync(merged, model="stub", source_video=src,
                        driving_audio=wav, out_dir=tmp_path / "o")
    assert out.error is None


def test_sharded_parity_is_checked_before_the_concat():
    """The concat re-encoded 723 real frames into 950 and hid the loss, so
    the verdict has to be reached per shard while the shards still exist."""
    src = (REPO / "backend" / "lipsync.py").read_text()
    # Scope to the sharded function: `_concat_shards_verified` is DEFINED
    # earlier in the file than it is called, so a whole-file index finds the
    # `def` and compares the wrong two positions.
    shard_fn = src[src.index("def _run_lipsync_sharded"):]
    gate = shard_fn.index("frame count still wrong after conform")
    concat = shard_fn.index("_concat_shards_verified(")
    assert gate < concat, "the per-shard check runs after the concat"


# --- the adapter's debug hook --------------------------------------------

def test_keep_work_dir_is_opt_in():
    """It copies MuseTalk's scratch dir out of the TemporaryDirectory, which
    is the only way to learn WHICH frames had no face after the fact."""
    import inspect
    from models.lipsync.musetalk_adapter import MuseTalkSync
    p = inspect.signature(MuseTalkSync).parameters["keep_work_dir"]
    assert p.default is False
    assert MuseTalkSync(keep_work_dir=True).keep_work_dir is True


# --- conform: pad a short render with the untouched source tail ----------

def test_conform_is_a_noop_when_lengths_already_match(tmp_path, monkeypatch):
    import backend.lipsync as L
    monkeypatch.setattr(L, "_frame_count", lambda p: 480)
    assert L._conform_to_source(tmp_path / "r.mp4", tmp_path / "s.mp4",
                                tmp_path / "a.wav", model="musetalk") == 0


def test_conform_trims_a_small_audio_driven_overshoot(tmp_path, monkeypatch):
    """A render LONGER than its source is trimmed back to the source.

    Wav2Lip and MuseTalk render one frame per audio chunk, so when the dub
    outlasts the picture the render runs past the last real frame -- and
    time-sharding hands that whole overhang to the final shard. Measured
    2026-10-19: shard 1 exact at 480, shard 2 at 488 for the same 480-frame
    source, the extra 0.334s matching the audio tail exactly.

    This used to return 0 and leave the mismatch to the parity check, which
    then failed the run with "frames still missing" -- reporting a surplus
    as a shortage. It discarded three batches of correct renders before it
    was fixed.
    """
    import backend.lipsync as L
    counts = iter([480, 488])          # source, rendered
    monkeypatch.setattr(L, "_frame_count", lambda p: next(counts, 480))
    monkeypatch.setattr(L, "_ffmpeg_bin", lambda: "/bin/true")
    monkeypatch.setattr(L, "_ff_run", lambda cmd, what: None)
    monkeypatch.setattr(L, "_remux_audio", lambda v, a, o, **kw: o)
    n = L._conform_to_source(tmp_path / "r.mp4", tmp_path / "s.mp4",
                             tmp_path / "a.wav", model="musetalk")
    assert n == -8, "an 8-frame overshoot should report as -8 (trimmed)"


def test_a_longer_render_is_never_an_error(tmp_path, monkeypatch):
    """Extra frames are NOT a failure. Only missing frames are.

    A render longer than its source is the dub outlasting the picture: the
    surplus frames have no source frame behind them, so nothing has been
    lost by their being there. The old code failed the whole run on it --
    and because the message said "frames still missing", three separate
    investigations went looking for dropped frames that never existed.
    """
    import backend.lipsync as L
    src = (REPO / "backend" / "lipsync.py").read_text()
    shard_fn = src[src.index("def _run_lipsync_sharded"):]
    gate = shard_fn[shard_fn.index("short.append") - 400:
                    shard_fn.index("short.append")]
    assert "out_n < src_n" in gate, (
        "the shard gate must flag only a SHORTFALL; `out_n != src_n` fails "
        "a render that merely ran long")


def test_conform_selects_by_frame_not_by_time(tmp_path, monkeypatch):
    """A re-encoded `-ss/-to` cut overshot by 22 frames when first tried
    (460 + 42 for a 20-frame request): the rendered file's nominal fps and
    its real frame spacing disagree. Frame selection is exact."""
    import backend.lipsync as L
    calls = []
    # source, rendered, then the post-concat verification read. A 2-item
    # iterator StopIteration'd the moment the conform gained a third call.
    counts = iter([480, 460])
    monkeypatch.setattr(L, "_frame_count", lambda p: next(counts, 480))
    monkeypatch.setattr(L, "_ffmpeg_bin", lambda: "/bin/true")
    monkeypatch.setattr(L, "_ff_run", lambda cmd, what: calls.append(cmd))
    monkeypatch.setattr(L, "_video_dims", lambda p: (960, 540))
    monkeypatch.setattr(L, "_frame_rate", lambda p: "24000/1001")
    monkeypatch.setattr(L, "concat_videos", lambda parts, out: out)
    monkeypatch.setattr(L, "_remux_audio", lambda v, a, o, **kw: o)
    n = L._conform_to_source(tmp_path / "r.mp4", tmp_path / "s.mp4",
                             tmp_path / "a.wav", model="musetalk")
    assert n == 20
    tail = [c for c in calls if any("select=" in str(x) for x in c)]
    assert tail, "the tail is not cut with a frame-select filter"
    flat = " ".join(tail[0])
    assert "gte(n" in flat and "460" in flat, "selection does not start at frame 460"
    assert "-frames:v 20" in flat.replace("'", ""), "exact frame count not requested"
    # No source audio may reach the dub. `-an` drops it; the concat call
    # achieves the same by mapping only the video output of the filter
    # graph, so either is acceptable.
    for c in calls:
        if "-c:v" not in c:
            continue
        assert "-an" in c or "[v]" in " ".join(c), (
            "a video call neither drops audio nor maps video only: "
            + " ".join(c))


# --- MuseTalk exits 0 on a crash; the adapter must not lose the reason ---

def test_no_mp4_error_carries_musetalks_swallowed_message(tmp_path, monkeypatch):
    """`scripts/inference.py:288` catches Exception and only PRINTS it, so a
    real crash still exits 0. Reporting just "produced no mp4" threw away the
    only evidence -- measured 2026-10-18, a 15-minute render failed with no
    recoverable reason."""
    import subprocess as sp
    from models.lipsync import musetalk_adapter as MA

    stdout = ("loading models\n"
              "Error occurred during processing: boom in the blend loop\n")
    monkeypatch.setattr(MA.subprocess, "run",
                        lambda *a, **k: sp.CompletedProcess(a, 0, stdout, ""))
    a = MA.MuseTalkSync()
    monkeypatch.setattr(a, "_ensure_loaded", lambda: None)
    from models.lipsync.base import LipSyncRequest
    r = a.sync(LipSyncRequest(video=tmp_path / "v.mp4",
                              audio=tmp_path / "a.wav",
                              out_path=tmp_path / "o.mp4"))
    assert r.error
    assert "Error occurred during processing: boom in the blend loop" in r.error
    assert "exited 0" in r.error, "the exit-0 nature must be stated"


# --- sharded path: conform FIRST, then judge ------------------------------

def test_shards_are_conformed_before_the_parity_check():
    """Order matters, and getting it wrong is a regression I shipped.

    A shard is routinely short -- MuseTalk renders `len(whisper_chunks)`
    frames, a count driven by the AUDIO. Measured 2026-10-18 on a 37 s clip:
    425 of 444 and 436 of 444. With the parity check placed FIRST the run
    was rejected outright, turning a recoverable shortfall into a hard
    failure and making sharded jobs worse than before the fix.
    """
    src = (REPO / "backend" / "lipsync.py").read_text()
    shard_fn = src[src.index("def _run_lipsync_sharded"):]
    conform = shard_fn.index("_conform_to_source(Path(r.out_video)")
    verdict = shard_fn.index("frame count still wrong after conform")
    concat = shard_fn.index("_concat_shards_verified(")
    assert conform < verdict < concat, (
        "order must be conform -> parity check -> concat")


def test_each_shard_conforms_against_its_own_part():
    """Conforming once on the concatenated file would splice shard 1's
    source tail after shard 2, i.e. out of order."""
    src = (REPO / "backend" / "lipsync.py").read_text()
    shard_fn = src[src.index("def _run_lipsync_sharded"):]
    assert "_conform_to_source(Path(r.out_video), pair[0], pair[1]," in shard_fn, \
        "the shard conform no longer uses that shard's own video/audio pair"


def test_conform_muxes_without_shortest(tmp_path, monkeypatch):
    """`-shortest` truncates the OUTPUT to the shorter input.

    Measured 2026-10-19: a shard conformed 690 -> 720 came back 717, because
    the soundtrack was ~3 frames shorter than the picture and the mux cut the
    video back down -- silently undoing the frames the conform had just
    restored. The whole point of the conform is that the picture outlives
    the audio, so it must mux with shortest=False.
    """
    import backend.lipsync as L
    seen = {}
    counts = iter([480, 460])
    monkeypatch.setattr(L, "_frame_count", lambda p: next(counts, 480))
    monkeypatch.setattr(L, "_ffmpeg_bin", lambda: "/bin/true")
    monkeypatch.setattr(L, "_ff_run", lambda cmd, what: None)
    monkeypatch.setattr(L, "_video_dims", lambda p: (960, 540))
    monkeypatch.setattr(L, "_frame_rate", lambda p: "24000/1001")
    monkeypatch.setattr(L, "concat_videos", lambda parts, out: out)
    monkeypatch.setattr(L, "_remux_audio",
                        lambda v, a, o, **kw: seen.update(kw))
    L._conform_to_source(tmp_path / "r.mp4", tmp_path / "s.mp4",
                         tmp_path / "a.wav", model="musetalk")
    assert seen.get("shortest") is False, \
        "the conform must not let a short soundtrack trim the picture"


def test_remux_default_still_uses_shortest():
    """Unchanged for every other caller: a dub should not outrun its video."""
    import inspect
    import backend.lipsync as L
    assert inspect.signature(L._remux_audio).parameters["shortest"].default is True


def test_a_long_dub_over_a_correct_render_is_not_a_failure(monkeypatch, tmp_path):
    """Regression. Measured 2026-10-19: source 60.02 s / 1439 frames, render
    conformed to 1439 -- correct -- but the dubbed track was 62.22 s because
    the TTS overran its slot. Comparing the picture against the AUDIO
    announced a 2.2 s freeze that did not exist and failed a good render.
    The reference is the SOURCE; `_remux_audio` already trims the overrun.
    """
    import logging
    import backend.lipsync as L
    from conftest import write_tiny_mp4, write_tone
    from test_lipsync_failure_honesty import _StubAdapter
    src = write_tiny_mp4(tmp_path / "src.mp4")
    wav = write_tone(tmp_path / "d.wav", 1.0, sample_rate=16000)
    merged = tmp_path / "m.json"; merged.write_text('{"segments": [], "extra": {}}')
    monkeypatch.setattr(L, "get_lipsync", lambda m, **k: _StubAdapter(write=True))
    monkeypatch.setattr(L, "available_models", lambda: ["stub"])
    monkeypatch.setattr(L, "pick_devices", lambda n: [])
    monkeypatch.setattr(L, "visible_cuda_devices", lambda: [])
    monkeypatch.setattr(L, "_needs_remux", lambda p: False)
    monkeypatch.setattr(L, "_conform_to_source", lambda *a, **k: 0)
    monkeypatch.setattr(L, "_frame_count", lambda p: 1439)
    # rendered: 60.02 video / 62.22 audio;  source: 60.02 video
    durs = iter([(60.02, 62.22), (60.02, 60.00)])
    monkeypatch.setattr(L, "_stream_durations", lambda p: next(durs))
    with caplog_at(logging.WARNING) as rec:
        out = L.run_lipsync(merged, model="stub", source_video=src,
                            driving_audio=wav, out_dir=tmp_path / "o")
    assert out.error is None, "a correct render must not be failed by a long dub"


import contextlib
import logging as _logging


@contextlib.contextmanager
def caplog_at(level):
    """Minimal capture so the test does not depend on pytest's caplog
    fixture ordering with the monkeypatches above."""
    recs = []

    class _H(_logging.Handler):
        def emit(self, r): recs.append(r)

    h = _H(level)
    lg = _logging.getLogger("backend.lipsync")
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(level)
    try:
        yield recs
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


def test_shard_concat_is_verified_not_assumed():
    """The shard join must COUNT its result, not trust the demuxer.

    `concat_videos` uses ffmpeg's concat demuxer, which silently drops an
    input whose codec parameters differ instead of failing. `_conform_to_source`
    re-encodes only the shard that needed fixing, so a mismatch between
    siblings is normal rather than exotic. Measured 2026-10-19: two 480-frame
    shards joined to 504 frames at rc=0, and the loss read as a lip-sync
    fault rather than a join fault.
    """
    src = (REPO / "backend" / "lipsync.py").read_text()
    shard_fn = src[src.index("def _run_lipsync_sharded"):]
    assert "_concat_shards_verified" in shard_fn, (
        "the sharded path must join through the verified helper, not call "
        "concat_videos directly and trust it")
    helper = src[src.index("def _concat_shards_verified"):]
    assert "concat=n=" in helper, "no concat-filter fallback for a short join"


def test_hd_stage2_is_given_the_exact_rational_frame_rate():
    """Stage 2 must re-encode at the EXACT rate, not cv2's float.

    cv2's CAP_PROP_FPS returns 23.976023976023978 for a 24000/1001 stream,
    and upstream Wav2Lip-HD is worse -- `video2frames.py` does
    `int(video.get(cv2.CAP_PROP_FPS))`, truncating it to 23, and
    `run_final.sh` then recombines at a hardcoded `-r 20`.

    This repo already solved it: `_frame_rate()` returns ffprobe's
    `r_frame_rate` as a rational string. A shard re-encoded at a decimal
    approximation can end up tagged differently from its sibling, and the
    concat demuxer drops a mismatched shard WITHOUT failing -- measured
    2026-10-19 as 480 + 480 frames joining to 504.

    The rate's owner MOVED on 2026-10-23. It used to be
    `wav2lip_hd_adapter`, which probed with `_frame_rate(rendered)` and
    passed `--fps rate` down. That adapter is gone -- restoration is stage
    9i now -- and `scripts/face_enhance.py` shards itself, so it probes the
    rate once in the parent and hands the SAME rational to every child.
    Same guarantee, one layer lower.
    """
    enh = (REPO / "scripts" / "face_enhance.py").read_text()
    assert "def probe_rate(" in enh, (
        "the parent must probe r_frame_rate itself, not let cv2 guess")
    assert "r_frame_rate" in enh, "not reading ffprobe's rational"
    assert '"--fps", fps' in enh, (
        "the probed rational is not handed to the shard workers — a shard "
        "tagged differently from its sibling is dropped by the concat "
        "demuxer without an error")
    assert "a.fps or" in enh, (
        "the worker must prefer the caller's exact rate over cv2's float")


# ---------------------------------------------------------------------------
# the 2026-10-20 memory bounds: segment, cap resolution, smaller batch
# ---------------------------------------------------------------------------

def test_lipsync_segments_long_clips_by_default():
    """A long clip is cut into <=1 min pieces, rendered one at a time.

    Both vendored models hold one decoded uint8 frame per source frame for
    the whole clip -- MuseTalk's `read_imgs()`, Wav2Lip's
    `full_frames.append(frame)`. Measured 5.87 GiB + 6.14 MiB/frame at
    1080p, which matches a 1920x1080x3 frame (5.93 MiB) to within 3%: it IS
    the pixels. A 7-minute clip projects to ~70 GiB on a 62 GiB host, so
    peak RAM has to be a function of the SEGMENT, not of the clip.

    60 since 2026-10-23, lowered from 120 by request. The number is pinned
    here rather than left free because it is a HOST bound -- it answers
    this machine's 62 GiB against this model's per-frame cost, so a change
    to it should be a decision someone made, not a default that drifted.
    """
    import inspect
    import backend.lipsync as L
    sig = inspect.signature(L.run_lipsync)
    assert sig.parameters["max_segment_s"].default == 60.0, (
        "segmenting must be the default, not opt-in")
    src = (REPO / "backend" / "lipsync.py").read_text()
    assert "def _run_lipsync_segmented" in src


def test_segments_run_one_at_a_time_not_in_parallel():
    """Sequential is the whole point.

    Sharding splits frames across PROCESSES that run together, so the total
    resident set is unchanged -- it buys wall time, not memory. Only
    running the pieces one after another bounds RAM. A ThreadPoolExecutor
    inside the segment loop would silently undo the entire change.
    """
    src = (REPO / "backend" / "lipsync.py").read_text()
    fn = src[src.index("def _run_lipsync_segmented"):
             src.index("def _run_lipsync_sharded")]
    assert "ThreadPoolExecutor" not in fn, (
        "segments must not run concurrently -- that restores the peak RAM "
        "the segmenting exists to remove")
    assert "for i, (v, a) in enumerate(pairs)" in fn


def test_a_supplied_coord_list_is_split_across_segments():
    """Target boxes are indexed against the whole video and segments
    renumber frames from zero, so every segment after the first would
    animate the wrong frames -- at exit code 0, like the sharding case.

    This used to be answered by DISABLING segmenting whenever a list was
    supplied. That traded the memory bound for the targeting: segmenting is
    what keeps peak RSS a function of the segment rather than of the whole
    clip (one decoded frame per source frame, ~6 MiB at 1080p and ~24 MiB
    at 4K), so a long gated render went back on course to OOM. Since
    2026-10-23 the list is cut the same way the video is, by counted
    frames, and neither has to be given up.
    """
    src = (REPO / "backend" / "lipsync.py").read_text()
    assert "if want_segments and coord_list:" not in src, (
        "segmenting is disabled again when targets are supplied")
    assert "coord_list=seg_coords[i]" in src, (
        "segments no longer receive their own slice of the target list")
    assert "_split_coords(" in src


def test_musetalk_caps_render_height_and_batch():
    """1080p cap and batch 8, both set on 2026-10-20.

    The cap is not cosmetic: at 3840x2160 MuseTalk's S3FD returned `[None]`
    and `preprocessing.py:135` divided by zero inside a print statement,
    after the model had loaded. It also costs nothing visible -- the mouth
    is rebuilt at 256x256 whatever the frame size.
    """
    from models.lipsync.musetalk_adapter import MAX_RENDER_HEIGHT, MuseTalkSync
    import inspect
    assert MAX_RENDER_HEIGHT == 1080
    sig = inspect.signature(MuseTalkSync.__init__)
    assert sig.parameters["batch_size"].default == 8
    assert sig.parameters["max_height"].default == MAX_RENDER_HEIGHT


def test_downscale_happens_before_the_coord_cache_key():
    """Boxes cached against the 4K file must not reach a 1080p render.

    The cache key is derived from the file handed to MuseTalk, so the
    rescale has to happen first or the key names the wrong resolution and
    the staged pickle puts every box in 4K coordinates.
    """
    src = (REPO / "models" / "lipsync" / "musetalk_adapter.py").read_text()
    body = src[src.index("def sync(self, request"):]
    cap = body.index("_cap_resolution")
    key = body.index("coords.key_for") if "coords.key_for" in body else body.index("coord_params")
    assert cap < key, "resolution is capped after the coord cache key is built"


def test_supplied_boxes_follow_the_frame_down():
    """A source-space coord list must be converted, not passed through.

    This is the crash the cache test above only half-covered. The cache key
    is derived after the rescale, so a CACHED list is safe -- but an ASD-gated
    list supplied by `backend.lipsync_targets.build_coord_list` is built
    against the SOURCE video and was written into the pickle untouched.

    Measured on the IIT lecture (`svwrTOm55y4`, 3840x2160): boxes up to
    y2=2151 reached a 1920x1080 render, `frame[y1:y2, x1:x2]` sliced past the
    bottom edge, numpy returned a 0-row array and cv2 raised
    `(-215:Assertion failed) !ssize.empty()` 87 s in -- after the frames were
    extracted and the model loaded. Exit code 0, no mp4.
    """
    from models.lipsync.musetalk_adapter import _rescale_boxes

    boxes = [(1000, 1800, 1400, 2100), (0.0, 0.0, 0.0, 0.0), (200, 100, 400, 300)]
    out = _rescale_boxes(boxes, (0.5, 0.5))
    assert out[0] == (500, 900, 700, 1050)
    assert out[2] == (100, 50, 200, 150)
    # Every real box must now fit inside the 1080p frame it will be sliced
    # from -- which is the whole point.
    for x1, y1, x2, y2 in (b for b in out if tuple(b) != (0.0, 0.0, 0.0, 0.0)):
        assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080


def test_the_placeholder_survives_rescaling_bit_identical():
    """`(0.0, 0.0, 0.0, 0.0)` is MuseTalk's `coord_placeholder`, meaning
    "write this frame through unchanged" -- not a box at the origin.
    `scripts/inference.py` compares against it by equality, so scaling it to
    `(0, 0, 0, 0)` ints would stop it matching and send 783 pass-through
    frames into the crop path instead."""
    from models.lipsync.musetalk_adapter import _rescale_boxes

    out = _rescale_boxes([(0.0, 0.0, 0.0, 0.0)], (0.5, 0.5))
    assert out[0] == (0.0, 0.0, 0.0, 0.0)
    assert all(isinstance(v, float) for v in out[0])


def test_an_unscaled_render_leaves_the_boxes_alone():
    from models.lipsync.musetalk_adapter import _rescale_boxes

    boxes = [(10, 20, 30, 40), (0.0, 0.0, 0.0, 0.0)]
    assert _rescale_boxes(boxes, (1.0, 1.0)) == boxes


def test_cap_resolution_reports_the_factor_it_applied():
    """The factor is the only way a caller can convert; a bare path is what
    let the mismatch through. Every failure path must say `(1.0, 1.0)`
    rather than omitting it."""
    import inspect

    from models.lipsync.musetalk_adapter import MuseTalkSync

    src = inspect.getsource(MuseTalkSync._cap_resolution)
    returns = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith("return ")]
    assert returns, "no returns found"
    for r in returns:
        assert r != "return video", (
            "a path is returned with no scale factor -- the caller cannot "
            "convert a source-space coord list")
    # And the one that did rescale must not claim 1.0.
    assert any("return scaled, (" in r for r in returns)


def test_ui_preset_batch_tracks_the_adapter():
    """The UI preset and the adapter default must not drift apart."""
    from models.lipsync.musetalk_adapter import MuseTalkSync
    import inspect
    src = (REPO / "ui" / "presets.py").read_text()
    want = inspect.signature(MuseTalkSync.__init__).parameters["batch_size"].default
    assert f'"lipsync_batch_size": {want}' in src, (
        f"ui/presets.py should ask for batch {want}, matching the adapter")


# --- lazy decode: pin the vendor patch AND the adapter flipping it on ----

def test_lazy_decode_patch_still_present_in_vendor():
    """A vendor re-clone would silently revert the memory win.

    Measured on scripts/ab_lazy_frames.py (data/gender_bench/cuts/
    ab_lazy_frames/ab_lazy_frames.json): eager 23.88 GiB RSS vs lazy 6.84
    GiB on a 2-min 1080p clip at identical 744.7 s wall and identical
    output -- but ONLY on the --use_saved_coord branch.
    """
    src = INFERENCE.read_text()
    assert '_LAZY_FRAMES = os.environ.get("MUSETALK_LAZY_FRAMES"' in src, \
        "the lazy-frames gate is gone from vendor/musetalk/scripts/inference.py"
    assert "class _LazyFrames" in src, \
        "the _LazyFrames sequence class is gone -- a re-clone has reverted it"
    assert "_LazyFrames(input_img_list) if _LAZY_FRAMES else read_imgs(input_img_list)" in src, \
        "the hot --use_saved_coord branch no longer swaps in _LazyFrames"


def test_adapter_turns_lazy_on_by_default_on_saved_coord():
    """The win only lands if production flips the env var. An operator-set
    value wins (set MUSETALK_LAZY_FRAMES=0 to disable for an A/B)."""
    src = (REPO / "models" / "lipsync" / "musetalk_adapter.py").read_text()
    assert "MUSETALK_LAZY_FRAMES" in src, (
        "the adapter never sets MUSETALK_LAZY_FRAMES -- lazy stays off in "
        "production and the A/B's -17 GiB win does not land")
    assert '"--use_saved_coord" in cmd' in src, (
        "lazy must only default-on for the branch where the A/B shows the "
        "win -- the cold-cache branch loads every frame anyway")
