"""Stage-less ASD gating for lip-sync (Route A).

Nothing in the adapters chooses the SPEAKING face: Wav2Lip takes the first
S3FD detection and MuseTalk vendors the same code. `backend.lipsync_targets`
decides on our side and hands MuseTalk the answer through its own
`--use_saved_coord` pickle.

No models here. The landmark head and the video decode are the parts that
need a GPU; the arithmetic and the frame bookkeeping are what break silently,
so those are what these pin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.lipsync_targets import (MAX_GAP_FILL, MIN_SPEAK_RUN, PLACEHOLDER,
                                     build_coord_list, musetalk_box,
                                     speaking_track_per_frame)


# --- the box arithmetic ---------------------------------------------------

def _ibug(cx=100.0, cy=100.0, scale=1.0):
    """68 points in iBUG order, enough for the three indices that matter.

    Only the nose-bridge pivot (29), the extremes in x, and the lowest y are
    load-bearing, so the rest is filler that cannot accidentally become an
    extreme.
    """
    lm = np.full((68, 3), np.nan)
    for i in range(68):
        lm[i] = (cx, cy, 0.0)
    lm[0] = (cx - 50 * scale, cy, 0.0)          # jaw left  -> min x
    lm[16] = (cx + 50 * scale, cy, 0.0)         # jaw right -> max x
    lm[8] = (cx, cy + 60 * scale, 0.0)          # chin      -> max y
    lm[27] = (cx, cy - 30 * scale, 0.0)
    lm[28] = (cx, cy - 20 * scale, 0.0)
    lm[29] = (cx, cy - 10 * scale, 0.0)         # the pivot
    lm[30] = (cx, cy, 0.0)
    return lm


def _upstream_box(lm, bbox_shift=0):
    """`get_landmark_and_bbox` (preprocessing.py:113-129), transcribed.

    Kept as an independent transcription rather than importing ours, so the
    test fails if our copy drifts from upstream's formula.
    """
    face = np.asarray(lm)[:, :2].astype(np.int32)
    half = face[29].copy()
    if bbox_shift != 0:
        half[1] = bbox_shift + half[1]
    half_face_dist = np.max(face[:, 1]) - half[1]
    upper_bond = max(0, half[1] - half_face_dist)
    return (int(np.min(face[:, 0])), int(upper_bond),
            int(np.max(face[:, 0])), int(np.max(face[:, 1])))


def test_box_matches_upstream_formula():
    lm = _ibug()
    assert musetalk_box(lm) == _upstream_box(lm)


@pytest.mark.parametrize("shift", [-20, -7, 0, 5, 30])
def test_bbox_shift_matches_upstream_and_moves_only_the_top(shift):
    """CLAUDE.md §3: this knob moves the TOP edge, not the bottom.

    A sweep was once designed around the opposite belief and cost four hours.
    """
    lm = _ibug()
    got, want = musetalk_box(lm, bbox_shift=shift), _upstream_box(lm, shift)
    assert got == want
    base = musetalk_box(lm)
    assert got[0] == base[0] and got[2] == base[2], "x edges must not move"
    assert got[3] == base[3], "the BOTTOM edge is extra_margin's job, not this"
    if shift:
        assert got[1] != base[1], "the top edge must move"


def test_degenerate_landmarks_fall_back_to_the_detector_box():
    lm = _ibug()
    lm[:, 0] = 100.0                      # zero width -> x2-x1 <= 0
    assert musetalk_box(lm, fallback=(5, 6, 7, 8)) == (5, 6, 7, 8)
    assert musetalk_box(lm, fallback=None) is None


# --- choosing the speaking track -----------------------------------------

class _Track:
    def __init__(self, tid, frames, box=(10, 20, 110, 140)):
        self.track_id = tid
        self.frames = np.asarray(frames, dtype=int)
        self.bboxes = np.asarray([list(box)] * len(frames), dtype=float)


class _Tracks:
    def __init__(self, tracks): self.tracks = tracks


class _Score:
    def __init__(self, tid, frames, scores):
        self.track_id = tid
        self.frames = np.asarray(frames, dtype=int)
        self.scores = np.asarray(scores, dtype=float)


class _ASD:
    def __init__(self, scores): self.per_track = {s.track_id: s for s in scores}


def test_the_loudest_face_wins_a_frame():
    """One box per frame, so simultaneous speech resolves by score."""
    tracks = _Tracks([_Track("A", range(40)), _Track("B", range(40))])
    asd = _ASD([_Score("A", range(40), [0.9] * 40),
                _Score("B", range(40), [0.6] * 40)])
    got = speaking_track_per_frame(tracks, asd)
    assert set(got.values()) == {"A"}


def test_a_silent_face_is_not_chosen():
    tracks = _Tracks([_Track("A", range(40))])
    asd = _ASD([_Score("A", range(40), [0.1] * 40)])
    assert speaking_track_per_frame(tracks, asd) == {}


def test_a_brief_dip_does_not_punch_holes_in_a_sentence():
    """The reason gap-filling exists.

    Un-animated frames BETWEEN animated ones read as a mouth stutter, which
    is worse than the wrong-face problem being fixed.
    """
    scores = [0.9] * 40
    scores[20] = scores[21] = 0.1                 # a 2-frame ASD dip
    tracks = _Tracks([_Track("A", range(40))])
    asd = _ASD([_Score("A", range(40), scores)])
    got = speaking_track_per_frame(tracks, asd)
    assert got.get(20) == "A" and got.get(21) == "A", "short dip must be filled"
    assert len(got) == 40


def test_a_long_silence_is_not_filled():
    scores = [0.9] * 60
    for i in range(20, 20 + MAX_GAP_FILL + 6):
        scores[i] = 0.0
    tracks = _Tracks([_Track("A", range(60))])
    asd = _ASD([_Score("A", range(60), scores)])
    got = speaking_track_per_frame(tracks, asd)
    assert 30 not in got, "a real pause must stay untouched"


def test_a_run_too_short_to_be_a_line_is_discarded():
    scores = [0.0] * 40
    for i in range(10, 10 + MIN_SPEAK_RUN - 1):
        scores[i] = 0.95
    tracks = _Tracks([_Track("A", range(40))])
    asd = _ASD([_Score("A", range(40), scores)])
    assert speaking_track_per_frame(tracks, asd) == {}


# --- the frame bookkeeping, which is what truncates a render -------------

def test_every_frame_gets_exactly_one_entry(tmp_path, monkeypatch):
    """`zip(coord_list, frame_list)` truncates to the SHORTER list.

    A short coord list therefore silently stops considering the tail of the
    video -- the same shape as the faceless-frame bug that ended a 40 s
    render at 10.64 s.
    """
    import backend.lipsync_targets as T

    n = 30
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(n)]

    class _Cap:
        def __init__(self, *a): self.i = 0
        def read(self):
            if self.i >= n: return False, None
            self.i += 1; return True, frames[self.i - 1]
        def get(self, *a): return float(n)
        def release(self): pass

    monkeypatch.setattr(T, "_landmark_model", lambda **k: object())
    monkeypatch.setattr(T, "_landmarks_for_box", lambda *a: _ibug(60, 40, 0.3))
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", _Cap)

    tracks = _Tracks([_Track("A", range(10, 25), box=(10, 10, 70, 70))])
    asd = _ASD([_Score("A", range(10, 25), [0.9] * 15)])
    coords, stats = build_coord_list(tmp_path / "v.mp4", tracks, asd, n_frames=n)

    assert len(coords) == n, "one entry per frame, always"
    assert stats["speaking"] == 15
    assert stats["placeholder"] == 15
    assert all(tuple(c) == PLACEHOLDER for c in coords[:10]), "silence first"
    assert all(tuple(c) != PLACEHOLDER for c in coords[10:25]), "then speech"
    assert all(tuple(c) == PLACEHOLDER for c in coords[25:]), "then silence"


def test_frames_with_no_speaker_are_the_untouched_sentinel(tmp_path, monkeypatch):
    """`(0,0,0,0)` is MuseTalk's coord_placeholder, which since 2026-10-18
    means "write this frame unchanged" rather than "drop it"."""
    import backend.lipsync_targets as T
    import cv2

    class _Cap:
        def __init__(self, *a): self.i = 0
        def read(self):
            if self.i >= 5: return False, None
            self.i += 1; return True, np.zeros((40, 40, 3), dtype=np.uint8)
        def get(self, *a): return 5.0
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    monkeypatch.setattr(T, "_landmark_model", lambda **k: object())
    coords, stats = build_coord_list(tmp_path / "v.mp4", _Tracks([]), _ASD([]),
                                     n_frames=5)
    assert coords == [PLACEHOLDER] * 5
    assert stats["no_speaker"] == 5


def test_supplied_targets_are_not_written_into_the_shared_cache():
    """A target list is gated on ONE run's ASD, not a property of the file.

    Caching it would serve a speaker-specific answer to an unrelated run --
    the shape of the voice-reference trap in CLAUDE.md, where a confident
    wrong answer looks exactly like a working feature.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent
    body = (src / "models" / "lipsync" / "musetalk_adapter.py").read_text()
    assert "if not targets and self.use_coord_cache and staged.is_file():" in body


# ---------------------------------------------------------------------------
# the wiring: ASD scores -> disk -> stage 9h
# ---------------------------------------------------------------------------
#
# `build_coord_list` was correct and had NO production caller for three
# years' worth of commits: `backend/pipeline.py` called `run_lipsync` without
# `coord_list`, so MuseTalk fell back to its own first-detection face choice
# and animated whoever it found. The visible symptom was a listening person
# with a moving mouth in a two-shot. These guard the two links that fixed it.

def test_per_frame_scores_round_trip_through_disk(tmp_path):
    """Stage 6 computed these and threw them away.

    Stage 9h runs in a DIFFERENT process on an edit replay, where stage 6
    does not re-run, so in-memory was never going to be enough — and
    re-deriving them there means a second TalkNet pass over the whole video.
    """
    import numpy as np

    from backend.asd import load_scores_npz, save_scores_npz
    from models.asd import ASDResult, ASDScore

    src = ASDResult(
        video_path=tmp_path / "v.mp4", model_name="talknet", fps=25.0,
        per_track={
            "FACE_01": ASDScore(track_id="FACE_01",
                                frames=np.array([0, 1, 2], dtype=np.int32),
                                scores=np.array([0.1, 0.9, 0.8],
                                                dtype=np.float32)),
            "FACE_02": ASDScore(track_id="FACE_02",
                                frames=np.array([1, 2], dtype=np.int32),
                                scores=np.array([0.7, 0.2], dtype=np.float32)),
        })
    back = load_scores_npz(save_scores_npz(src, tmp_path / "a.scores.npz"))

    assert back.model_name == "talknet" and back.fps == 25.0
    assert sorted(back.per_track) == ["FACE_01", "FACE_02"]
    for tid, orig in src.per_track.items():
        np.testing.assert_array_equal(back.per_track[tid].frames, orig.frames)
        np.testing.assert_allclose(back.per_track[tid].scores, orig.scores,
                                   rtol=1e-6)


def test_an_empty_score_set_round_trips(tmp_path):
    """A video where no track was scored must not crash the writer: stage 6
    persists unconditionally and a raise there would take the whole run."""
    from backend.asd import load_scores_npz, save_scores_npz
    from models.asd import ASDResult

    back = load_scores_npz(save_scores_npz(
        ASDResult(video_path=tmp_path / "v.mp4", model_name="talknet",
                  fps=25.0, per_track={}),
        tmp_path / "e.scores.npz"))
    assert back.per_track == {}


def test_stage_9h_passes_the_target_list_to_run_lipsync():
    """The link that was missing.

    Greps the source rather than running the stage, because running it means
    a GPU, a model download and a TalkNet pass. What is being guarded is one
    keyword argument at one call site — exactly the kind of thing a refactor
    drops without any test noticing, which is how it came to be missing in
    the first place.
    """
    import inspect

    import backend.pipeline as P

    src = inspect.getsource(P.run_post_mt_stages)
    assert "coord_list=_coords" in src, (
        "stage 9h no longer passes the ASD-gated target list, so the "
        "lip-sync model is back to choosing a face by itself")
    assert "_asd_coord_list(" in src


def test_stage_6_persists_the_scores_stage_9h_needs():
    """The other half: without the npz, `_asd_coord_list` has nothing."""
    import inspect

    import backend.pipeline as P

    src = inspect.getsource(P)
    assert "save_scores_npz(" in src, (
        "stage 6 stopped writing per-frame ASD scores, so stage 9h silently "
        "falls back to the model's own face choice")


def test_only_musetalk_is_offered_a_target_list():
    """Wav2Lip and LatentSync ignore `coord_list` rather than half-honouring
    it, so building one for them spends a landmark pass per frame to produce
    nothing."""
    import backend.pipeline as P

    assert P.COORD_LIST_MODELS == ("musetalk",)
    assert P._asd_coord_list(Path("/nonexistent"), Path("v.mp4"),
                             "wav2lip") is None


def test_the_lipsync_only_replay_also_gates_on_asd():
    """`POST /api/jobs/{id}/lipsync` does NOT go through stage 9h.

    `edit_agent.rerun` short-circuits a lipsync-only edit to
    `_rerun_lipsync_only`, which drives the adapter directly because the mix
    is still valid. That is the path the UI's "Start lip-sync" button takes,
    so wiring only `run_post_mt_stages` would have fixed the pipeline while
    the button everyone actually presses kept animating whoever MuseTalk
    found first.
    """
    import inspect

    from backend import edit_agent as A

    src = inspect.getsource(A._rerun_lipsync_only)
    assert "_asd_coord_list(" in src, (
        "the lipsync-only replay no longer builds ASD-gated targets")
    assert "coord_list=" in src, (
        "the lipsync-only replay no longer passes them to run_lipsync")


def test_a_window_slices_the_target_list_to_match():
    """A windowed render cuts a sub-clip that renumbers frames from zero.

    A whole-video list applied to it animates the wrong frames at exit code
    0 — the same hazard sharding and segmenting already refuse, except this
    one has an exact answer: slice the list to the window.
    """
    import inspect

    from backend import lipsync as L

    src = inspect.getsource(L.run_lipsync)
    assert "_probe_fps" in src and "_slice_coords_to_window" in src, (
        "run_lipsync stopped aligning a supplied coord_list to the window")


def test_the_window_slice_is_measured_off_the_cut_clip(tmp_path, monkeypatch):
    """The LENGTH comes from the probe, not from rounding both endpoints.

    Regression for job 20261023-011027. `round(w_end*fps) - round(w_start*
    fps)` is off by one whenever the window's two ends round in opposite
    directions, and `_split_coords` measures with the probe — so the two
    rulers disagreed, the second of two segments came up one box short, and
    the targeting was dropped for 75 s of a 150 s dub. It rendered on
    MuseTalk's own first-face pick at exit code 0.
    """
    from backend import lipsync as L

    clip = tmp_path / "window.mp4"; clip.write_bytes(b"\0")
    monkeypatch.setattr(L, "_probe_frames", lambda p: 3748)

    coords = [(i, i, i, i) for i in range(3775)]
    got = L._slice_coords_to_window(
        coords, clip, 0.6722187500000001, 150.55596875, 25.0,
        model="musetalk")

    # Arithmetic on both ends gives 3764 - 17 = 3747. The clip has 3748.
    assert len(got) == 3748, "the slice no longer matches the cut clip"
    assert got[0] == (17, 17, 17, 17), "the offset moved off the window start"

    # And the split that used to refuse now covers both halves.
    pairs = _fake_pairs(tmp_path, [1874, 1874])
    monkeypatch.setattr(L, "_probe_frames", lambda p: 1874)
    pieces = L._split_coords(got, pairs, model="musetalk", what="segment")
    assert [len(p) for p in pieces] == [1874, 1874], (
        "a segment lost its targeting — the off-by-one is back")


def test_a_rounding_shortfall_repeats_the_last_box(tmp_path, monkeypatch):
    """ffmpeg decides for itself whether the boundary frame is in the cut.

    One frame of slack is rounding, and the cheap answer is to hold the last
    box for it. The expensive answer — what this replaced — was dropping the
    targeting for every frame in the piece because one was missing.
    """
    from backend import lipsync as L

    clip = tmp_path / "window.mp4"; clip.write_bytes(b"\0")
    monkeypatch.setattr(L, "_probe_frames", lambda p: 12)

    coords = [(i, i, i, i) for i in range(11)]      # one short of the clip
    got = L._slice_coords_to_window(coords, clip, 0.0, 0.44, 25.0,
                                    model="musetalk")
    assert len(got) == 12
    assert got[-1] == got[-2] == (10, 10, 10, 10), (
        "the pad should hold the last box, not freeze the mouth with the "
        "untouched sentinel")


def test_a_shortfall_too_big_to_be_rounding_is_not_padded(tmp_path,
                                                          monkeypatch):
    """Padding a real disagreement would be guessing which frames the boxes
    belong to — the misalignment this module refuses everywhere else."""
    from backend import lipsync as L

    clip = tmp_path / "window.mp4"; clip.write_bytes(b"\0")
    monkeypatch.setattr(L, "_probe_frames", lambda p: 100)

    coords = [(i, i, i, i) for i in range(10)]
    got = L._slice_coords_to_window(coords, clip, 0.0, 4.0, 25.0,
                                    model="musetalk")
    assert len(got) == 10, "a 90-frame gap was padded as if it were rounding"


def test_an_unreadable_clip_falls_back_to_the_window_arithmetic(
        tmp_path, monkeypatch):
    """`_probe_frames` answers 0 when it cannot read the file, and 0 is a
    silent answer: it would slice the list to nothing and lose the gating
    without saying so."""
    from backend import lipsync as L

    clip = tmp_path / "window.mp4"; clip.write_bytes(b"\0")
    monkeypatch.setattr(L, "_probe_frames", lambda p: 0)

    coords = [(i, i, i, i) for i in range(100)]
    got = L._slice_coords_to_window(coords, clip, 0.0, 2.0, 25.0,
                                    model="musetalk")
    assert len(got) == 50


# ---------------------------------------------------------------------------
# splitting the list the same way the video is split
# ---------------------------------------------------------------------------

def _fake_pairs(tmp_path, counts):
    """(video, audio) pairs whose videos report the given frame counts."""
    pairs = []
    for i, _n in enumerate(counts):
        v = tmp_path / f"part{i}.mp4"; v.write_bytes(b"\0")
        a = tmp_path / f"part{i}.wav"; a.write_bytes(b"\0")
        pairs.append((v, a))
    return pairs


def test_coords_are_split_by_real_frame_counts(tmp_path, monkeypatch):
    """Each piece gets the slice for the frames IT contains.

    By counted frames, not by recomputed time boundaries:
    `split_video_audio` divides by duration and hands the last chunk
    whatever is left, so time arithmetic would have to reproduce both that
    asymmetry and ffmpeg's rounding.
    """
    from backend import lipsync as L

    counts = [10, 10, 5]
    pairs = _fake_pairs(tmp_path, counts)
    monkeypatch.setattr(L, "_probe_frames",
                        lambda p: counts[int(p.stem[-1])])

    coords = [(i, i, i, i) for i in range(25)]
    got = L._split_coords(coords, pairs, model="musetalk", what="segment")

    assert [len(g) for g in got] == counts
    assert got[0][0] == (0, 0, 0, 0)
    assert got[1][0] == (10, 10, 10, 10)      # continues where part 0 ended
    assert got[2][-1] == (24, 24, 24, 24)     # and the tail is consumed
    assert [c for g in got for c in g] == coords


def test_a_piece_with_an_unreadable_frame_count_loses_targeting_only(
        tmp_path, monkeypatch):
    """Dropping the targeting for one piece is recoverable. Misaligning it
    is not — the boxes would land on the wrong frames at exit code 0."""
    from backend import lipsync as L

    pairs = _fake_pairs(tmp_path, [10, 10])
    monkeypatch.setattr(L, "_probe_frames", lambda p: 0)
    got = L._split_coords([(1, 1, 1, 1)] * 20, pairs,
                          model="musetalk", what="shard")
    assert got == [None, None]


def test_a_one_frame_shortfall_does_not_cost_a_whole_piece(tmp_path,
                                                           monkeypatch):
    """`split_video_audio` re-encodes, so the last chunk can come back a
    frame longer than the list has left. Dropping the targeting for all of
    its frames is the expensive answer to one missing frame — and it is what
    cost job 20261023-011027 the second half of its dub."""
    from backend import lipsync as L

    counts = [10, 10]
    pairs = _fake_pairs(tmp_path, counts)
    monkeypatch.setattr(L, "_probe_frames",
                        lambda p: counts[int(p.stem[-1])])
    got = L._split_coords([(i, i, i, i) for i in range(19)], pairs,
                          model="musetalk", what="segment")
    assert [len(g) for g in got] == [10, 10]
    assert got[1][-1] == got[1][-2] == (18, 18, 18, 18)


def test_a_short_list_does_not_misalign_the_later_pieces(tmp_path, monkeypatch):
    """If the list runs out, every piece from there on renders ungated
    rather than being handed somebody else's boxes."""
    from backend import lipsync as L

    counts = [10, 10]
    pairs = _fake_pairs(tmp_path, counts)
    monkeypatch.setattr(L, "_probe_frames",
                        lambda p: counts[int(p.stem[-1])])
    got = L._split_coords([(1, 1, 1, 1)] * 15, pairs,
                          model="musetalk", what="segment")
    assert len(got[0]) == 10
    assert got[1] is None


def test_no_coord_list_means_no_slices(tmp_path):
    from backend import lipsync as L
    pairs = _fake_pairs(tmp_path, [10, 10])
    assert L._split_coords(None, pairs, model="musetalk",
                           what="segment") == [None, None]


def test_targets_no_longer_cost_the_second_gpu_or_the_memory_bound():
    """The two guards these replaced.

    `gpus = gpus[:1]` lost the second card, and `want_segments = False` lost
    the memory bound — which is the worse one, since
    `_run_lipsync_segmented` is what keeps peak RSS a function of the
    segment rather than of the whole clip.
    """
    import inspect

    from backend import lipsync as L

    src = inspect.getsource(L.run_lipsync)
    assert "gpus = gpus[:1]" not in src
    assert "want_segments = False" not in src
    assert "coord_list=coord_list" in src
