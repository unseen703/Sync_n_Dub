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
