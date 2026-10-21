"""`assignments.json` was written by stage 6 and read by nothing.

That file is the only artifact that says WHICH FACE IS THIS SPEAKER, and
without it `predict_gender_face` had no caller — the one signal that gets the
W4L10 lecturer right (5/5, where wav2vec2 scores 0/5 and F0 2/5) was sitting
on disk unused. `backend.assign_faces.speaker_face_tracks` is the reader that
closes the loop, and these tests pin the two decisions inside it that are
easy to get wrong and impossible to notice:

  * the winning track is the one with the most ON-SCREEN SECONDS, not the
    most turns, so a pile of reaction shots cannot outvote a close-up; and
  * a speaker with almost no face time is LEFT OUT rather than handed its
    best guess, because a confident vote on the wrong person is worse than
    two-method agreement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.assign_faces import speaker_face_tracks          # noqa: E402
from models.faces.base import FaceTrack, FaceTracks           # noqa: E402


def _tracks(*ids):
    return FaceTracks(
        video_path=Path("v.mp4"), fps=25.0, width=1920, height=1080,
        num_frames=1000, detector="retinaface", tracker="bytetrack",
        tracks=[FaceTrack(track_id=t,
                          frames=np.arange(10) + 10 * i,
                          bboxes=np.tile([0., 0., 200., 200.], (10, 1)),
                          scores=np.ones(10))
                for i, t in enumerate(ids)],
    )


def _seg(spk, trk, t0, t1, on=True):
    return {"seg_id": f"{spk}-{t0}", "speaker": spk, "face_track": trk,
            "start_s": t0, "end_s": t1, "on_screen": on, "asd_score": 0.9}


def test_the_longest_on_screen_track_wins_not_the_most_turns():
    """Six one-second reaction shots must not outvote one forty-second
    close-up: the close-up is where the face is actually readable."""
    assignments = [_seg("SPK_00", "FACE_01", i, i + 1) for i in range(6)]
    assignments.append(_seg("SPK_00", "FACE_02", 100, 140))

    out = speaker_face_tracks(assignments, _tracks("FACE_01", "FACE_02"))

    frames, _ = out["SPK_00"]
    assert frames.tolist() == list(range(10, 20)), "picked FACE_01 on turn count"


def test_off_screen_turns_are_not_evidence():
    """`assign_faces_to_turns` already nulls the track for an off-screen
    turn; a turn ASD scored below the threshold is narration or an off-camera
    line, where the face in frame belongs to somebody else."""
    assignments = [_seg("SPK_00", "FACE_01", 0, 60, on=False),
                   _seg("SPK_00", "FACE_02", 60, 65, on=True)]

    out = speaker_face_tracks(assignments, _tracks("FACE_01", "FACE_02"))

    frames, _ = out["SPK_00"]
    assert frames.tolist() == list(range(10, 20)), "voted on an off-screen turn"


def test_a_speaker_barely_on_screen_is_omitted():
    """Half a second of face is what `predict_gender_face` abstains on
    anyway. Omitting the speaker says "no face evidence"; an empty track
    would say "no faces found", which is a different claim."""
    out = speaker_face_tracks([_seg("SPK_00", "FACE_01", 0, 0.4)],
                              _tracks("FACE_01"))
    assert out == {}


def test_a_track_named_in_the_json_but_missing_from_the_npz_is_skipped():
    """The two files are written by different stages and a job folder can
    hold a mismatched pair. A KeyError here would take down stage 9d over a
    signal that is optional by design."""
    out = speaker_face_tracks([_seg("SPK_00", "FACE_99", 0, 60)],
                              _tracks("FACE_01"))
    assert out == {}


def test_dataclass_assignments_work_too():
    """Stage 6 has `SegmentAssignment` objects in hand; the edit agent's
    replay only has the JSON. Both callers exist, so both paths must."""
    from backend.assign_faces import SegmentAssignment

    a = SegmentAssignment(seg_id="s0", speaker="SPK_00", start_s=0.0,
                          end_s=60.0, face_track="FACE_01", asd_score=0.9,
                          on_screen=True)
    out = speaker_face_tracks([a], _tracks("FACE_01"))
    assert "SPK_00" in out


def test_two_speakers_can_hold_two_different_faces():
    assignments = [_seg("SPK_00", "FACE_01", 0, 30),
                   _seg("SPK_01", "FACE_02", 30, 60)]

    out = speaker_face_tracks(assignments, _tracks("FACE_01", "FACE_02"))

    assert set(out) == {"SPK_00", "SPK_01"}
    assert out["SPK_00"][0].tolist() != out["SPK_01"][0].tolist()
