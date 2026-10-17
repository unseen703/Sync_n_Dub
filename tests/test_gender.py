"""Gender prediction is a hard filter, so a wrong label is unrecoverable.

`backend/voices.py` matches a speaker only against catalog voices of the
PREDICTED gender (`primary, fallback = male_pool, ...`). The cosine ranking
behind it never sees the other half of the catalog, so it cannot correct a bad
label — it can only pick the least-bad voice of the wrong sex, which is how a
Dark Knight mob scene (two male speakers at 149.9 Hz and 106.9 Hz) came out in
Assamese and Manipuri female voices at cosine 0.02 and 0.16.

The classifier that produced that label was measured on 2026-10-17 against the
45 catalog voices whose gender is known: 44.4% overall, 0% female recall, and
a best-possible 55.6% at ANY threshold against a 51% majority-class baseline.
F0 scored 95.6% on the same set. These tests pin the resulting decision — that
F0 decides — because the failure mode is silent on both sides: a wrong label
errors nowhere, and so does a stage that quietly stops consulting pitch.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import write_tone                                # noqa: E402

import backend.gender as G                                     # noqa: E402


@pytest.fixture
def male_wav(tmp_path):
    """110 Hz — a deep male voice. The Joker clip's SPK_01 measured 106.9 Hz."""
    return write_tone(tmp_path / "male.wav", 2.0, sample_rate=16000, freq=110.0)


@pytest.fixture
def female_wav(tmp_path):
    """220 Hz — clearly above the 165 Hz line."""
    return write_tone(tmp_path / "female.wav", 2.0, sample_rate=16000, freq=220.0)


def test_f0_reads_pitch_in_both_directions(male_wav, female_wav):
    assert G.predict_gender_f0(male_wav) == "male"
    assert G.predict_gender_f0(female_wav) == "female"


def test_f0_decides_even_when_ecapa_disagrees(male_wav, monkeypatch):
    """The regression itself.

    ECAPA called both Dark Knight speakers female. If it ever regains the
    casting vote without the docstring's table being re-measured, this fails.
    """
    monkeypatch.setattr(G, "predict_gender_ecapa", lambda p, **kw: "female")
    monkeypatch.delenv(G.GENDER_METHOD_ENV, raising=False)

    assert G.predict_gender(male_wav) == "male"


def test_a_disagreement_is_logged_loudly(male_wav, monkeypatch, caplog):
    """Silence is what let an all-male lecture ship in a female voice."""
    monkeypatch.setattr(G, "predict_gender_ecapa", lambda p, **kw: "female")
    monkeypatch.delenv(G.GENDER_METHOD_ENV, raising=False)

    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        G.predict_gender(male_wav)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "the two methods disagreed and nothing said so"
    assert "DISAGREE" in warnings[0].getMessage()


def test_agreement_does_not_cry_wolf(male_wav, monkeypatch, caplog):
    monkeypatch.setattr(G, "predict_gender_ecapa", lambda p, **kw: "male")
    monkeypatch.delenv(G.GENDER_METHOD_ENV, raising=False)

    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        assert G.predict_gender(male_wav) == "male"

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_ecapa_can_be_restored_deliberately(male_wav, monkeypatch):
    """An escape hatch, so fixing the sklearn pin does not need a code edit."""
    monkeypatch.setattr(G, "predict_gender_ecapa", lambda p, **kw: "female")
    monkeypatch.setenv(G.GENDER_METHOD_ENV, "ecapa")

    assert G.predict_gender(male_wav) == "female"


def test_a_missing_ecapa_is_not_an_error(male_wav, monkeypatch):
    """speechbrain absent must degrade to pitch, not raise."""
    monkeypatch.setattr(G, "predict_gender_ecapa", lambda p, **kw: None)
    monkeypatch.delenv(G.GENDER_METHOD_ENV, raising=False)

    assert G.predict_gender(male_wav) == "male"


def test_voices_still_filters_hard_on_the_label():
    """If gender became a weight, a bad label would stop being fatal --
    but so would the guarantee this module is relied on for. Pin the shape.
    """
    src = (REPO_ROOT / "backend" / "voices.py").read_text()
    assert 'primary, fallback = male_pool' in src
    assert 'primary, fallback = female_pool' in src
    assert 'spk.get("predicted_gender")' in src


# ---------------------------------------------------------------------------
# Method 3: the face
# ---------------------------------------------------------------------------
#
# Identity is the whole difficulty. The clip this was built against has 17
# face tracks and 7 faces on screen at once, so "the biggest detection" is a
# different person from the one asked about most of the time -- the same
# confusion that makes lip-sync animate the wrong mouth.

class _FakeFace:
    def __init__(self, bbox, sex="M"):
        self.bbox = bbox
        self.sex = sex


def test_best_iou_picks_the_overlapping_face_not_the_biggest():
    from backend.gender import _best_iou

    asked = (100, 100, 200, 200)
    faces = [
        _FakeFace((0, 0, 400, 400), "F"),       # much bigger, barely overlaps
        _FakeFace((105, 105, 205, 205), "M"),   # the one actually asked about
    ]
    assert _best_iou(asked, faces).sex == "M"


def test_best_iou_abstains_when_nothing_really_matches():
    from backend.gender import _best_iou

    assert _best_iou((100, 100, 200, 200), [_FakeFace((900, 900, 1000, 1000))]) is None


def test_face_predictor_abstains_on_an_empty_track(tmp_path):
    import numpy as np
    from backend.gender import predict_gender_face

    g, conf, n = predict_gender_face(
        tmp_path / "nope.mp4", np.array([]), np.zeros((0, 4)))
    assert (g, conf, n) == (None, 0.0, 0)


def test_face_predictor_abstains_on_faces_too_small_to_read(tmp_path):
    """A 40 px face still yields a confident-looking label. Refuse it."""
    import numpy as np
    from backend.gender import predict_gender_face, FACE_GENDER_MIN_BOX_PX

    w = FACE_GENDER_MIN_BOX_PX - 10
    boxes = np.array([[0, 0, w, w]] * 5, dtype=float)
    g, conf, n = predict_gender_face(
        tmp_path / "nope.mp4", np.arange(5), boxes)
    assert g is None and n == 0


def test_face_tracks_npz_round_trips(tmp_path):
    """`save_npz` shipped without a reader; the pair must stay in step."""
    import numpy as np
    from models.faces.base import FaceTrack, FaceTracks

    ft = FaceTracks(
        video_path=tmp_path / "v.mp4", fps=25.0, width=1920, height=1080,
        num_frames=100, detector="retinaface", tracker="bytetrack",
        tracks=[FaceTrack(track_id="FACE_01",
                          frames=np.array([0, 1, 2]),
                          bboxes=np.array([[1., 2., 3., 4.]] * 3),
                          scores=np.array([0.9, 0.8, 0.7]))],
    )
    back = FaceTracks.load_npz(ft.save_npz(tmp_path / "t.npz"))

    assert back.fps == 25.0 and back.width == 1920 and back.num_frames == 100
    assert back.detector == "retinaface" and back.tracker == "bytetrack"
    assert [t.track_id for t in back.tracks] == ["FACE_01"]
    np.testing.assert_allclose(back.tracks[0].bboxes, ft.tracks[0].bboxes)
    np.testing.assert_allclose(back.tracks[0].scores, ft.tracks[0].scores)
