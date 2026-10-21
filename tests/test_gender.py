"""Gender prediction is a hard filter, so a wrong label is unrecoverable.

`backend/voices.py` matches a speaker only against catalog voices of the
PREDICTED gender (`primary, fallback = male_pool, ...`). The cosine ranking
behind it never sees the other half of the catalog, so it cannot correct a bad
label — it can only pick the least-bad voice of the wrong sex.

As of 2026-10-18 THREE methods vote and the majority wins: wav2vec2 over the
reference clip, F0 median pitch over the same clip, and insightface genderage
over the face track ASD attributed to the speaker. The measurements that
forced a vote are in `backend/gender.py`; the short version is that one
lecturer in the six-video test set defeats both audio methods simultaneously
(wav2vec2 0/5, F0 2/5) while the face is right 5/5 on him.

Every failure here is silent — a wrong label errors nowhere, an inverted
label map errors nowhere, a method that quietly stopped voting errors
nowhere, and neither does a tie resolved the wrong way. So these tests pin
the arithmetic of the vote, the label map's SOURCE, the tiebreak order, the
loudness of the no-evidence fallback, and the fact that the catalog filter is
still hard.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import write_tone                                # noqa: E402

import backend.gender as G                                     # noqa: E402


@pytest.fixture
def clip(tmp_path):
    return write_tone(tmp_path / "spk.wav", 2.0, sample_rate=16000, freq=110.0)


# ---------------------------------------------------------------------------
# The classifier decides, and nothing else does
# ---------------------------------------------------------------------------

def _ballot(monkeypatch, w2v2, f0, face, *, f0_hz=120.0, conf=1.0, n=8):
    """Install one fixed ballot. None is an abstention, not a vote."""
    monkeypatch.setattr(G, "predict_gender_w2v2", lambda p, **kw: w2v2)
    monkeypatch.setattr(G, "predict_gender_f0", lambda p, **kw: f0)
    monkeypatch.setattr(G, "median_f0_hz", lambda p, **kw: f0_hz)
    monkeypatch.setattr(G, "predict_gender_face",
                        lambda v, fr, bb, **kw: (face, conf, n))


def _face_kwargs(tmp_path):
    import numpy as np
    return {"video": tmp_path / "v.mp4",
            "face_frames": np.arange(8),
            "face_bboxes": np.tile([0, 0, 200, 200], (8, 1)).astype(float)}


def test_a_unanimous_vote_is_the_easy_case(clip, tmp_path, monkeypatch):
    _ballot(monkeypatch, "male", "male", "male")
    v = G.vote_gender(clip, **_face_kwargs(tmp_path))

    assert v.gender == "male"
    assert v.decided_by == "majority" and v.unanimous
    assert v.tally == {"male": 3, "female": 0}


def test_the_majority_overrules_a_confident_minority(clip, tmp_path, monkeypatch):
    """THE case this vote exists for. On the W4L10 GAN lecturer wav2vec2
    returns female at p=0.997 while F0 and the face both say male; a
    single-model build cast him female, and nothing in its output said so.
    """
    _ballot(monkeypatch, "female", "male", "male")
    v = G.vote_gender(clip, **_face_kwargs(tmp_path))

    assert v.gender == "male"
    assert v.decided_by == "majority" and not v.unanimous
    assert v.votes["w2v2"] == "female"      # the losing vote is still recorded


def test_an_abstention_does_not_vote(clip, monkeypatch):
    """No video means no face vote -- not a face vote of None that counts
    toward the tally and drags every speaker to the fallback."""
    _ballot(monkeypatch, "female", "female", "male")
    v = G.vote_gender(clip)                  # no video/frames/bboxes

    assert v.votes["face"] is None
    assert v.tally == {"male": 0, "female": 2}
    assert v.gender == "female" and not v.unanimous


def test_a_tie_goes_to_the_face(clip, tmp_path, monkeypatch):
    """1-1 happens when one voter abstains. Face leads `TIEBREAK_ORDER`
    because it was never wrong when it answered in the six-video set."""
    _ballot(monkeypatch, "female", None, "male")
    v = G.vote_gender(clip, **_face_kwargs(tmp_path))

    assert v.gender == "male" and v.decided_by == "tiebreak"


def test_a_tie_without_a_face_follows_the_declared_order(clip, monkeypatch):
    _ballot(monkeypatch, "female", "male", None)
    v = G.vote_gender(clip)

    assert G.TIEBREAK_ORDER == ("face", "w2v2", "f0")
    assert v.gender == "female" and v.decided_by == "tiebreak"


def test_a_lone_voter_still_decides(clip, monkeypatch):
    _ballot(monkeypatch, None, "female", None)
    v = G.vote_gender(clip)

    assert v.gender == "female" and v.decided_by == "majority"


def test_a_split_vote_is_logged_loudly(clip, monkeypatch, caplog):
    """A 2-1 produces a usable label from methods that disagree. Silence is
    what let an all-male lecture ship in a female voice."""
    _ballot(monkeypatch, "female", "male", None)
    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        G.vote_gender(clip)

    assert [r for r in caplog.records if "DISAGREE" in r.getMessage()]


def test_an_abstention_is_not_a_disagreement(clip, monkeypatch, caplog):
    """A 2-0 with the face abstaining is agreement. Warning on it would fire
    on every audio-only run and train the reader to ignore the line that
    matters."""
    _ballot(monkeypatch, "male", "male", None)
    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        v = G.vote_gender(clip)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert v.unanimous is False, "2 of 3 voted; that is not a sweep"


def test_a_unanimous_vote_does_not_cry_wolf(clip, tmp_path, monkeypatch, caplog):
    _ballot(monkeypatch, "male", "male", "male")
    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        G.vote_gender(clip, **_face_kwargs(tmp_path))

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_ecapa_stays_removed():
    """ECAPA measured 44.4% here with 0.0% female recall and a best-possible
    55.6% at ANY threshold against a 51% majority-class baseline -- no class
    separation, most likely a sklearn 1.9.0 -> 1.7.2 unpickle mismatch.

    F0 came BACK on 2026-10-18 as a voter, which is a different thing from
    the sole method it once was: a cheap scalar whose errors do not correlate
    with a neural classifier's is exactly what a third voter should be.
    """
    for gone in ("predict_gender_ecapa", "ECAPA_THRESHOLD_KEY",
                 "GENDER_METHOD_ENV"):
        assert not hasattr(G, gone), f"{gone} is back in backend/gender.py"

    src = (REPO_ROOT / "backend" / "gender.py").read_text()
    assert "from speechbrain" not in src, "the ECAPA head reintroduced"
    assert "joblib" not in src, "a pickled sklearn head is back"


def test_all_three_methods_are_actually_wired_in(clip, tmp_path, monkeypatch):
    """A voter that silently stops being called leaves a 2-method vote that
    looks identical to a 3-method one in every artifact."""
    called = []
    monkeypatch.setattr(G, "predict_gender_w2v2",
                        lambda p, **kw: called.append("w2v2") or "male")
    monkeypatch.setattr(G, "predict_gender_f0",
                        lambda p, **kw: called.append("f0") or "male")
    monkeypatch.setattr(G, "median_f0_hz", lambda p, **kw: 120.0)
    monkeypatch.setattr(G, "predict_gender_face",
                        lambda v, fr, bb, **kw: (called.append("face"), ("male", 1.0, 8))[1])

    G.vote_gender(clip, **_face_kwargs(tmp_path))
    assert sorted(called) == ["f0", "face", "w2v2"]


def test_the_face_vote_runs_on_cpu(clip, tmp_path, monkeypatch):
    """By request, and because the CUDA EP holds ~460 MiB for the life of the
    process (CLAUDE.md §1) upstream of a lip-sync stage with ~100 MiB of
    headroom. A GPU device passed for the audio models must not leak here.
    """
    seen = {}
    monkeypatch.setattr(G, "predict_gender_w2v2", lambda p, **kw: "male")
    monkeypatch.setattr(G, "predict_gender_f0", lambda p, **kw: "male")
    monkeypatch.setattr(G, "median_f0_hz", lambda p, **kw: 120.0)
    monkeypatch.setattr(G, "predict_gender_face",
                        lambda v, fr, bb, **kw: (seen.update(kw), ("male", 1.0, 8))[1])

    G.vote_gender(clip, device="cuda:0", **_face_kwargs(tmp_path))

    assert G.FACE_GENDER_DEVICE == "cpu"
    assert seen["device"] == "cpu"


def test_f0_reads_real_pitch_in_both_directions(tmp_path):
    """Not mocked: the threshold itself is load-bearing and every number
    quoted in the module was measured at 165 Hz."""
    low = write_tone(tmp_path / "low.wav", 2.0, sample_rate=16000, freq=110.0)
    high = write_tone(tmp_path / "high.wav", 2.0, sample_rate=16000, freq=220.0)

    assert G.F0_THRESHOLD_HZ == 165.0
    assert G.predict_gender_f0(low) == "male"
    assert G.predict_gender_f0(high) == "female"


def test_f0_abstains_rather_than_guessing_male_on_silence(tmp_path):
    """It used to answer "male" on a clip with no voiced frames, which put a
    real vote behind no evidence -- and in a 3-way vote that is worse than
    the wrong answer it was before, because it can now carry a majority."""
    import numpy as np, soundfile as sf
    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros(16000, dtype="float32"), 16000)

    assert G.predict_gender_f0(silence) is None


# ---------------------------------------------------------------------------
# The label map
# ---------------------------------------------------------------------------
#
# This checkpoint is {0: female, 1: male} -- the opposite of the alphabetical
# guess, and the opposite of the {0: male} convention most binary audio
# classifiers ship. Hard-coding it produces a fully inverted catalog filter
# that is confident, consistent, and reports no error anywhere.

class _FakeModel:
    def __init__(self, id2label, logits_for=None):
        self.config = types.SimpleNamespace(id2label=id2label)
        self._logits_for = logits_for

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **feats):
        import torch
        n = feats["input_values"].shape[-1]
        return types.SimpleNamespace(
            logits=torch.tensor([self._logits_for(n)], dtype=torch.float32))


def _fake_extractor(chunk, sampling_rate=None, return_tensors=None, padding=None):
    import torch
    return {"input_values": torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)}


def _install(monkeypatch, id2label, logits_for):
    monkeypatch.setattr(G, "_w2v2_ready", True)
    monkeypatch.setattr(G, "_w2v2_device", "cpu")
    monkeypatch.setattr(G, "_w2v2_id2label",
                        {int(k): str(v).lower() for k, v in id2label.items()})
    monkeypatch.setattr(G, "_w2v2_extractor", _fake_extractor)
    monkeypatch.setattr(G, "_w2v2_model", _FakeModel(id2label, logits_for))


def test_the_label_map_comes_from_the_model_not_from_position(clip, monkeypatch):
    """Index 0 is female on this checkpoint. Logits that favour index 0 must
    read 'female' — if the module ever assumes index 0 is male, this flips."""
    _install(monkeypatch, {0: "female", 1: "male"}, lambda n: [8.0, 0.0])
    assert G.predict_gender_w2v2(clip) == "female"

    _install(monkeypatch, {0: "female", 1: "male"}, lambda n: [0.0, 8.0])
    assert G.predict_gender_w2v2(clip) == "male"


def test_a_model_with_other_labels_is_refused_not_guessed(monkeypatch):
    """A checkpoint labelled {0: LABEL_0, 1: LABEL_1} carries no information
    about which class is which, and guessing is a coin flip applied to every
    speaker at once. `DUBBING_GENDER_MODEL` makes this reachable by env var.
    """
    monkeypatch.setattr(G, "_w2v2_ready", None)
    monkeypatch.setattr(G, "_w2v2_device", "cpu")

    import transformers
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _fake_extractor))
    monkeypatch.setattr(
        transformers.AutoModelForAudioClassification, "from_pretrained",
        classmethod(lambda cls, *a, **k: _FakeModel({0: "LABEL_0", 1: "LABEL_1"},
                                                    lambda n: [1.0, 0.0])))

    assert G._ensure_w2v2("cpu") is False


def test_scores_are_probabilities_over_both_classes(clip, monkeypatch):
    _install(monkeypatch, {0: "female", 1: "male"}, lambda n: [0.0, 0.0])
    scores = G.predict_gender_w2v2_scores(clip)

    assert set(scores) == {"male", "female"}
    assert scores["male"] == pytest.approx(0.5)
    assert sum(scores.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_a_long_clip_is_chunked_and_weighted_by_length(tmp_path, monkeypatch):
    """Self-attention is quadratic, so a long reference clip is windowed. The
    windows must then be averaged by LENGTH: a 2 s tail of room tone voting as
    loudly as the 20 s of speech before it is how a whole speaker gets recast
    on the quietest part of their own clip.
    """
    long_clip = write_tone(tmp_path / "long.wav", 22.0,
                           sample_rate=16000, freq=110.0)
    full = int(G.W2V2_CHUNK_S * G.TARGET_SR)

    seen = []

    def logits_for(n):
        seen.append(n)
        # Full windows say female quietly; the 2 s tail screams male.
        return [2.0, 0.0] if n == full else [0.0, 9.0]

    _install(monkeypatch, {0: "female", 1: "male"}, logits_for)
    scores = G.predict_gender_w2v2_scores(long_clip)

    assert seen == [full, full, 2 * G.TARGET_SR], "not chunked at W2V2_CHUNK_S"
    assert max(scores, key=scores.__getitem__) == "female", (
        "the 2 s tail outvoted 20 s of speech — windows are not length-weighted")


def test_a_clip_too_short_to_classify_is_refused(tmp_path, monkeypatch):
    """Below the encoder's receptive field this raises a shape error rather
    than answering, so it must be caught before inference, not after."""
    tiny = write_tone(tmp_path / "tiny.wav", 0.02, sample_rate=16000, freq=110.0)
    _install(monkeypatch, {0: "female", 1: "male"}, lambda n: [9.0, 0.0])

    assert G.predict_gender_w2v2_scores(tiny) is None


# ---------------------------------------------------------------------------
# The fallback
# ---------------------------------------------------------------------------

def test_an_empty_ballot_still_returns_a_label(clip, monkeypatch):
    """`predict_gender` never returns None: `backend/voices.py` filters the
    catalog on whatever it gets and has no branch for "don't know"."""
    _ballot(monkeypatch, None, None, None)

    v = G.vote_gender(clip)
    assert v.gender == G.FALLBACK_GENDER == "male"
    assert v.decided_by == "fallback"
    assert v.tally == {"male": 0, "female": 0}


def test_the_fallback_says_so_at_warning(clip, monkeypatch, caplog):
    """Silence is what let an all-male lecture ship in a female voice. A
    constant standing in for a prediction has to be visible as one."""
    _ballot(monkeypatch, None, None, None)

    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        G.predict_gender(clip)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "every method abstained and nothing said so"
    assert "fallback" in warnings[0].getMessage().lower()


def test_the_fallback_asks_for_a_target_language_voice(monkeypatch, tmp_path):
    """"No gender detected -> target-language male voice" is the requested
    behaviour, and it needs `assign_voices` to STOP ranking by timbre: a
    placeholder gender ranked by cosine is a guess wearing a score.
    """
    src = (REPO_ROOT / "backend" / "voices.py").read_text()
    assert 'spk.get("gender_basis") == "fallback"' in src
    assert "no gender evidence" in src

    pipe = (REPO_ROOT / "backend" / "pipeline.py").read_text()
    assert 'spk["gender_basis"] = "fallback"' in pipe, (
        "nothing marks a no-evidence speaker, so assign_voices can never see it")


def test_a_real_prediction_does_not_cry_wolf(clip, monkeypatch, caplog):
    _ballot(monkeypatch, "male", "male", None)

    with caplog.at_level(logging.WARNING, logger="backend.gender"):
        assert G.predict_gender(clip) == "male"

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failed_load_is_not_retried_per_speaker(monkeypatch):
    """Stage 9d calls this once per speaker. A missing model that re-attempts
    the download for each one turns one warning into a stalled stage."""
    calls = []
    monkeypatch.setattr(G, "_w2v2_ready", False)
    monkeypatch.setattr(G, "_w2v2_device", "cpu")

    import transformers
    monkeypatch.setattr(transformers.AutoFeatureExtractor, "from_pretrained",
                        classmethod(lambda cls, *a, **k: calls.append(1)))

    assert G._ensure_w2v2("cpu") is False
    assert G._ensure_w2v2("cpu") is False
    assert calls == []


def test_predict_genders_skips_speakers_with_no_clip(monkeypatch, tmp_path):
    _ballot(monkeypatch, "female", "female", None)
    clip = write_tone(tmp_path / "a.wav", 1.0, sample_rate=16000, freq=200.0)

    out = G.predict_genders({"SPK_00": clip,
                             "SPK_01": tmp_path / "missing.wav",
                             "SPK_02": None})
    assert out == {"SPK_00": "female"}


def test_a_speaker_with_only_a_face_still_gets_a_vote(monkeypatch, tmp_path):
    """No usable reference clip but a face track ASD attributed to them is
    real evidence -- better than the fallback, which is none."""
    _ballot(monkeypatch, "male", "male", "female")      # audio must not run
    import numpy as np

    out = G.predict_genders(
        {"SPK_00": tmp_path / "missing.wav"},
        video=tmp_path / "v.mp4",
        speaker_face_tracks={"SPK_00": (np.arange(8),
                                        np.tile([0, 0, 200, 200], (8, 1)))},
    )
    assert out == {"SPK_00": "female"}


def test_the_ballot_is_recorded_for_audit(monkeypatch, tmp_path):
    """Stage 9d writes this into the speaker voice map. A 3-0 sweep, a 1-1
    tiebreak and a no-evidence fallback all produce a usable label and only
    one of them is worth trusting, so the artifact has to tell them apart.
    """
    _ballot(monkeypatch, "female", "male", None, f0_hz=164.0)
    clip = write_tone(tmp_path / "a.wav", 1.0, sample_rate=16000, freq=200.0)

    details = {}
    G.predict_genders({"SPK_00": clip}, details_out=details)

    row = details["SPK_00"]
    assert row["gender"] == "female" and row["decided_by"] == "tiebreak"
    assert row["votes"] == {"w2v2": "female", "f0": "male", "face": None}
    assert row["median_f0_hz"] == 164.0 and row["unanimous"] is False
    import json
    json.dumps(row)            # must survive the manifest round trip


# ---------------------------------------------------------------------------
# The consumer
# ---------------------------------------------------------------------------

def test_voices_still_filters_hard_on_the_label():
    """If gender became a weight, a bad label would stop being fatal --
    but so would the guarantee this module is relied on for. Pin the shape.
    """
    src = (REPO_ROOT / "backend" / "voices.py").read_text()
    assert 'primary, fallback = male_pool' in src
    assert 'primary, fallback = female_pool' in src
    assert 'spk.get("predicted_gender")' in src


# ---------------------------------------------------------------------------
# Method 2: the face
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
