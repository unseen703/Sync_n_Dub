"""Speaker gender prediction (plan §8 — voice assignment signal #1).

This answer is a HARD FILTER downstream, not a weight: `backend/voices.py`
matches a speaker only against catalog voices of the predicted gender, so a
wrong label here cannot be recovered by the cosine ranking behind it. It is
the single input most able to ruin a dub while every stage reports success —
CLAUDE.md §4 records one all-male lecture that shipped in a female voice.

**Three predictors vote, and the majority wins.** Rebuilt 2026-10-18 after a
single-model build was measured failing on one speaker in a way nothing in
its output revealed:

  1. **wav2vec2-XLSR-53** (`alefiury/wav2vec2-large-xlsr-53-gender-
     recognition-librispeech`) — a `Wav2Vec2ForSequenceClassification` head on
     a multilingual encoder. Reads the waveform.
  2. **F0 median pitch** — librosa.pyin, below `F0_THRESHOLD_HZ` (165) male.
     One scalar, no model, and it fails on entirely different inputs.
  3. **The speaker's FACE** — insightface's `genderage.onnx`, voted over
     frames of the face track ASD attributed to this speaker. **CPU only**
     (see `FACE_GENDER_DEVICE`). Needs a video and a track; silent on audio
     alone, which is why it cannot be the only method either.

WHY A VOTE, measured on this host 2026-10-18 over 30 VAD-selected 5 s windows
from six videos, with a visual check of who is actually speaking
(`data/gender_bench/eval_5s_*_2026-10-18/`):

    video                                   w2v2     F0     face
    Dark Knight mob scene                    5/5    5/5    2 of 5 answer, both right
    John Wick "I'm Back"                     5/5    4/5    3 of 5 answer, all right
    L10 McCulloch Pitts (Khapra)             5/5    5/5    5/5
    W3L9 DCGAN / Conditional GANs            1/5    2/5    4 of 5 answer, all right
    W4L10 Saturation of GAN training         0/5    2/5    5/5

The last two are the same lecturer. On him wav2vec2 returns female on ~60% of
windows at mean confidence 0.96, and his median F0 of 150–207 Hz straddles the
165 Hz line. Meanwhile John Wick at 103 s is a scream at 436 Hz, where F0 says
female and wav2vec2 is right. No single one of these is safe.

**WHAT THE VOTE ACTUALLY BUYS, measured end to end on those 25 clips
(2026-10-18): wav2vec2 alone 16/25 = 64%, the three-way majority 18/25 = 72%.
It is an improvement and it is NOT a fix.** The reason is the thing a vote is
supposed to protect against and here does not: on the GAN lecturer the two
AUDIO methods fail TOGETHER — his pitch really is above the 165 Hz line, and
whatever wav2vec2 keys on moves with it — so they outvote the single correct
face 2-1 on 7 of his 10 clips. Their errors are correlated by construction,
because both read the same waveform.

The face, by contrast, was right on 23 of the 23 clips where it answered,
including every one of those seven. So the honest summary is that this vote
fixes the uncorrelated failures (the 436 Hz scream, an unvoiced clip, a
missing model) and leaves the correlated one standing. Giving the face a veto
when it has `FACE_GENDER_MIN_VOTES` frames would score 25/25 on this set —
that is a deliberate departure from majority rule, not an oversight, and it
is not what runs today.

TIES. Three voters over two classes always produce a majority, so a tie only
happens when one abstains and the other two disagree. `TIEBREAK_ORDER` settles
it: **face first**, because in the measurements above it was never wrong when
it answered, and it is the only one that is right about the GAN lecturer.

NO VOTES AT ALL — no video, an unreadable clip, nothing loaded — returns
`FALLBACK_GENDER` ("male") at WARNING. `backend/voices.py` then casts a voice
recorded in the TARGET LANGUAGE of that gender (`rationale.basis` =
`"language+gender"`), so the degenerate case is a target-language male voice
rather than an arbitrary one.

``predict_gender`` always returns ``"male"`` or ``"female"`` — never None,
because the caller filters the voice catalog on whatever it gets and has no
branch for "don't know". ``vote_gender`` returns the whole ballot, which is
what stage 9d records so a finished run can be audited without re-running
anything.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared audio loading
# ---------------------------------------------------------------------------

#: Everything here infers at 16 kHz mono: it is the rate the feature extractor
#: declares (`preprocessor_config.json`) and silently resampling to anything
#: else shifts every learned filter.
TARGET_SR = 16000


def _load_mono16k(wav_path: Path):
    """Read a clip as float32 mono at 16 kHz."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR and len(audio):
        idx = np.linspace(0, len(audio) - 1,
                          int(len(audio) * TARGET_SR / sr)).astype(np.float64)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    return audio

# ---------------------------------------------------------------------------
# Method 1: wav2vec2-XLSR-53 sequence classifier  (the only voice method)
# ---------------------------------------------------------------------------

W2V2_REPO_ENV = "DUBBING_GENDER_MODEL"
W2V2_REPO = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

#: Seconds per inference window. Self-attention costs grow with the SQUARE of
#: the input length, and a stage-9d reference clip is not bounded — it is
#: however long that speaker's turns add up to. Chunking makes the cost linear
#: and makes the answer an average over the whole clip rather than a guess
#: from whatever its first seconds happened to contain.
W2V2_CHUNK_S = 10.0

#: Under this, there is nothing to classify. 0.1 s is also below the encoder's
#: own receptive field, so a shorter clip produces a shape error rather than a
#: bad answer.
MIN_AUDIO_S = 0.1

_w2v2_model = None
_w2v2_extractor = None
_w2v2_id2label: Dict[int, str] = {}
_w2v2_device = "cpu"
_w2v2_ready: Optional[bool] = None  # None = not tried yet


def _ensure_w2v2(device: str = "cpu") -> bool:
    """Lazy-load the classifier. Returns True on success, False once and for
    all on failure — a missing model must not be retried per segment."""
    global _w2v2_model, _w2v2_extractor, _w2v2_id2label, _w2v2_device, _w2v2_ready
    if _w2v2_ready is not None and _w2v2_device == device:
        return _w2v2_ready
    try:
        import os

        from transformers import (AutoFeatureExtractor,
                                  AutoModelForAudioClassification)

        repo = os.environ.get(W2V2_REPO_ENV, W2V2_REPO).strip() or W2V2_REPO
        _w2v2_extractor = AutoFeatureExtractor.from_pretrained(repo)
        _w2v2_model = AutoModelForAudioClassification.from_pretrained(repo)
        _w2v2_model.to(device)
        _w2v2_model.eval()
        _w2v2_device = device

        # The label ORDER is the model's, read off its own config, never
        # assumed. This checkpoint is {0: female, 1: male} — the opposite of
        # the alphabetical guess — and hard-coding it the other way round is
        # the one mistake here that yields a confident, fully inverted filter
        # on the voice catalog with nothing anywhere reporting an error.
        _w2v2_id2label = {int(k): str(v).strip().lower()
                          for k, v in _w2v2_model.config.id2label.items()}
        unexpected = set(_w2v2_id2label.values()) - {"male", "female"}
        if unexpected:
            raise ValueError(
                f"{repo} labels {sorted(unexpected)}; this module only "
                f"understands male/female")

        _w2v2_ready = True
        log.info("gender: wav2vec2 classifier loaded (%s, device=%s, labels=%s)",
                 repo, device, _w2v2_id2label)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("gender: wav2vec2 classifier UNAVAILABLE (%s: %s). Every "
                    "speaker will fall back to %r — check the dub's voices.",
                    type(exc).__name__, exc, FALLBACK_GENDER)
        _w2v2_model = _w2v2_extractor = None
        _w2v2_ready = False
    return _w2v2_ready


def predict_gender_w2v2_scores(
    wav_path: Path,
    *,
    device: str = "cpu",
) -> Optional[Dict[str, float]]:
    """``{"male": p, "female": p}`` for one clip, or None if it cannot answer.

    Probabilities are averaged over `W2V2_CHUNK_S` windows, WEIGHTED by each
    window's length, so a 2 s tail does not count as much as the 10 s before
    it. Returning the scores rather than only the label is what lets a caller
    (and `scripts/benchmark_gender.py`) see a 0.51 that happens to be right
    differently from a 0.99.
    """
    if not _ensure_w2v2(device):
        return None
    import numpy as np
    import torch

    wav_path = Path(wav_path)
    try:
        audio = _load_mono16k(wav_path)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("gender(w2v2): cannot read %s (%s: %s)",
                    wav_path.name, type(exc).__name__, exc)
        return None

    if audio.size < MIN_AUDIO_S * TARGET_SR:
        log.warning("gender(w2v2): %s is %.3f s — too short to classify",
                    wav_path.name, audio.size / TARGET_SR)
        return None

    win = int(W2V2_CHUNK_S * TARGET_SR)
    chunks = [audio[i:i + win] for i in range(0, audio.size, win)]
    # Drop a final sliver rather than let it vote at full weight; if that
    # leaves nothing, the clip was shorter than one window and IS the chunk.
    chunks = [c for c in chunks if c.size >= MIN_AUDIO_S * TARGET_SR] or [audio]

    totals = np.zeros(len(_w2v2_id2label), dtype="float64")
    weight = 0.0
    try:
        with torch.no_grad():
            for chunk in chunks:
                feats = _w2v2_extractor(chunk, sampling_rate=TARGET_SR,
                                        return_tensors="pt", padding=True)
                feats = {k: v.to(_w2v2_device) for k, v in feats.items()}
                logits = _w2v2_model(**feats).logits
                probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
                totals += probs * chunk.size
                weight += chunk.size
    except Exception as exc:                                  # noqa: BLE001
        log.warning("gender(w2v2): inference failed on %s (%s: %s)",
                    wav_path.name, type(exc).__name__, exc)
        return None

    totals /= weight
    return {_w2v2_id2label[i]: float(totals[i]) for i in range(totals.size)}


def predict_gender_w2v2(wav_path: Path, *, device: str = "cpu") -> Optional[str]:
    """Predict gender with the wav2vec2 classifier.

    Returns ``"male"`` or ``"female"``, or ``None`` when the model is
    unavailable or the clip is unusable — the caller decides what to do with
    that, so the failure cannot be mistaken for a prediction.
    """
    scores = predict_gender_w2v2_scores(wav_path, device=device)
    if not scores:
        return None
    label = max(scores, key=scores.__getitem__)
    log.debug("gender(w2v2): %s — %s → %s", Path(wav_path).name,
              {k: round(v, 4) for k, v in scores.items()}, label)
    return label


# ---------------------------------------------------------------------------
# Method 2: F0 median pitch
# ---------------------------------------------------------------------------
#
# Restored 2026-10-18 as a VOTER after a spell as the sole method and a spell
# deleted. Neither extreme was right. It is a proxy for sex rather than a
# classifier of it -- a contralto, a tenor, a child, a whisper or music bleeding
# into the clip all move the median without moving the speaker -- but it is one
# cheap scalar whose errors do not correlate with a neural classifier's, which
# is the only property a third voter needs.
#
# Measured on this host: 95.6% on the 45 `indic-speak` catalog clips, 72.7% on
# 11 ASD-gated natural-speech tracks, and on the six-video set in the module
# docstring it beat wav2vec2 on the one speaker wav2vec2 cannot read (2/5 vs
# 0/5) while losing to it on a scream (436 Hz, called female).

#: Below this, male. 165 Hz is the conventional split and the one every number
#: quoted in this module was measured at; moving it invalidates all of them.
F0_THRESHOLD_HZ = 165.0


def predict_gender_f0(wav_path: Path, *, sr: int = TARGET_SR) -> Optional[str]:
    """Predict gender from median fundamental frequency via librosa.pyin.

    Returns ``"male"``, ``"female"``, or ``None`` when the clip has no voiced
    frames at all. None is an ABSTENTION, not a guess: this used to answer
    "male" on an unvoiced clip, which put a real vote behind no evidence.
    """
    import numpy as np

    try:
        # Inside the try, not above it. librosa reaches this env only as a
        # transitive dependency of `funasr`, which is an OPTIONAL stage, so
        # a lean install can legitimately lack it -- and an ImportError
        # escaping here would take down stage 9d over a method that is
        # explicitly allowed to abstain.
        import librosa
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=2048,
        )
    except Exception as exc:                                  # noqa: BLE001
        log.warning("gender(f0): failed on %s (%s: %s)",
                    Path(wav_path).name, type(exc).__name__, exc)
        return None

    voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
    if len(voiced_f0) == 0:
        log.info("gender(f0): no voiced frames in %s; abstaining",
                 Path(wav_path).name)
        return None

    median_f0 = float(np.median(voiced_f0))
    gender = "female" if median_f0 >= F0_THRESHOLD_HZ else "male"
    log.debug("gender(f0): %s — median_f0=%.1f Hz → %s",
              Path(wav_path).name, median_f0, gender)
    return gender


def median_f0_hz(wav_path: Path, *, sr: int = TARGET_SR) -> Optional[float]:
    """The median voiced F0, or None. Recorded alongside the vote, because
    a value sitting on the 165 Hz line is the signature of the speaker this
    whole vote exists for."""
    import numpy as np

    try:
        import librosa
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr,
                                     frame_length=2048)
    except Exception:                                         # noqa: BLE001
        return None
    v = f0[voiced & np.isfinite(f0)]
    return float(np.median(v)) if len(v) else None


# ---------------------------------------------------------------------------
# Method 3: the speaker's FACE
# ---------------------------------------------------------------------------
#
# The classifier above answers "what does this voice sound like"; the face
# answers "who is this". They fail on different inputs, which is the whole
# reason to keep both: the audio model is confused by singing, whisper, a
# phone-band recording or background music bleeding into the reference clip,
# and none of those touch a face crop. Conversely the face is unavailable
# off-screen, in profile, or in the dark, where the voice is unbothered.
#
# It IS a voter in `predict_gender` as of 2026-10-18, but only when the caller
# supplies a face track ASD attributed to this speaker. Without one it simply
# abstains and the vote is between the two audio methods.
#
# This reuses `genderage.onnx`, which insightface ALREADY downloads as part of
# the buffalo_l pack that stage 5 uses for detection -- it is sitting unused in
# ~/.insightface/models/buffalo_l/. The detector adapter asks for
# `allowed_modules=("detection",)`, so nothing ever loads it. No new model, no
# new dependency.
#
# WHAT THIS DOES NOT DO: it cannot tell you which face is the speaker. That
# link is `backend/assign_faces.py`'s job (ASD + Hungarian matching, written to
# `<job>/asd/*.assignments.json`); `speaker_face_tracks()` there now reads that
# file back, which is what finally gives this method a caller. Pass a track
# that ASD attributed to the speaker; passing "the biggest face" instead is a
# guess, and on a 7-faces-on-screen shot it is usually the wrong one.

#: CPU, always, and not because the GPU would be slow. `genderage.onnx` is a
#: few MB and runs on a dozen 112x112 crops per speaker — it is the detector
#: in front of it that costs anything, and even that is ~56 ms/frame on CPU
#: against a dozen frames. What the CUDA EP costs is ~460 MiB resident in the
#: process for as long as it lives (CLAUDE.md §1), and stage 9d sits upstream
#: of a lip-sync stage that is already within 100 MiB of the 16 GiB ceiling at
#: MuseTalk's default batch. Trading a second of wall clock for that headroom
#: is the right way round.
FACE_GENDER_DEVICE = "cpu"

_facega_app = None
_facega_device: Optional[str] = None
_facega_ready: Optional[bool] = None

#: Below this, say nothing rather than guess. A face seen in profile or at
#: 40 px still produces a confident-looking label.
FACE_GENDER_MIN_VOTES = 3
FACE_GENDER_MIN_BOX_PX = 60


def _ensure_face_genderage(device: str = FACE_GENDER_DEVICE):
    """Load buffalo_l with the genderage module switched ON.

    Cached, including the failure: stage 9d calls this once per speaker and a
    missing model must not re-attempt the download for each of them.
    """
    global _facega_app, _facega_device, _facega_ready
    if _facega_ready is not None and _facega_device == device:
        return _facega_app if _facega_ready else None
    try:
        from insightface.app import FaceAnalysis

        providers = (["CUDAExecutionProvider"] if device.startswith("cuda")
                     else ["CPUExecutionProvider"])
        if device.startswith("cuda"):
            from backend.gpu import ensure_onnxruntime_cuda
            ensure_onnxruntime_cuda()
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=("detection", "genderage"),
                           providers=providers)
        app.prepare(ctx_id=0 if device.startswith("cuda") else -1,
                    det_size=(640, 640))
        _facega_app, _facega_ready, _facega_device = app, True, device
        log.info("gender(face): insightface genderage loaded (device=%s)", device)
    except Exception as exc:                                 # noqa: BLE001
        log.info("gender(face): insightface genderage unavailable (%s: %s); "
                 "the vote proceeds on the audio methods alone",
                 type(exc).__name__, exc)
        _facega_app, _facega_ready, _facega_device = None, False, device
    return _facega_app


def predict_gender_face(
    video: Path,
    frames,                       # np.ndarray (N,) frame indices
    bboxes,                       # np.ndarray (N, 4) xyxy
    *,
    device: str = FACE_GENDER_DEVICE,
    max_samples: int = 12,
    margin: float = 0.35,
) -> Tuple[Optional[str], float, int]:
    """Vote a gender from the face itself.

    Returns ``(gender, confidence, n_votes)``; ``(None, 0.0, n)`` when the
    track gave too little to be worth trusting. `confidence` is the share of
    votes the winner took, NOT a model probability -- a 3-0 sweep of a blurry
    profile still reads 1.0, so weigh it with `n_votes`.

    Samples evenly across the track rather than taking the first frames: a
    track usually begins mid-turn or mid-pan, and the opening frames are the
    most likely to be motion-blurred.
    """
    import numpy as np

    app = _ensure_face_genderage(device)
    if app is None:
        return None, 0.0, 0

    import cv2

    frames = np.asarray(frames)
    bboxes = np.asarray(bboxes)
    if frames.size == 0:
        return None, 0.0, 0

    big = (bboxes[:, 2] - bboxes[:, 0]) >= FACE_GENDER_MIN_BOX_PX
    idx = np.flatnonzero(big)
    if idx.size == 0:
        return None, 0.0, 0
    pick = idx[np.linspace(0, idx.size - 1, min(max_samples, idx.size)).astype(int)]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning("gender(face): cannot open %s", video)
        return None, 0.0, 0

    votes = {"male": 0, "female": 0}
    try:
        for i in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[i]))
            ok, img = cap.read()
            if not ok or img is None:
                continue
            # Detect on the FULL FRAME, then match the track's box to one of
            # the detections. Cropping to the track box first and detecting
            # inside it returns ZERO faces -- measured on a 378x272 crop that
            # was nothing but face. RetinaFace needs context around the head
            # (it regresses the box from a wider receptive field), so a tight
            # crop is out of distribution no matter how much margin is added.
            try:
                faces = app.get(img)
            except Exception as exc:                         # noqa: BLE001
                log.warning("gender(face): inference failed (%s: %s)",
                            type(exc).__name__, exc)
                break
            if not faces:
                continue
            f = _best_iou(bboxes[i], faces)
            if f is None:
                continue
            sex = getattr(f, "sex", None)
            if sex == "M":
                votes["male"] += 1
            elif sex == "F":
                votes["female"] += 1
    finally:
        cap.release()

    n = votes["male"] + votes["female"]
    if n < FACE_GENDER_MIN_VOTES:
        log.info("gender(face): only %d usable face(s) in the track; abstaining", n)
        return None, 0.0, n
    winner = "male" if votes["male"] >= votes["female"] else "female"
    return winner, votes[winner] / n, n


#: A track box and a fresh detection of the same face rarely agree exactly --
#: different detector settings, a frame of drift. Below this they are not the
#: same face, and voting on the wrong one is worse than abstaining.
FACE_MATCH_MIN_IOU = 0.3


def _best_iou(box, faces):
    """The detected face that best overlaps `box`, or None if none really do.

    Identity on a multi-face shot is the entire difficulty here: the clip this
    was built against has 17 tracks and 7 faces on screen at once, so "the
    biggest detection" picks a different person from the one asked about most
    of the time.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    best, best_iou = None, 0.0
    for f in faces:
        a1, b1, a2, b2 = (float(v) for v in f.bbox)
        ix = max(0.0, min(x2, a2) - max(x1, a1))
        iy = max(0.0, min(y2, b2) - max(y1, b1))
        inter = ix * iy
        union = area + max(0.0, a2 - a1) * max(0.0, b2 - b1) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = f, iou
    return best if best_iou >= FACE_MATCH_MIN_IOU else None


# ---------------------------------------------------------------------------
# The vote
# ---------------------------------------------------------------------------

#: What a speaker gets when NOTHING could answer. It is a placeholder standing
#: in for a prediction, not a prediction, and it is logged at WARNING every
#: time. `backend/voices.py` turns it into a voice recorded in the TARGET
#: LANGUAGE of this gender (`rationale.basis == "language+gender"`), so the
#: degenerate case is a target-language male voice rather than an arbitrary
#: catalog entry.
FALLBACK_GENDER = "male"

#: Three voters over two classes always yield a majority, so this is reached
#: only when one abstains and the other two split 1-1. Face leads because in
#: every measurement on this host it was right whenever it answered — notably
#: on the one lecturer both audio methods get wrong. See the module docstring.
TIEBREAK_ORDER = ("face", "w2v2", "f0")

#: The order votes are reported in. Not the tiebreak order: this is for
#: humans reading a log line.
METHODS = ("w2v2", "f0", "face")


@dataclass
class GenderVote:
    """One speaker's ballot, kept whole so a finished run can be audited.

    `gender` is always "male" or "female". `decided_by` distinguishes the
    three very different ways that happened — a 3-0 sweep, a 1-1 tiebreak
    and a no-evidence fallback all produce a usable label and only one of
    them is worth trusting.
    """
    gender: str
    decided_by: str                       # "majority" | "tiebreak" | "fallback"
    votes: Dict[str, Optional[str]] = field(default_factory=dict)
    tally: Dict[str, int] = field(default_factory=dict)
    median_f0_hz: Optional[float] = None
    face_confidence: float = 0.0
    face_frames_used: int = 0
    unanimous: bool = False

    def to_jsonable(self) -> dict:
        return {
            "gender": self.gender,
            "decided_by": self.decided_by,
            "unanimous": self.unanimous,
            "votes": dict(self.votes),
            "tally": dict(self.tally),
            "median_f0_hz": (round(self.median_f0_hz, 1)
                             if self.median_f0_hz is not None else None),
            "face_confidence": round(self.face_confidence, 3),
            "face_frames_used": self.face_frames_used,
        }


def vote_gender(
    wav_path: Optional[Path],
    *,
    device: str = "cpu",
    video: Optional[Path] = None,
    face_frames=None,
    face_bboxes=None,
) -> GenderVote:
    """Run every applicable method and take the majority.

    `wav_path` drives methods 1 and 2. Methods 3 needs all three of `video`,
    `face_frames` and `face_bboxes` — the face track ASD attributed to THIS
    speaker. Passing "the biggest face" instead is a guess, and on a
    multi-face shot it is usually the wrong person, so the caller is required
    to have done the attribution rather than this function guessing at it.

    Every method may abstain (None). Abstentions do not vote and do not count
    toward the majority.
    """
    votes: Dict[str, Optional[str]] = {m: None for m in METHODS}
    f0_hz: Optional[float] = None
    face_conf, face_n = 0.0, 0

    if wav_path is not None and Path(wav_path).exists():
        votes["w2v2"] = predict_gender_w2v2(wav_path, device=device)
        votes["f0"] = predict_gender_f0(wav_path)
        f0_hz = median_f0_hz(wav_path)

    if video is not None and face_frames is not None and face_bboxes is not None:
        # Always CPU — see FACE_GENDER_DEVICE.
        votes["face"], face_conf, face_n = predict_gender_face(
            Path(video), face_frames, face_bboxes,
            device=FACE_GENDER_DEVICE,
        )

    cast = [g for g in votes.values() if g in ("male", "female")]
    tally = {"male": cast.count("male"), "female": cast.count("female")}

    if not cast:
        log.warning("gender: no method could answer (%s); using the fallback "
                    "%r. This is NOT a prediction — it is a hard filter on "
                    "the voice catalog, so check the cast.",
                    Path(wav_path).name if wav_path else "no clip",
                    FALLBACK_GENDER)
        return GenderVote(gender=FALLBACK_GENDER, decided_by="fallback",
                          votes=votes, tally=tally, median_f0_hz=f0_hz,
                          face_confidence=face_conf, face_frames_used=face_n)

    if tally["male"] != tally["female"]:
        winner = "male" if tally["male"] > tally["female"] else "female"
        decided_by = "majority"
    else:
        # 1-1: one voter abstained and the other two disagree.
        winner = next(votes[m] for m in TIEBREAK_ORDER
                      if votes[m] in ("male", "female"))
        decided_by = "tiebreak"

    # Two distinct things, and conflating them cries wolf on every ordinary
    # audio-only run: `disagreement` is voters CONTRADICTING each other,
    # which is worth a warning. `unanimous` additionally requires that all
    # three actually voted -- a 2-0 with the face abstaining is agreement,
    # not a sweep, and the artifact should not claim three-way support it
    # does not have.
    disagreement = len(set(cast)) > 1
    unanimous = len(cast) == len(METHODS) and not disagreement
    vote = GenderVote(gender=winner, decided_by=decided_by, votes=votes,
                      tally=tally, median_f0_hz=f0_hz,
                      face_confidence=face_conf, face_frames_used=face_n,
                      unanimous=unanimous)

    name = Path(wav_path).name if wav_path else "no clip"
    detail = " ".join(f"{m}={votes[m] or '-'}" for m in METHODS)
    if disagreement:
        # A split is the cheapest available warning that one method has
        # drifted, and the reason this is a vote rather than one model.
        log.warning("gender: %s → %s by %s — voters DISAGREE (%s, f0=%s Hz). "
                    "This label is a hard filter on the voice catalog.",
                    name, winner, decided_by, detail,
                    f"{f0_hz:.0f}" if f0_hz else "n/a")
    else:
        log.info("gender: %s → %s (%s%s)", name, winner, detail,
                 ", unanimous" if unanimous else "")
    return vote


def predict_gender(
    wav_path: Optional[Path],
    *,
    device: str = "cpu",
    video: Optional[Path] = None,
    face_frames=None,
    face_bboxes=None,
) -> str:
    """Majority gender for one speaker. Always "male" or "female"."""
    return vote_gender(wav_path, device=device, video=video,
                       face_frames=face_frames, face_bboxes=face_bboxes).gender


def predict_genders(
    speaker_refs: Dict[str, Path],
    *,
    device: str = "cpu",
    video: Optional[Path] = None,
    speaker_face_tracks: Optional[Dict[str, Tuple]] = None,
    details_out: Optional[Dict[str, dict]] = None,
) -> Dict[str, str]:
    """Predict gender for every speaker.

    `speaker_face_tracks` maps a speaker id to that speaker's `(frames,
    bboxes)` — the track ASD attributed to them, which
    `backend.assign_faces.speaker_face_tracks()` builds. A speaker missing
    from it simply votes with two methods instead of three.

    `details_out`, when given, is filled with the whole ballot per speaker
    (`GenderVote.to_jsonable()`), the same `metadata_out={}` convention
    `backend/downloader.py` uses. Returns ``{speaker_id: "male"|"female"}``.
    """
    tracks = speaker_face_tracks or {}
    results: Dict[str, str] = {}
    for spk, ref_path in speaker_refs.items():
        has_clip = bool(ref_path) and Path(ref_path).exists()
        trk = tracks.get(spk)
        if not has_clip and not trk:
            log.warning("gender: no reference audio and no face track for %s; "
                        "skipping", spk)
            continue
        frames, bboxes = trk if trk else (None, None)
        vote = vote_gender(Path(ref_path) if has_clip else None, device=device,
                           video=video, face_frames=frames, face_bboxes=bboxes)
        results[spk] = vote.gender
        if details_out is not None:
            details_out[spk] = vote.to_jsonable()
    return results
