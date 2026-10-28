"""Speaker gender prediction (voice assignment signal #1).

This is a HARD FILTER downstream, not a weight: `backend/voices.py` matches
a speaker only against catalog voices of the predicted gender, so a wrong
label here cannot be recovered by the cosine ranking behind it.

Three predictors vote and the majority wins:

  1. **wav2vec2-XLSR-53** (`alefiury/wav2vec2-large-xlsr-53-gender-
     recognition-librispeech`), a sequence-classification head on a
     multilingual encoder. Reads the waveform.
  2. **F0 median pitch** via librosa.pyin -- below `F0_THRESHOLD_HZ` (165)
     is male. One scalar, no model.
  3. **The speaker's FACE**, insightface's `genderage.onnx` voted over
     frames of the face track ASD attributed to this speaker. CPU only
     (see `FACE_GENDER_DEVICE`), and silent without a video and a track.

Accuracy measured end to end on 25 clips: wav2vec2 alone 16/25, the
three-way majority 18/25. The vote fixes uncorrelated failures (a 436 Hz
scream, an unvoiced clip, a missing model) but not correlated ones -- the
two audio methods read the same waveform, so on one lecturer whose median
F0 straddles 165 Hz they fail together and outvote the single correct face
2-1. The face was right on all 23 clips where it answered.

Ties happen only when one voter abstains and the other two disagree.
`TIEBREAK_ORDER` settles it, face first.

No votes at all returns `FALLBACK_GENDER` ("male") at WARNING;
`backend/voices.py` then casts a voice recorded in the target language of
that gender.

`predict_gender` always returns `"male"` or `"female"`, never None, because
the caller filters the voice catalog on whatever it gets. `vote_gender`
returns the whole ballot, which stage 9d records for auditing.
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

#: Everything here infers at 16 kHz mono: the rate the feature extractor
#: declares (`preprocessor_config.json`). Resampling elsewhere shifts every
#: learned filter.
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

#: Seconds per inference window. Self-attention cost grows with the square
#: of input length and a stage-9d reference clip is unbounded, so chunking
#: makes the cost linear and averages the answer over the whole clip.
W2V2_CHUNK_S = 10.0

#: Under this, there is nothing to classify. 0.1 s is below the encoder's
#: receptive field, so a shorter clip raises a shape error.
MIN_AUDIO_S = 0.1

_w2v2_model = None
_w2v2_extractor = None
_w2v2_id2label: Dict[int, str] = {}
_w2v2_device = "cpu"
_w2v2_ready: Optional[bool] = None  # None = not tried yet


def _ensure_w2v2(device: str = "cpu") -> bool:
    """Lazy-load the classifier. Returns True on success, and False once and
    for all on failure -- a missing model must not be retried per segment."""
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

        # The label ORDER is read off the model's own config, never
        # assumed. This checkpoint is {0: female, 1: male} -- the opposite
        # of the alphabetical guess -- and hard-coding it the other way
        # gives a confident, fully inverted voice filter with no error.
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
    """`{"male": p, "female": p}` for one clip, or None if it cannot answer.

    Probabilities are averaged over `W2V2_CHUNK_S` windows, weighted by each
    window's length. Returning scores rather than only the label lets a
    caller tell a 0.51 that happens to be right from a 0.99.
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
    # leaves nothing, the clip was shorter than one window and is the chunk.
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

    Returns `"male"` or `"female"`, or `None` when the model is unavailable
    or the clip unusable, so a failure cannot be mistaken for a prediction.
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
# A proxy for sex rather than a classifier of it -- a contralto, a tenor, a
# child, a whisper or music bleeding into the clip all move the median
# without moving the speaker. It earns its place as one cheap scalar whose
# errors do not correlate with a neural classifier's.
#
# Measured on this host: 95.6% on the 45 `indic-speak` catalog clips, 72.7%
# on 11 ASD-gated natural-speech tracks.

#: Below this, male. 165 Hz is the conventional split and the one every
#: number quoted in this module was measured at; moving it invalidates them.
F0_THRESHOLD_HZ = 165.0


def predict_gender_f0(wav_path: Path, *, sr: int = TARGET_SR) -> Optional[str]:
    """Predict gender from median fundamental frequency via librosa.pyin.

    Returns `"male"`, `"female"`, or `None` when the clip has no voiced
    frames. None is an abstention, not a guess.
    """
    import numpy as np

    try:
        # Inside the try, not above it: librosa reaches this env only as a
        # transitive dependency of the optional `funasr`, so a lean install
        # can legitimately lack it and an ImportError here would take down
        # stage 9d over a method allowed to abstain.
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
    """The median voiced F0, or None.

    Recorded alongside the vote: a value sitting on the 165 Hz line is the
    signature of the speaker this vote exists for.
    """
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
# The audio methods answer "what does this voice sound like"; the face
# answers "who is this", and they fail on different inputs -- singing,
# whisper, phone-band audio or background music confuse the voice and not
# the face; off-screen, profile and darkness do the reverse.
#
# It votes only when the caller supplies a face track ASD attributed to this
# speaker. Without one it abstains and the vote is between the two audio
# methods.
#
# Reuses `genderage.onnx` from the buffalo_l pack stage 5 already downloads
# -- the detector adapter asks for `allowed_modules=("detection",)`, so it
# otherwise sits unused. No new model, no new dependency.
#
# It cannot tell you which face is the speaker. That is
# `backend/assign_faces.py`'s job; pass a track ASD attributed to the
# speaker, not "the biggest face".

#: CPU, always. `genderage.onnx` is a few MB over a dozen 112x112 crops, so
#: the GPU buys nothing, while the CUDA EP costs ~460 MiB resident for the
#: life of the process -- and stage 9d sits upstream of a lip-sync stage
#: already within 100 MiB of the 16 GiB ceiling.
FACE_GENDER_DEVICE = "cpu"

_facega_app = None
_facega_device: Optional[str] = None
_facega_ready: Optional[bool] = None

#: Below this, say nothing rather than guess. A face in profile or at 40 px
#: still produces a confident-looking label.
FACE_GENDER_MIN_VOTES = 3
FACE_GENDER_MIN_BOX_PX = 60


def _ensure_face_genderage(device: str = FACE_GENDER_DEVICE):
    """Load buffalo_l with the genderage module switched on.

    Cached, including the failure: stage 9d calls this once per speaker and
    a missing model must not re-attempt the download for each.
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

    Returns `(gender, confidence, n_votes)`; `(None, 0.0, n)` when the track
    gave too little to trust. `confidence` is the share of votes the winner
    took, not a model probability -- a 3-0 sweep of a blurry profile still
    reads 1.0, so weigh it with `n_votes`.

    Samples evenly across the track rather than taking the first frames,
    which are the most likely to be motion-blurred.
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
            # Detect on the FULL FRAME, then match the track's box to a
            # detection. Cropping to the track box first returns zero faces
            # -- RetinaFace regresses the box from a wider receptive field,
            # so a tight crop is out of distribution however much margin is
            # added.
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


#: A track box and a fresh detection of the same face rarely agree exactly.
#: Below this they are not the same face, and voting on the wrong one is
#: worse than abstaining.
FACE_MATCH_MIN_IOU = 0.3


def _best_iou(box, faces):
    """The detected face that best overlaps `box`, or None if none really do.

    Identity on a multi-face shot is the whole difficulty: "the biggest
    detection" picks a different person most of the time on a shot with
    seven faces.
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

#: What a speaker gets when nothing could answer. A placeholder, not a
#: prediction, and logged at WARNING every time. `backend/voices.py` turns
#: it into a voice recorded in the target language of this gender
#: (`rationale.basis == "language+gender"`).
FALLBACK_GENDER = "male"

#: Reached only when one voter abstains and the other two split 1-1. Face
#: leads: in every measurement on this host it was right whenever it
#: answered, including the one lecturer both audio methods get wrong.
TIEBREAK_ORDER = ("face", "w2v2", "f0")

#: The order votes are reported in, for humans reading a log line. Not the
#: tiebreak order.
METHODS = ("w2v2", "f0", "face")


@dataclass
class GenderVote:
    """One speaker's ballot, kept whole so a finished run can be audited.

    `gender` is always "male" or "female". `decided_by` separates a 3-0
    sweep, a 1-1 tiebreak and a no-evidence fallback -- all three produce a
    usable label and only one is worth trusting.
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

    `wav_path` drives methods 1 and 2. Method 3 needs all of `video`,
    `face_frames` and `face_bboxes` -- the face track ASD attributed to THIS
    speaker. Passing "the biggest face" is a guess and on a multi-face shot
    usually the wrong person, so the caller must have done the attribution.

    Every method may abstain (None). Abstentions do not vote and do not
    count toward the majority.
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

    # Two distinct things: `disagreement` is voters contradicting each
    # other. `unanimous` additionally requires that all three voted -- a 2-0
    # with the face abstaining is agreement, not a sweep.
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
        # drifted.
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

    `speaker_face_tracks` maps a speaker id to that speaker's
    `(frames, bboxes)`, built by
    `backend.assign_faces.speaker_face_tracks()`. A speaker missing from it
    votes with two methods instead of three.

    `details_out`, when given, is filled with the whole ballot per speaker
    (`GenderVote.to_jsonable()`). Returns `{speaker_id: "male"|"female"}`.
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
