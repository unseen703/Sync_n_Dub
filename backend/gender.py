"""Speaker gender prediction (plan §8 — voice assignment signal #1).

This answer is a HARD FILTER downstream, not a weight: `backend/voices.py`
matches a speaker only against catalog voices of the predicted gender, so a
wrong label here cannot be recovered by the cosine ranking behind it. It is
the single input most able to ruin a dub while every stage reports success —
CLAUDE.md §3 records one all-male lecture that shipped in a female voice.

Two methods. **F0 is primary**, and that is a measured choice, not a default:

  1. **F0 pitch threshold** (primary).  Median fundamental frequency via
     librosa.pyin; below 165 Hz → male, above → female.

  2. **ECAPA-TDNN + logistic regression** (second opinion only).  SpeechBrain's
     ECAPA encoder plus a logreg head (moorlee/gender-voice-classifier-ecapa),
     advertised at 99.2% on Hindi and Tamil.

MEASURED ON THIS HOST, 2026-10-17, against the 45 `indic-speak` catalog voices
whose gender is known (22 male / 23 female), using their own reference clips in
`data/voice_refs/`:

    method                     overall   male recall   female recall
    F0 @165 Hz                   95.6%        90.9%         100.0%
    ECAPA @0.05 (was in use)     44.4%        90.9%           0.0%
    ECAPA @0.50 ("global")       48.9%       100.0%           0.0%

ECAPA is not merely mis-thresholded — it carries no usable signal here. The two
classes overlap almost completely and point the wrong way (true males median
p=0.0014, true females median p=0.0005), so the best accuracy reachable with
ANY threshold is 55.6%, against a 51% majority-class baseline. No cut-off fixes
a feature with no separation; that is why the fix is not a threshold change.

The likely cause is the version mismatch sklearn itself warns about on every
load — `StandardScaler` and `LogisticRegression` pickled with 1.9.0, unpickled
under 1.7.2, "may lead to breaking code or invalid results". A scaler whose
state does not survive the round trip yields exactly this shape: near-zero
probabilities everywhere, no class separation. Until the pin is resolved and
the table above re-measured, ECAPA does not get a vote.

CAVEAT on the measurement: those 45 clips are TTS renders, not natural speech,
so 95.6% is an F0 number for synthetic voices. It agreed with pitch on the real
speakers it has been checked against (two Dark Knight mob characters at 149.9
and 106.9 Hz, both male, both of which ECAPA called female). Re-measure on
natural speech before treating 95.6% as the production figure.

Both methods accept a WAV path and return ``"male"`` or ``"female"``.
``predict_gender`` decides on F0 and logs when ECAPA disagrees.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method 1: ECAPA-TDNN + logistic regression
# ---------------------------------------------------------------------------

_ecapa_encoder = None
_ecapa_scaler = None
_ecapa_clf = None
_ecapa_threshold: float = 0.5
_ecapa_ready: Optional[bool] = None  # None = not tried yet


def _ensure_ecapa(device: str = "cpu") -> bool:
    """Lazy-load the ECAPA backbone + logreg head. Returns True on success."""
    global _ecapa_encoder, _ecapa_scaler, _ecapa_clf, _ecapa_threshold, _ecapa_ready
    if _ecapa_ready is not None:
        return _ecapa_ready
    try:
        import json
        import joblib
        import torch  # noqa: F401
        from huggingface_hub import hf_hub_download
        from speechbrain.inference.speaker import EncoderClassifier

        repo = "moorlee/gender-voice-classifier-ecapa"
        scaler_path = hf_hub_download(repo, "scaler.pkl")
        clf_path = hf_hub_download(repo, "logreg.pkl")
        thresh_path = hf_hub_download(repo, "thresholds.json")

        _ecapa_scaler = joblib.load(scaler_path)
        _ecapa_clf = joblib.load(clf_path)
        # Fix for sklearn >=1.6: LogisticRegression dropped `multi_class`.
        if not hasattr(_ecapa_clf, "multi_class"):
            _ecapa_clf.multi_class = "auto"
        with open(thresh_path) as f:
            _ecapa_threshold = json.load(f).get("female_optimized", 0.5)

        _ecapa_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        _ecapa_encoder.eval()
        _ecapa_ready = True
        log.info("gender: ECAPA classifier loaded (device=%s)", device)
    except Exception as exc:
        log.info("gender: ECAPA classifier unavailable (%s); will use F0 fallback", exc)
        _ecapa_ready = False
    return _ecapa_ready


def predict_gender_ecapa(wav_path: Path, *, device: str = "cpu") -> Optional[str]:
    """Predict gender using ECAPA-TDNN embeddings + logistic regression.

    Returns ``"male"`` or ``"female"``, or ``None`` if the model isn't available.
    """
    if not _ensure_ecapa(device):
        return None
    import numpy as np
    import soundfile as sf
    import torch

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1,
                          int(len(audio) * 16000 / sr)).astype(np.float64)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")

    with torch.no_grad():
        emb = _ecapa_encoder.encode_batch(
            torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
        ).squeeze().cpu().numpy()

    prob_female = float(
        _ecapa_clf.predict_proba(
            _ecapa_scaler.transform(emb.reshape(1, -1))
        )[0, 1]
    )
    return "female" if prob_female >= _ecapa_threshold else "male"


# ---------------------------------------------------------------------------
# Method 2: F0 pitch threshold (fallback)
# ---------------------------------------------------------------------------

_F0_THRESHOLD_HZ = 165.0


def predict_gender_f0(wav_path: Path, *, sr: int = 16000) -> str:
    """Predict gender from median fundamental frequency via librosa.pyin.

    Returns ``"male"`` or ``"female"``.  Never returns None — this is the
    last-resort fallback.
    """
    import librosa
    import numpy as np

    y, file_sr = librosa.load(str(wav_path), sr=sr, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr, frame_length=2048,
    )
    voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
    if len(voiced_f0) == 0:
        log.warning("gender(f0): no voiced frames in %s; defaulting to male", wav_path)
        return "male"

    median_f0 = float(np.median(voiced_f0))
    gender = "female" if median_f0 >= _F0_THRESHOLD_HZ else "male"
    log.debug("gender(f0): %s — median_f0=%.1f Hz → %s", wav_path.name, median_f0, gender)
    return gender


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

#: Set to "ecapa" to put the neural classifier back in charge, once the
#: sklearn pin is fixed AND the table in this module's docstring has been
#: re-measured. "f0" (the default) decides on pitch.
GENDER_METHOD_ENV = "DUBBING_GENDER_METHOD"


def predict_gender(wav_path: Path, *, device: str = "cpu") -> str:
    """Predict speaker gender. F0 decides; ECAPA is logged, not obeyed.

    Always returns ``"male"`` or ``"female"`` — never None.

    The two methods are both evaluated when ECAPA is available, because a
    disagreement is the cheapest available warning that one of them has
    drifted. Only F0 is acted on: see the measured table in the module
    docstring for why. A disagreement is logged at WARNING precisely so it
    cannot pass silently the way the old all-female output did.
    """
    import os

    prefer = os.environ.get(GENDER_METHOD_ENV, "f0").strip().lower()

    f0_result = predict_gender_f0(wav_path)
    ecapa_result = predict_gender_ecapa(wav_path, device=device)

    if prefer == "ecapa" and ecapa_result is not None:
        result, method = ecapa_result, "ecapa"
    else:
        result, method = f0_result, "f0"

    if ecapa_result is not None and ecapa_result != f0_result:
        log.warning(
            "gender: %s — methods DISAGREE (f0=%s, ecapa=%s); using %s=%s. "
            "This label is a hard filter on the voice catalog, so check it "
            "before trusting the dub's voices.",
            wav_path.name, f0_result, ecapa_result, method, result,
        )
    else:
        log.info("gender: %s → %s (method=%s)", wav_path.name, result, method)
    return result


def predict_genders(
    speaker_refs: Dict[str, Path],
    *,
    device: str = "cpu",
) -> Dict[str, str]:
    """Predict gender for every speaker reference clip.

    Returns ``{speaker_id: "male"|"female"}``.
    """
    results: Dict[str, str] = {}
    for spk, ref_path in speaker_refs.items():
        if ref_path and Path(ref_path).exists():
            results[spk] = predict_gender(Path(ref_path), device=device)
        else:
            log.warning("gender: no reference audio for %s; skipping", spk)
    return results


# ---------------------------------------------------------------------------
# Method 3: the speaker's FACE
# ---------------------------------------------------------------------------
#
# Pitch answers "what does this voice sound like"; the face answers "who is
# this". They fail on different inputs, which is the whole reason to have both:
# F0 is confused by singing, whisper, a child, a phone-band recording or
# background music bleeding into the reference clip, and none of those touch a
# face crop. Conversely the face is unavailable off-screen, in profile, or in
# the dark, where pitch is unbothered.
#
# This reuses `genderage.onnx`, which insightface ALREADY downloads as part of
# the buffalo_l pack that stage 5 uses for detection -- it is sitting unused in
# ~/.insightface/models/buffalo_l/. The detector adapter asks for
# `allowed_modules=("detection",)`, so nothing ever loads it. No new model, no
# new dependency.
#
# WHAT THIS DOES NOT DO: it cannot tell you which face is the speaker. That
# link is `backend/assign_faces.py`'s job (ASD + Hungarian matching, written to
# `<job>/asd/*.assignments.json`), and today nothing consumes that file. Pass a
# track that ASD attributed to the speaker; passing "the biggest face" instead
# is a guess, and on a 7-faces-on-screen shot it is usually the wrong one.

_facega_app = None
_facega_ready: Optional[bool] = None

#: Below this, say nothing rather than guess. A face seen in profile or at
#: 40 px still produces a confident-looking label.
FACE_GENDER_MIN_VOTES = 3
FACE_GENDER_MIN_BOX_PX = 60


def _ensure_face_genderage(device: str = "cpu"):
    """Load buffalo_l with the genderage module switched ON."""
    global _facega_app, _facega_ready
    if _facega_ready is not None:
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
        _facega_app, _facega_ready = app, True
    except Exception as exc:                                 # noqa: BLE001
        log.info("gender(face): insightface genderage unavailable (%s: %s)",
                 type(exc).__name__, exc)
        _facega_app, _facega_ready = None, False
    return _facega_app


def predict_gender_face(
    video: Path,
    frames,                       # np.ndarray (N,) frame indices
    bboxes,                       # np.ndarray (N, 4) xyxy
    *,
    device: str = "cpu",
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
