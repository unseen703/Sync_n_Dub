"""Face crop and dubbed-voice sample for the speaker review card.

The review screen asks a human "is this the right voice for this person?",
which needs both halves on screen: the FACE the pipeline attributed to that
speaker, and the dub it will actually read in. `/api/jobs/{id}/speakers`
already carried the detected gender, tone, assigned voice and the speaker's
ORIGINAL reference clip; these two were the gap.

Both are derived on demand and cached in the job folder, because both are
cheap to recompute and expensive to carry around:

  * the crop is one decoded frame out of a video already on disk
  * the dub sample is a clip the TTS stage already wrote -- this picks one,
    it does not synthesise anything

NO MODEL RUNS HERE. The face a speaker owns was decided by stage 6 and read
back by `backend.assign_faces.speaker_face_tracks`; re-deciding it inside a
GET would mean a second, differently-wrong answer to a question the pipeline
already answered.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

#: Where crops land. Inside the job folder, per the one-run-one-folder rule.
CROP_SUBDIR = "faces/crops"

#: Padding around the face box, as a fraction of its size. A tight crop of a
#: detector box reads as a mugshot and makes two speakers hard to tell apart;
#: some surrounding context is what makes the thumbnail recognisable.
CROP_MARGIN = 0.35

#: A dub sample shorter than this is a breath or a filler word and tells the
#: operator nothing about the voice.
MIN_SAMPLE_S = 0.8


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def dub_sample_paths(merged: Dict[str, Any]) -> Dict[str, str]:
    """`{speaker: path}` — one representative dubbed clip each.

    The LONGEST clip, not the first: an opening cue is often a one-word
    answer, and the point of the sample is to judge a voice.
    """
    best: Dict[str, tuple] = {}
    for seg in merged.get("segments") or []:
        spk, path = seg.get("speaker"), seg.get("tts_audio")
        if not spk or not path:
            continue
        if (seg.get("qc") or {}).get("tts_error"):
            continue                      # a failed line has no audio to play
        dur = float(seg.get("tts_duration_s") or 0.0)
        if dur < MIN_SAMPLE_S:
            continue
        cur = best.get(str(spk))
        if cur is None or dur > cur[0]:
            best[str(spk)] = (dur, str(path))
    return {k: v[1] for k, v in best.items()}


def face_crop_paths(job_root: Path, *, video: Optional[Path] = None,
                    regenerate: bool = False) -> Dict[str, Path]:
    """`{speaker: crop.jpg}`, writing any that are missing.

    Returns only what exists on disk afterwards, so a speaker with no face
    evidence is simply absent rather than pointing at a file that is not
    there. That distinction matters: stage 6 deliberately omits a speaker
    whose on-screen total is too short to trust, and inventing a crop for
    them would assert evidence the pipeline refused to assert.
    """
    out_dir = job_root / CROP_SUBDIR
    existing = {}
    if out_dir.is_dir() and not regenerate:
        for p in out_dir.glob("*.jpg"):
            existing[p.stem] = p
        if existing:
            return existing

    assign = next(iter(sorted(job_root.glob("asd/*.assignments.json"))), None)
    npz = next(iter(sorted(job_root.glob("faces/*.npz"))), None)
    if assign is None or npz is None:
        log.info("speaker crops: no stage 5/6 output in %s; skipping",
                 job_root.name)
        return existing
    video = video or next(iter(sorted(job_root.glob("videos/*.mp4"))), None)
    if video is None or not Path(video).is_file():
        return existing

    try:
        import cv2
        import numpy as np

        from backend.assign_faces import speaker_face_tracks
        from models.faces import FaceTracks

        tracks = FaceTracks.load_npz(npz)
        owned = speaker_face_tracks(_read(assign) or [], tracks)
    except Exception as e:                              # noqa: BLE001
        log.warning("speaker crops: could not resolve faces (%s)", e)
        return existing

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    for spk, (frames, bboxes) in sorted(owned.items()):
        frames = np.asarray(frames)
        bboxes = np.asarray(bboxes)
        if frames.size == 0:
            continue
        # The MIDDLE of the track, not its first frame: a track usually
        # begins mid-pan or mid-cut, where the face is most likely blurred.
        k = len(frames) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[k]))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bboxes[k][:4]]
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        dst = out_dir / f"{spk}.jpg"
        cv2.imwrite(str(dst), frame[cy1:cy2, cx1:cx2],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        existing[spk] = dst
    cap.release()
    log.info("speaker crops: %d written to %s", len(existing), out_dir)
    return existing
