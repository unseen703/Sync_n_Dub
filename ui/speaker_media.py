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
import re
import subprocess
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


#: Where the pre-cast preview clips land. NOT `voices/<stem>.refs/`: that glob
#: is how `ui.speakers._reference_clips` (and the voices stage) recognise the
#: references the pipeline itself wrote, and a preview must never pass for one.
PREVIEW_SUBDIR = "voices/previews"

#: Longest preview. Long enough to judge a voice, short enough to stay a clip.
PREVIEW_MAX_S = 8.0

_RTTM_LINE = re.compile(r"^SPEAKER\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+"
                        r"\S+\s+\S+\s+(\S+)")


def _turns(job_root: Path) -> List[tuple]:
    """`(start, end, speaker)` from the first RTTM in `diarization/`.

    Same pick as `ui.speakers._from_rttm`, so the preview and the speech
    totals on the row describe the same turns.
    """
    d = job_root / "diarization"
    for p in sorted(d.glob("*.rttm")) if d.is_dir() else []:
        try:
            lines = p.read_text().splitlines()
        except OSError:
            continue
        out = []
        for line in lines:
            m = _RTTM_LINE.match(line.strip())
            if m:
                s = float(m.group(1))
                out.append((s, s + float(m.group(2)), m.group(3)))
        if out:
            return out
    return []


def _solo_spans(turns: List[tuple]) -> Dict[str, List[tuple]]:
    """Per speaker, the stretches nobody else is talking over."""
    out: Dict[str, List[tuple]] = {}
    for s, e, spk in turns:
        spans = [(s, e)]
        for s2, e2, spk2 in turns:
            if spk2 == spk:
                continue
            nxt = []
            for a, b in spans:
                if e2 <= a or s2 >= b:
                    nxt.append((a, b))
                    continue
                if s2 > a:
                    nxt.append((a, s2))
                if e2 < b:
                    nxt.append((e2, b))
            spans = nxt
        out.setdefault(spk, []).extend(spans)
    return out


def _preview_source(job_root: Path) -> Optional[Path]:
    """The cleanest source audio on disk: vocals stem, else the dialogue mix."""
    for pattern in ("stems/*/*/vocals.wav", "audio/*.dialogue.16k.mono.wav",
                    "audio/*.16k.mono.wav"):
        hit = next(iter(sorted(job_root.glob(pattern))), None)
        if hit is not None:
            return hit
    return None


def speaker_preview_paths(job_root: Path, *,
                          regenerate: bool = False) -> Dict[str, Path]:
    """`{speaker: clip.wav}` — the speaker's own voice, cut from the source.

    This exists because the voices stage only writes its reference clips
    AFTER the cast is chosen, and the operator needs to hear a speaker to
    choose their gender. Picks the longest stretch of that speaker talking
    alone, capped at `PREVIEW_MAX_S`. No model runs. Returns only what
    exists afterwards; a speaker with no solo stretch of `MIN_SAMPLE_S` is
    absent rather than given a link to nothing.
    """
    out_dir = job_root / PREVIEW_SUBDIR
    existing = {p.stem: p for p in sorted(out_dir.glob("*.wav"))} \
        if out_dir.is_dir() and not regenerate else {}
    if existing:
        return existing

    src = _preview_source(job_root)
    spans = _solo_spans(_turns(job_root))
    if src is None or not spans:
        return existing

    from backend.media_io import _ffmpeg_bin
    out_dir.mkdir(parents=True, exist_ok=True)
    for spk, items in sorted(spans.items()):
        a, b = max(items, key=lambda ab: ab[1] - ab[0])
        if b - a < MIN_SAMPLE_S:
            continue
        dst = out_dir / f"{spk}.wav"
        cmd = [_ffmpeg_bin(), "-y", "-loglevel", "error",
               "-ss", f"{a:.3f}", "-t", f"{min(b - a, PREVIEW_MAX_S):.3f}",
               "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("speaker preview: ffmpeg failed for %s (%s)", spk, e)
            continue
        if dst.is_file():
            existing[spk] = dst
    log.info("speaker previews: %d written to %s", len(existing), out_dir)
    return existing
