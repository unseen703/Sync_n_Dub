"""Face crop and voice samples for the speaker review card.

Cuts a face crop from a frame the pipeline already attributed to each speaker,
picks a representative dubbed clip, and cuts a pre-cast preview from the source
audio. Caches all three in the job folder. No model runs here.
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
#: detector box makes two speakers hard to tell apart.
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

    The LONGEST clip, not the first: an opening cue is often a one-word answer.
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


#: Candidates must be at least this many frames apart, or the top-3 by ASD
#: score collapse onto the same instant (speech bursts cluster, and three
#: near-identical frames defeat the point of offering a choice).
CANDIDATE_MIN_GAP_FRAMES = 15

#: How many candidate crops to offer per speaker, when ASD scores let us rank.
CANDIDATES_PER_SPEAKER = 3

_CROP_NAME = re.compile(r"^(?P<spk>.+)\.(?P<idx>\d+)\.jpg$")


def face_crop_paths(job_root: Path, *, video: Optional[Path] = None,
                    regenerate: bool = False) -> Dict[str, List[Path]]:
    """`{speaker: [crop.0.jpg, crop.1.jpg, ...]}`, writing any that are missing.

    Up to `CANDIDATES_PER_SPEAKER` frames, ranked by stage 6's own ASD score
    for the speaker's winning track -- a frame ASD is confident is mid-speech
    beats the middle-of-track frame picked blind, and three options let the
    operator correct for an awkward expression ASD can't see. Falls back to
    the single middle frame when no `*.scores.npz` is on disk (an older job,
    or an ASD model that did not write one): one honest candidate beats none.

    Returns only what exists on disk afterwards. Stage 6 omits a speaker whose
    on-screen total is too short to trust, so inventing a crop would assert
    evidence the pipeline refused to.
    """
    out_dir = job_root / CROP_SUBDIR
    existing: Dict[str, List[Path]] = {}
    if out_dir.is_dir() and not regenerate:
        for p in sorted(out_dir.glob("*.jpg")):
            m = _CROP_NAME.match(p.name)
            if m:
                existing.setdefault(m.group("spk"), []).append(p)
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
        import numpy as np

        from backend.faces.faces import speaker_face_tracks
        from backend.faces.faces import FaceTracks

        tracks = FaceTracks.load_npz(npz)
        owned = speaker_face_tracks(_read(assign) or [], tracks,
                                    with_track_id=True)
    except Exception as e:                              # noqa: BLE001
        log.warning("speaker crops: could not resolve faces (%s)", e)
        return existing

    # Best-effort: a speaker simply gets the middle-frame fallback if this
    # can't be loaded, never a hard failure for the whole crop pass.
    asd_scores = {}
    try:
        from backend.asd.asd import find_scores_npz, load_scores_npz
        scores_path = find_scores_npz(job_root / "asd")
        if scores_path is not None:
            asd_scores = load_scores_npz(scores_path).per_track
    except Exception as e:                              # noqa: BLE001
        log.info("speaker crops: no ASD scores available (%s); "
                 "falling back to middle-frame crops", e)

    def _pick_frames(trk: str, frames: "np.ndarray") -> List[int]:
        sc = asd_scores.get(trk)
        if sc is None or len(sc.frames) == 0:
            # No ASD evidence for this track -- the one middle frame, as before.
            return [int(frames[len(frames) // 2])]
        valid = set(int(f) for f in frames)
        ranked = sorted(
            ((float(s), int(f)) for f, s in zip(sc.frames, sc.scores)
             if int(f) in valid),
            reverse=True,
        )
        picked: List[int] = []
        for _, f in ranked:
            if all(abs(f - p) >= CANDIDATE_MIN_GAP_FRAMES for p in picked):
                picked.append(f)
            if len(picked) == CANDIDATES_PER_SPEAKER:
                break
        return picked or [int(frames[len(frames) // 2])]

    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    cap = cv2.VideoCapture(str(video))
    for spk, (trk, frames, bboxes) in sorted(owned.items()):
        frames = np.asarray(frames)
        bboxes = np.asarray(bboxes)
        if frames.size == 0:
            continue
        frame_to_idx = {int(f): i for i, f in enumerate(frames)}
        paths: List[Path] = []
        for idx, frame_no in enumerate(_pick_frames(trk, frames)):
            bi = frame_to_idx.get(frame_no)
            if bi is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in bboxes[bi][:4]]
            mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
            cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            dst = out_dir / f"{spk}.{idx}.jpg"
            cv2.imwrite(str(dst), frame[cy1:cy2, cx1:cx2],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            paths.append(dst)
        if paths:
            existing[spk] = paths
    cap.release()
    log.info("speaker crops: %d speaker(s) written to %s",
             len(existing), out_dir)
    return existing


#: Where pre-cast preview clips land. NOT `voices/<stem>.refs/` — that glob is
#: how the voices stage recognises its own references, and a preview must never
#: pass for one.
PREVIEW_SUBDIR = "voices/previews"

#: Longest preview. Long enough to judge a voice, short enough to stay a clip.
PREVIEW_MAX_S = 8.0

_RTTM_LINE = re.compile(r"^SPEAKER\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+"
                        r"\S+\s+\S+\s+(\S+)")


def _turns(job_root: Path) -> List[tuple]:
    """`(start, end, speaker)` from the first RTTM in `diarization/`.

    Same pick as `server.service.speakers._from_rttm`, so the preview and the row's speech
    totals describe the same turns.
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

    The voices stage only writes reference clips after the cast is chosen, but
    the operator needs to hear a speaker to pick their gender. Takes the
    longest stretch of that speaker talking alone, capped at `PREVIEW_MAX_S`.
    A speaker with no solo stretch of `MIN_SAMPLE_S` is absent rather than
    linked to nothing.
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

    from backend.common.media_io import _ffmpeg_bin
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
