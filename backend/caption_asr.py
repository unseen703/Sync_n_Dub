"""Use a video's published captions as the transcript instead of running ASR.

Picks a manual caption track (automatic ones are YouTube's own ASR and are
opt-in), checks it is usable, and reshapes its cues onto the diarization turns
so the result has the same shape turn-sliced ASR produces. Returns None for
every "no" answer, leaving the caller to run ASR.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from models.asr.base import ASRResult, ASRSegment

log = logging.getLogger(__name__)

#: Above this, the track is YouTube's rolling "karaoke" style where each cue
#: restates the previous one's tail, which duplicates whole phrases in the dub.
#: Measured by `backend.transcripts.rolling_duplication_ratio`.
MAX_ROLLING_DUPLICATION = 0.35

#: Below this the track covers too little speech to be a transcript — forced
#: captions on a few signs, or a partial upload. ASR is better than dubbing a
#: fraction of the video.
MIN_TURN_COVERAGE = 0.60


def pick_caption_track(
    subtitles: Optional[Dict[str, Dict[str, str]]],
    *,
    language: str = "en",
    allow_auto: bool = False,
) -> Optional[Dict[str, str]]:
    """The usable caption track for `language`, or None.

    `subtitles` is what `downloader.download_youtube` fills in:
    `{lang: {"path": ..., "source": "manual"|"auto"}}`.

    Keys are matched by BASE code — YouTube publishes manual English under
    `en-GB` and even channel-suffixed keys like `en-LUU0EuDKgKo`, where asking
    for plain `en` returns an auto-translation instead.
    """
    if not subtitles:
        return None
    base = (language or "en").split("-")[0].lower()
    manual, auto = [], []
    for key, entry in subtitles.items():
        if not entry or not entry.get("path"):
            continue
        if str(key).split("-")[0].lower() != base:
            continue
        (manual if entry.get("source") == "manual" else auto).append((key, entry))

    for pool, kind in ((manual, "manual"), (auto, "auto")):
        if not pool:
            continue
        if kind == "auto" and not allow_auto:
            log.info("captions: a %s %r track exists but only manual ones are "
                     "accepted (it is YouTube's own ASR; pass "
                     "--captions-allow-auto to use it anyway)", kind, base)
            return None
        key, entry = sorted(pool)[0]
        return {"path": entry["path"], "source": kind, "lang_key": key}
    return None


def caption_quality_problem(path: Path) -> Optional[str]:
    """Why this track should not be trusted as a transcript, or None."""
    from .transcripts import parse_vtt, rolling_duplication_ratio

    p = Path(path)
    if not p.is_file():
        return f"caption file missing at {p}"
    try:
        cues = parse_vtt(p) if p.suffix.lower() == ".vtt" else None
        if cues is None:
            from .transcripts import load_transcript
            cues = load_transcript(p)
    except Exception as e:                               # noqa: BLE001
        return f"could not parse {p.name}: {e}"
    if not cues:
        return f"{p.name} parsed to 0 cues"
    if not any(c.text.strip() for c in cues):
        return f"{p.name} has cues but no text"
    try:
        dup = rolling_duplication_ratio(p)
    except Exception:                                    # noqa: BLE001
        dup = 0.0
    if dup > MAX_ROLLING_DUPLICATION:
        return (f"{p.name} repeats {dup:.0%} of its text between cues "
                f"(rolling auto-caption style); it would duplicate phrases "
                f"in the dub")
    return None


def captions_to_asr_result(
    caption_path: Path,
    *,
    audio_path: Path,
    turns: Optional[list] = None,
    language: str = "en",
    source: str = "manual",
) -> ASRResult:
    """Build an `ASRResult` from a caption file, shaped like turn-sliced ASR.

    With `turns`, one segment per diarized turn carrying that turn's speaker,
    identical to `backend.asr.transcribe(..., turns=...)`. Without `turns`, one
    segment per cue and no speaker.
    """
    from .transcripts import load_transcript

    cues = load_transcript(Path(caption_path))
    cues = [c for c in cues if c.text and c.text.strip()]

    if not turns:
        segments = [
            ASRSegment(id=f"SEG_{i:04d}", start=float(c.start), end=float(c.end),
                       text=c.text.strip(), speaker=None, language=language)
            for i, c in enumerate(cues)
        ]
        return ASRResult(
            audio_path=Path(audio_path), model_name=f"captions:{source}",
            language=language, segments=segments,
            extra={"captions": {"source": source, "cues": len(cues),
                                "path": str(caption_path)}})

    # Each cue goes to exactly one turn, the one it overlaps most. Assigning it
    # to every turn it touches repeats the sentence once per speaker.
    buckets: List[List[str]] = [[] for _ in turns]
    unplaced = 0
    for c in cues:
        best_i, best_ov = None, 0.0
        for i, t in enumerate(turns):
            ov = min(float(c.end), float(t.end)) - max(float(c.start), float(t.start))
            if ov > best_ov:
                best_i, best_ov = i, ov
        if best_i is None:
            unplaced += 1
            continue
        buckets[best_i].append(c.text.strip())

    segments = [
        ASRSegment(id=f"SEG_{i:04d}", start=float(t.start), end=float(t.end),
                   text=" ".join(buckets[i]).strip(),
                   speaker=getattr(t, "speaker", None), language=language)
        for i, t in enumerate(turns)
    ]
    filled = sum(1 for s in segments if s.text)
    coverage = filled / len(segments) if segments else 0.0
    log.info("captions: %d cue(s) -> %d/%d turn(s) carry text (%.0f%% coverage)"
             "%s", len(cues), filled, len(segments), 100 * coverage,
             f", {unplaced} cue(s) outside every turn" if unplaced else "")
    return ASRResult(
        audio_path=Path(audio_path), model_name=f"captions:{source}",
        language=language, segments=segments,
        extra={"captions": {"source": source, "cues": len(cues),
                            "path": str(caption_path),
                            "turn_coverage": round(coverage, 4),
                            "cues_outside_turns": unplaced}})


def transcript_from_captions(
    subtitles: Optional[Dict[str, Dict[str, str]]],
    *,
    audio_path: Path,
    turns: Optional[list] = None,
    language: str = "en",
    allow_auto: bool = False,
) -> Optional[ASRResult]:
    """The whole decision in one call: a usable ASRResult, or None for ASR.

    Never raises — the caller is choosing between this and running ASR, so a
    caption problem must cost the saving, not the run.
    """
    track = pick_caption_track(subtitles, language=language, allow_auto=allow_auto)
    if not track:
        return None

    problem = caption_quality_problem(Path(track["path"]))
    if problem:
        log.info("captions: not using the %s %r track -- %s; running ASR",
                 track["source"], track["lang_key"], problem)
        return None

    try:
        result = captions_to_asr_result(
            Path(track["path"]), audio_path=audio_path, turns=turns,
            language=language, source=track["source"])
    except Exception as e:                               # noqa: BLE001
        log.warning("captions: could not build a transcript from %s (%s); "
                    "running ASR", track["path"], e)
        return None

    if turns:
        cov = float(result.extra.get("captions", {}).get("turn_coverage", 0.0))
        if cov < MIN_TURN_COVERAGE:
            log.info("captions: the %s %r track covers only %.0f%% of the "
                     "diarized turns (floor %.0f%%) -- partial or forced "
                     "captions; running ASR instead",
                     track["source"], track["lang_key"], 100 * cov,
                     100 * MIN_TURN_COVERAGE)
            return None
    return result
