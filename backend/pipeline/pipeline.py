"""End-to-end orchestrator.

    input (URL or file)
        │
        ▼   backend.downloader.resolve_input
    video
        │
        ▼   backend.media_io.extract_audio (16 kHz mono WAV, for diarization + ASR)
    audio_16k
        │
        ▼   backend.stem_separation.separate_stems (on the source video/audio at native rate)
    stems.dialogue, stems.music_effects
        │
        ▼   backend.diarize.diarize (on stems.dialogue if present, else audio_16k)
    DiarizationResult + RTTM
        │
        ▼   backend.faces.detect_and_track (RetinaFace|MediaPipe + ByteTrack)
    FaceTracks
        │
        ▼   backend.asd.score_asd + assign_faces_to_turns (LoCoNet + Hungarian)
    SegmentAssignment[]
        │
        ▼   backend.asr.transcribe (Bodhan|WhisperX|IndicConformer)
    ASRResult (per-turn segments)
        │
        ▼   backend.align.align_asr (MMS-FA|WhisperX)   — word boundaries
    ASRResult with words[]
        │
        ▼   backend.translate.translate_asr (Bodhan)
    MTResult (per-segment target-language text)

Each post-diarization stage is behind a flag; the baseline pipeline still
stops at diarization for anyone who only wants that.
"""
from __future__ import annotations

import functools
import json
import logging
import shutil
import sys
from collections import Counter, namedtuple
from os import link as _os_link
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import jobs as _jobs
from backend.media.align.align import (align_asr, align_tts_segments, apply_alignment_to_asr,
                    attach_visemes)
from backend.asd import (SCORES_SUFFIX as ASD_SCORES_SUFFIX, assign_faces_to_turns,
                  assignments_to_jsonable, find_scores_npz, save_scores_npz,
                  score_asd)
from backend.asr import transcribe as asr_transcribe
from backend.translate import codemix_segments
from backend.diarize.diarize import diarize as diarize_stage
from backend.media.segment import resegment
from backend.diarize import refine_speakers as _refine_speaker_labels
from backend.media.downloader import resolve_input
from backend.media.duration import plan_segment_durations
from backend.faces import detect_and_track
from backend.lipsync import run_lipsync
from backend.common.media_io import extract_audio
from backend.media.mix import mix_dubbed_audio
from backend.media.stem_separation import separate_stems
from backend.asr.transcripts import evaluate_asr, evaluate_mt, write_report
from backend.translate import translate_asr, write_mt_outputs
from backend.tts import synthesize_segments
from backend.asd.gender import predict_genders
from backend.tts.voices import (
    assign_voices,
    assignments_to_jsonable as voice_assignments_to_jsonable,
    extract_speaker_references,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

#: Fitted duration-model coefficients (`<lang>.<voice>.json`). These are a
#: shared model asset, not per-run output: fitting one costs a full TTS
#: calibration pass, so a job reads them from here rather than owning a copy.
DURATION_MODEL_DIR = REPO_ROOT / "data" / "duration_model"


@dataclass
class PipelineResult:
    """Everything the orchestrator produced. Serializable to JSON.

    `dialogue_stem` and `me_stem` are None when stem separation was skipped —
    do NOT substitute the full mix, because a downstream stage would then
    treat the mixed audio as an M&E bed and re-mix dialogue over dialogue.
    """
    input: str
    video: Path
    audio_16k: Path
    dialogue_stem: Optional[Path]
    me_stem: Optional[Path]
    rttm: Path
    num_turns: int
    num_speakers: int
    diarization_model: str
    manifest: Path
    stems_skipped: bool = False

    # Job folder this run owns: `data/jobs/<job_id>/`. Every path above and
    # below is inside it.
    job_id: Optional[str] = None
    job_dir: Optional[Path] = None

    # Faces / ASD / ASR — populated when the corresponding stage ran.
    face_tracks_npz: Optional[Path] = None
    num_face_tracks: int = 0
    face_detector: Optional[str] = None
    face_tracker: Optional[str] = None

    asd_model: Optional[str] = None
    assignments_json: Optional[Path] = None
    num_on_screen: int = 0

    asr_model: Optional[str] = None
    asr_language: Optional[str] = None
    asr_json: Optional[Path] = None

    align_model: Optional[str] = None
    align_json: Optional[Path] = None
    num_aligned_words: int = 0

    mt_model: Optional[str] = None
    mt_target_language: Optional[str] = None
    mt_target_script: Optional[str] = None
    mt_json: Optional[Path] = None
    mt_txt: Optional[Path] = None

    # Code-mixing / script normalisation (plan §6).
    translit_model: Optional[str] = None
    codemix_retained_terms: int = 0

    # Duration-aware text adjustment (plan §7).
    duration_model_language: Optional[str] = None
    duration_model_voice: Optional[str] = None
    duration_estimated_violations: int = 0

    # Voice assignment per speaker + TTS (plan §8).
    voice_engine: Optional[str] = None
    voice_assignments_json: Optional[Path] = None
    speaker_references_dir: Optional[Path] = None
    tts_model: Optional[str] = None
    tts_out_dir: Optional[Path] = None
    tts_budget_violations: int = 0
    tts_errors: int = 0

    # Timing plan.
    timing_producer: Optional[str] = None
    timing_plan_json: Optional[Path] = None
    timing_summary: dict = field(default_factory=dict)

    # Final mix.
    mix_wav: Optional[Path] = None
    mix_mp4: Optional[Path] = None

    #: Stage 9b1 — the content register the TTS style is taken from, and
    #: which signal decided it ("metadata", "keywords", "llm" or "none").
    content_register: Optional[str] = None
    content_register_tier: Optional[str] = None

    # Lip-sync (plan §8).
    lipsync_model: Optional[str] = None
    lipsync_mp4: Optional[Path] = None
    lipsync_error: Optional[str] = None

    # True when phase one stopped after MT (stop_after_mt) and the post-MT
    # cast stages have not run yet -- the UI reads this to show the gender/
    # voice gate and offer "Cast & continue".
    awaiting_cast: bool = False

    # Ground-truth transcripts fetched alongside the video (lang → metadata).
    # `source` is "manual" (human-authored on YouTube), "auto" (YouTube ASR),
    # or "local" (sidecar file next to a local input video).
    subtitles: Dict[str, Dict[str, str]] = field(default_factory=dict)
    asr_eval_json: Optional[Path] = None
    mt_eval_json: Optional[Path] = None

    def to_json(self, path: Path) -> Path:
        def _enc(v):
            if isinstance(v, Path):
                return str(v)
            return v
        payload = {k: _enc(v) for k, v in asdict(self).items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        return path


# ---------------------------------------------------------------------------
# post-MT stages (9b .. 9h)
# ---------------------------------------------------------------------------

_Turn = namedtuple("_Turn", "speaker start end")


def _turns_from_merged(data: dict) -> List["_Turn"]:
    """Reconstruct diarization turns from the merged JSON's own segments, so the
    post-MT stages do not need a live DiarizationResult.

    `extract_speaker_references` and the `assign_voices` screen-time tally are
    the only consumers, and both read nothing but `.speaker`, `.start` and
    `.end` off a turn — so a three-field record duck-types in cleanly.
    Segments with no speaker label (full-audio ASR mode) are dropped: they
    would otherwise collapse into one bogus "None" speaker.
    """
    turns: List[_Turn] = []
    segments = data.get("segments", [])
    unlabelled = malformed = 0
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            unlabelled += 1
            continue
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        turns.append(_Turn(spk, start, end))
    # Dropping segments quietly would hand stage 9d a short (or empty) speaker
    # list and silently dub everything in one default voice, so say so.
    if unlabelled or malformed:
        log.warning(
            "post-MT turns: dropped %d/%d segment(s) — %d with no speaker "
            "label (full-audio ASR?), %d with unusable start/end",
            unlabelled + malformed, len(segments), unlabelled, malformed,
        )
    if segments and not turns:
        log.warning(
            "post-MT turns: no usable diarization turns in the merged JSON; "
            "speaker reference extraction and per-speaker voice assignment "
            "will be skipped"
        )
    return turns


@dataclass
class PostMTResult:
    """Whatever stages 9b..9h produced for one merged JSON."""
    mt_json: Optional[Path] = None
    translit_model: Optional[str] = None
    codemix_retained_terms: int = 0
    duration_estimated_violations: int = 0
    tts_model: Optional[str] = None
    tts_out_dir: Optional[Path] = None
    tts_errors: int = 0
    tts_budget_violations: int = 0
    mix_wav: Optional[Path] = None
    mix_mp4: Optional[Path] = None
    lipsync_mp4: Optional[Path] = None
    lipsync_error: Optional[str] = None
    # Remaining PipelineResult fields these stages own, so the caller can copy
    # the whole lot back without re-deriving any of it.
    duration_model_language: Optional[str] = None
    duration_model_voice: Optional[str] = None
    voice_engine: Optional[str] = None
    voice_assignments_json: Optional[Path] = None
    speaker_references_dir: Optional[Path] = None
    lipsync_model: Optional[str] = None
    # Timing plan (stage 9f2). Deliberately NOT mirrored onto `PipelineResult`:
    # that dataclass is serialised verbatim into `pipeline.json`, and adding a
    # field there would change the manifest bytes of every existing run. The
    # path is deterministic (`<out_root>/timing/<producer>.plan.json`) and the
    # plan itself is also written into the merged JSON's `extra.timing_plan`,
    # so nothing is lost by keeping it off the manifest.
    timing_producer: Optional[str] = None
    timing_plan_json: Optional[Path] = None
    timing_summary: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


#: TTS engines that read from a stock voice catalog rather than cloning the
#: source speaker. Since IndicF5 was removed on 2026-10-28 that is every
#: engine there is, so `voice_engine` is no longer a choice — it is the
#: catalog name stage 9d casts from, derived from `--tts`.
_CATALOG_ENGINES = {"bodhan": "indic-speak"}


def _default_voice_engine(voice_engine: Optional[str],
                          tts_model: Optional[str]) -> Optional[str]:
    """The voice catalog stage 9d casts from, or None when TTS is skipped.

    `--voice-engine` was a separate flag until 2026-10-28, and the gap
    between it and `--tts` was a silent wrong output: `--tts bodhan` on its
    own skipped 9d's casting entirely, every speaker got `voice=None`, and
    `bodhan_adapter._post_one` substituted its hardcoded `"voice": voice or
    "Amit"` — one Hindi male voice reading every line of every speaker, with
    the predicted genders computed and discarded. Nothing errored.

    `voice_engine` is still a parameter because every `request.json` on disk
    carries the key and the edit agent replays from it; an explicit value
    still wins. Nothing offers a choice any more.
    """
    if voice_engine:
        return voice_engine
    return _CATALOG_ENGINES.get((tts_model or "").lower())


def _speaker_face_tracks(out_root: Path, video: Path) -> Dict[str, tuple]:
    """`{speaker: (frames, bboxes)}` from this job's stage-5 and stage-6 output.

    Discovered from the job folder rather than threaded in as parameters,
    because `run_post_mt_stages` is also the edit agent's replay entry point
    — there the face stages ran in an EARLIER process and only their files
    survive. Everything here is best-effort: no ASD model, no face detector,
    a post-MT-only bootstrap, or a corrupt npz all mean the gender vote runs
    on its two audio methods, which is a weaker answer and not a failure.
    """
    try:
        npz = (sorted((out_root / "faces").glob(f"{video.stem}.*.npz"))
               or sorted((out_root / "faces").glob("*.npz")))
        js = (sorted((out_root / "asd").glob(f"{video.stem}.*.assignments.json"))
              or sorted((out_root / "asd").glob("*.assignments.json")))
        if not npz or not js:
            log.info("voices: no face tracks + ASD assignments in %s; gender "
                     "votes on audio alone", out_root.name)
            return {}
        from backend.faces.base import FaceTracks
        from backend.faces import speaker_face_tracks
        tracks = FaceTracks.load_npz(npz[0])
        assignments = json.loads(js[0].read_text())
        return speaker_face_tracks(assignments, tracks)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("voices: could not read face tracks (%s: %s); gender "
                    "votes on audio alone", type(exc).__name__, exc)
        return {}


#: Lip-sync models that honour a supplied per-frame coord list (box per
#: frame). MuseTalk takes it through its own `--use_saved_coord` pickle.
COORD_LIST_MODELS = ("musetalk",)

#: Lip-sync models that honour a per-frame targets .npz (106 landmarks per
#: frame). LatentSync's `lipsync_pipeline` reads the file and skips its own
#: InsightFace pass for every frame that carries landmarks.
TARGETS_NPZ_MODELS = ("latentsync15",)


def _asd_coord_list(out_root: Path, video: Path, model: str, *,
                    bbox_shift: int = 0, device: Optional[str] = None):
    """Per-frame boxes for the SPEAKING face, or None.

    Nothing in the lip-sync adapters picks the speaking face. Wav2Lip takes
    the first S3FD detection (`d = d[0]`) and MuseTalk vendors the same
    face_detection lineage plus a whole-frame DWPose estimate that is never
    matched to the box it is paired with — so on a multi-face shot both
    animate an arbitrary face, and a person who is listening gets a moving
    mouth. `backend/lipsync_targets.py` answers both halves on our side.

    Discovered from the job folder rather than threaded in, for the reason
    `_speaker_face_tracks` above gives: `run_post_mt_stages` is also the edit
    agent's replay entry point, and there stages 5 and 6 ran in an EARLIER
    process and only their files survive. Best-effort throughout — a missing
    detector, a missing ASD pass or a corrupt file all mean the adapter
    falls back to its own choice, which is a worse answer and not a failure.

    Returns `(coord_list, stats)` or None.
    """
    if model not in COORD_LIST_MODELS:
        return None
    try:
        npz = (sorted((out_root / "faces").glob(f"{video.stem}.*.npz"))
               or sorted((out_root / "faces").glob("*.npz")))
        # Pick the scores by MODEL (loconet first), not by filename sort: a
        # job can hold several models' `*.scores.npz`, and a bare glob picks
        # whatever sorts first (light_asd before loconet). See find_scores_npz.
        scores_pick = find_scores_npz(out_root / "asd", video.stem)
        if not npz or scores_pick is None:
            log.warning(
                "lipsync: no face tracks + per-frame ASD scores in %s, so %s "
                "will choose a face by itself — on a multi-face shot that is "
                "whoever it finds first, not whoever is speaking. Stages 5 "
                "and 6 must have run, and scores are only written from "
                "2026-10-22; for an older job re-run those stages.",
                out_root.name, model)
            return None

        from backend.faces.base import FaceTracks
        from backend.asd import load_scores_npz
        from backend.asd.lipsync_targets import build_coord_list

        tracks = FaceTracks.load_npz(npz[0])
        asd = load_scores_npz(scores_pick)
        coords, stats = build_coord_list(
            video, tracks, asd, bbox_shift=bbox_shift, device=device or "cpu")
        animated = stats.get("landmarked", 0) + stats.get("fallback_box", 0)
        # "It ran" and "it animated the right face" are different claims, and
        # only the second one matters.
        log.info("lipsync: ASD-gated targets — %d frame(s), %d animated, "
                 "%d passed through unchanged, tracks=%s",
                 len(coords), animated, stats.get("placeholder", 0),
                 ",".join(stats.get("track_ids") or []) or "-")
        if not animated:
            # Every frame would be written unchanged. That is the correct
            # answer to "nobody on screen is speaking", but it is also what a
            # broken ASD pass looks like, and the two are worth separating
            # loudly rather than shipping a render identical to its input.
            log.error("lipsync: no frame has a confirmed speaking face, so "
                      "every frame would pass through unchanged. Check the "
                      "ASD scores before blaming the lip-sync.")
            return None
        return coords, stats
    except Exception as exc:                                  # noqa: BLE001
        log.warning("lipsync: could not build ASD-gated targets (%s: %s); "
                    "%s will choose a face by itself",
                    type(exc).__name__, exc, model)
        return None


def _asd_targets_npz(out_root: Path, video: Path, model: str, *,
                     device: Optional[str] = None):
    """Per-frame 106-landmark targets for LatentSync, gated on ASD.

    The LatentSync equivalent of `_asd_coord_list`: same ASD scores, same
    speaking-track logic, but the artifact carries 106 landmarks per frame
    instead of a box, because that is what LatentSync's warp consumes.
    Returns `(targets_npz_path, stats)` or None.
    """
    if model not in TARGETS_NPZ_MODELS:
        return None
    try:
        npz = (sorted((out_root / "faces").glob(f"{video.stem}.*.npz"))
               or sorted((out_root / "faces").glob("*.npz")))
        scores_pick = find_scores_npz(out_root / "asd", video.stem)
        if not npz or scores_pick is None:
            log.warning(
                "lipsync: no face tracks + per-frame ASD scores in %s, so %s "
                "will choose a face by itself — on a multi-face shot that is "
                "whoever InsightFace finds first, not whoever is speaking.",
                out_root.name, model)
            return None

        from backend.faces.base import FaceTracks
        from backend.asd import load_scores_npz
        from backend.asd.lipsync_targets import build_targets, save_targets

        tracks = FaceTracks.load_npz(npz[0])
        asd = load_scores_npz(scores_pick)
        payload, stats = build_targets(
            video, tracks, asd, device=device or "cpu")
        targeted = stats.get("landmarked", 0)
        if not targeted:
            log.error("lipsync: no frame has a confirmed speaking face with "
                      "landmarks, so every frame would fall back to stock "
                      "detection. Check the ASD scores.")
            return None
        dest = out_root / "lipsync" / f"{video.stem}.targets.npz"
        save_targets(payload, dest)
        log.info("lipsync: ASD-gated LatentSync targets — %d frame(s), "
                 "%d targeted, %d left alone, tracks=%s -> %s",
                 stats.get("frames", 0), targeted,
                 stats.get("no_speaker", 0),
                 ",".join(stats.get("track_ids") or []) or "-",
                 dest.name)
        return dest, stats
    except Exception as exc:                                  # noqa: BLE001
        log.warning("lipsync: could not build ASD-gated targets (%s: %s); "
                    "%s will use its own face detection",
                    type(exc).__name__, exc, model)
        return None


#: Where phase one leaves its gender ballot. Deliberately NOT
#: `speaker_voice_map.json`: stage 9d rewrites that file wholesale every time
#: it runs, so a prediction stored there would be destroyed by the very step
#: that is supposed to reuse it.
GENDER_SUFFIX = ".gender.json"


def _read_gender_stage(out_root: Path, video: Path) -> Dict[str, dict]:
    """Stage 6b's ballot for this job, or `{}`.
    variant writes elsewhere and must predict for itself.
    """
    path = Path(out_root) / "voices" / f"{Path(video).stem}{GENDER_SUFFIX}"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict):
        return {}
    return {spk: rec for spk, rec in doc.items()
            if isinstance(rec, dict) and rec.get("gender")}


def _run_gender_stage(
    out_root: Path,
    video: Path,
    *,
    turns,
    reference_audio: Optional[Path],
    voices_dir: Path,
    asd_model: Optional[str] = None,
    asd_on_screen_threshold: float = 0.3,
    device: Optional[str] = None,
    adapter_kwargs: Optional[dict] = None,
    result=None,
) -> Optional[Path]:
    """Stage 6b: predict every speaker's gender BEFORE the MT gate.

    The staged UI flow stops after MT and asks the operator to cast voices.
    It was asking with an empty gender column, because gender was predicted
    in stage 9d -- on the far side of the gate. This moves the prediction to
    where the question is asked.

    All three voters run, which is why ASD runs here too: without a
    face-to-speaker attribution `predict_gender_face` abstains, every
    disagreement between the two audio voters falls to the tiebreak, and the
    documented failure case in `backend/gender.py` (the lecturer both audio
    methods get wrong and only the face gets right) arrives at the gate
    confidently mislabelled.

    That ASD pass is not an extra one. `_run_asd_stage` is idempotent on
    `<stem>.<model>.scores.npz` + `.assignments.json`, and the cast gate and
    the lip-sync job both reuse what is on disk, so for a job that gets cast
    the pass simply MOVES from after the gate to before it. A job abandoned
    at the gate is the one that pays for nothing.

    Writes `voices/<stem>.gender.json` and returns its path, or None when
    there is nothing to predict from.
    """
    out_root, video = Path(out_root), Path(video)
    if not turns:
        log.warning("gender: no diarization turns; skipping stage 6b")
        return None

    ref_dir = Path(voices_dir) / f"{video.stem}.refs"
    # The same clips 9d would cut, cut early. `backend/gender.py`'s
    # measurements were taken on these -- 3-15 s off the dialogue stem -- not
    # on the UI's shorter listening previews, and writing them now makes 9d's
    # own call a rewrite of identical audio rather than new work.
    speaker_refs = extract_speaker_references(
        reference_audio, turns, out_dir=ref_dir,
    )
    if not speaker_refs:
        log.warning("gender: no reference clips under %s; skipping stage 6b",
                    ref_dir)
        return None

    if asd_model:
        _run_asd_stage(
            out_root, video,
            asd_model=asd_model, turns=turns,
            reference_audio=reference_audio,
            asd_on_screen_threshold=asd_on_screen_threshold,
            device=device, adapter_kwargs=adapter_kwargs,
            result=result,
        )
    else:
        # No model to score with. Let the face voter abstain rather than
        # inventing one: a mismatched model name would have its output swept
        # as stale by `_run_asd_stage` at the cast gate anyway.
        log.info("gender: no ASD model configured; the face vote abstains")

    face_tracks_by_speaker = _speaker_face_tracks(out_root, video)
    votes: Dict[str, dict] = {}
    gender_map = predict_genders(
        speaker_refs, device=device or "cpu", video=video,
        speaker_face_tracks=face_tracks_by_speaker,
        details_out=votes,
    )

    doc = {}
    for spk, gender in gender_map.items():
        vote = votes.get(spk) or {}
        doc[spk] = {
            "gender": gender,
            "gender_vote": vote,
            # Same key names stage 9d writes into `speaker_voice_map`, so
            # nothing downstream has to learn a second vocabulary.
            "gender_basis": ("fallback" if vote.get("decided_by") == "fallback"
                             else vote.get("decided_by")),
            "source": "stage6b",
        }
    out_path = Path(voices_dir) / f"{video.stem}{GENDER_SUFFIX}"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    log.info("gender: wrote %s (%d speaker(s))", out_path, len(doc))
    # No new `PipelineResult` field for the path: the dataclass is serialised
    # verbatim into `pipeline.json`, and the timing-plan comment on that class
    # already settles the question -- a deterministic path does not need a
    # field. `voices/<stem>.gender.json` is that path.
    if result is not None:
        result.speaker_references_dir = ref_dir
    return out_path


def _run_asd_stage(
    out_root: Path,
    video: Path,
    *,
    asd_model: str,
    turns,
    reference_audio: Optional[Path],
    asd_on_screen_threshold: float = 0.3,
    device: Optional[str] = None,
    adapter_kwargs: Optional[dict] = None,
    result=None,
    force: bool = False,
):
    """Score ASD + assign faces to turns, persisting scores + assignments.

    Relocated OUT of the pre-MT pipeline (2026-10-26). ASD now runs at the
    post-MT cast gate -- so stage 9d's face-gender vote has a face-to-speaker
    attribution to read -- and again, as a fallback, at the lip-sync job, so
    the adapter animates the speaking face. It no longer runs mid-pipeline
    where nothing downstream could consume it until much later.

    Idempotent: if this model's `*.scores.npz` and `*.assignments.json` are
    already on disk and `force` is false, nothing is re-scored and the
    assignments on disk are returned -- so the lip-sync job reuses whatever
    the cast gate produced instead of paying a second ASD pass.

    Returns the jsonable assignments list (seg_id, speaker, face_track,
    asd_score, on_screen, ...), or None when it could not run.
    """
    adapter_kwargs = adapter_kwargs or {}
    out_root = Path(out_root)
    video = Path(video)
    asd_dir = out_root / "asd"
    faces = (sorted((out_root / "faces").glob(f"{video.stem}.*.npz"))
             or sorted((out_root / "faces").glob("*.npz")))
    if not faces:
        log.warning("asd: no face tracks under %s; skipping ASD", out_root / "faces")
        return None
    if reference_audio is None or not Path(reference_audio).exists():
        log.warning("asd: no driving audio for %s; skipping ASD", video.name)
        return None

    scores_path = asd_dir / f"{video.stem}.{asd_model}{ASD_SCORES_SUFFIX}"
    assignments_json = asd_dir / f"{video.stem}.{asd_model}.assignments.json"

    # Reuse only THIS model's (loconet's) own output. An old job scored on
    # talknet/light_asd is re-scored on loconet the FIRST time through, and its
    # stale scores are replaced below -- everything standardises on loconet.
    # Once loconet's scores are on disk (a prior lip-sync, or the cast gate),
    # they are reused rather than scored a second time.
    if not force and scores_path.exists() and assignments_json.exists():
        log.info("asd: reusing %s", scores_path.name)
        if result is not None:
            result.asd_model = asd_model
            result.assignments_json = assignments_json
        try:
            return json.loads(assignments_json.read_text())
        except (OSError, ValueError):
            return None

    from backend.faces.base import FaceTracks
    tracks = FaceTracks.load_npz(faces[0])
    log.info("=== ASD (%s) + Hungarian assignment ===", asd_model)
    asd_result = score_asd(
        video, reference_audio, tracks,
        model=asd_model, device=device,
        **_pick(adapter_kwargs, ("repo_path", "checkpoint",
                                 "speaker_encoder_checkpoint",
                                 "crop_size", "target_fps")),
    )
    assignments = assign_faces_to_turns(
        turns or [], tracks, asd_result,
        on_screen_threshold=asd_on_screen_threshold,
    )
    asd_dir.mkdir(parents=True, exist_ok=True)
    jsonable = assignments_to_jsonable(assignments)
    assignments_json.write_text(json.dumps(jsonable, indent=2))
    try:
        save_scores_npz(asd_result, scores_path)
    except Exception as e:                        # noqa: BLE001
        # Never fatal: the per-segment assignments are what gender reads, and
        # the lip-sync job degrades to its own face choice with a warning.
        log.warning("asd: could not persist per-frame scores (%s); the "
                    "lip-sync stage will fall back to its own face choice", e)
    # Replace any OLDER model's ASD output for this stem: loconet is now the
    # scored answer, and nothing downstream should still see the stale
    # talknet/light_asd scores or assignments next to it.
    for stale in (list(asd_dir.glob(f"{video.stem}.*{ASD_SCORES_SUFFIX}"))
                  + list(asd_dir.glob(f"{video.stem}.*.assignments.json"))):
        if stale not in (scores_path, assignments_json):
            try:
                stale.unlink()
                log.info("asd: replaced stale %s", stale.name)
            except OSError:
                pass
    if result is not None:
        result.asd_model = asd_model
        result.assignments_json = assignments_json
        result.num_on_screen = sum(1 for a in assignments if a.on_screen)
    return jsonable


def run_post_mt_stages(
    mt_json: Optional[Path],
    *,
    source_video: Path,
    me_stem: Optional[Path],
    reference_audio_src: Optional[Path],
    out_root: Path,
    target_language: Optional[str],
    device: Optional[str] = None,
    target_script: Optional[str] = None,
    source_language: Optional[str] = None,
    turns: Optional[List] = None,
    manifest: Optional[Path] = None,
    # 6 (relocated). ASD moved out of the pre-MT pipeline (2026-10-26): it runs
    # HERE, at the cast gate, so stage 9d's face-gender vote has a face<->
    # speaker attribution, and it runs lazily -- only when a speaker still
    # needs gender autodetected, or when lip-sync runs in this same call.
    # All-manual gender + a separate lip-sync job means ASD runs at the
    # lip-sync job instead (see `_run_asd_stage`). None = ASD off.
    asd_model: Optional[str] = None,
    asd_on_screen_threshold: float = 0.3,
    asd_adapter_kwargs: Optional[dict] = None,
    # 9b code-mix / script normalisation
    translit_model: Optional[str] = None,
    termbase_path: Optional[Path] = None,
    # 9c duration-aware text adjustment
    duration_adjust: bool = False,
    duration_voice: str = "default",
    duration_safety_margin_s: float = 0.10,
    duration_cache_dir: Optional[Path] = None,
    # 9c2. Ask an LLM to shorten lines 9c predicted will overrun their slot.
    # Off by default: it spends a call per over-long segment and rewrites
    # the user's translation, which is not something to do unasked.
    condense_overlong_text: bool = False,
    condense_model: Optional[str] = None,     # None → condense.DEFAULT_MODEL
    # Rewrite only what the mix cannot speed up far enough to fit. 1.5 by
    # request; deliberately NOT `--speed-cap` (1.8) — see condense.py.
    condense_speed_cap: float = 1.5,
    condense_tolerance: float = 0.0,          # slack above the cap
    # Segments per 9c2 request. None → condense.DEFAULT_BATCH_SIZE (2).
    condense_batch_size: Optional[int] = None,
    # 9d voice assignment per speaker
    voice_engine: Optional[str] = None,
    voice_catalog_json: Optional[Path] = None,
    allow_voice_reuse: bool = False,
    # 9d overrides from `backend.edits` — {speaker: "male"|"female"} forced
    # into the gender prediction, and {speaker: {engine, voice}} forced over
    # whatever the assignment picked. Both default to None, so a run with no
    # edits takes the identical path it always did.
    gender_overrides: Optional[Dict[str, str]] = None,
    speaker_voice_overrides: Optional[Dict[str, dict]] = None,
    # 9e TTS synthesis
    tts_model: Optional[str] = None,
    # 9e delivery style. Reaches the engine via `speaker_voice_map[spk]
    # ["style"]`, which is the ONLY route TTSRequest.style has -- before
    # this it was never populated and the adapter's style branch was dead.
    tts_style: Optional[str] = None,
    # 9b2 per-segment delivery from the SOURCE speaker's emotion. None skips
    # the stage entirely; when it runs it writes `tts_style` onto individual
    # segments, which outrank `tts_style` above (backend/tts.py reads
    # `seg.get("tts_style") or info.get("style")`).
    emotion_model: Optional[str] = None,
    emotion_min_confidence: float = 0.60,
    emotion_min_run: int = 2,
    tts_sample_rate: int = 24000,
    tts_max_stretch: float = 0.20,
    stretch_backend: str = "auto",
    tts_apply_stretch: bool = True,
    # 9e engine-side speed control (method M1). Defaults are copied verbatim
    # from `synthesize_segments`, so leaving them alone is bit-identical to
    # not passing them at all.
    tts_speed_control: str = "off",                  # off|oneshot|closed_loop
    tts_speed_bounds: Tuple[float, float] = (0.85, 1.15),
    tts_speed_tolerance: float = 0.05,
    tts_speed_max_rounds: int = 2,
    # 9f1 post-TTS forced alignment (method M5). Fills `segments[].tts_words[]`
    # so the word-anchored (M4) and DP (M7) timing producers have real TTS word
    # boundaries to cut on instead of falling back to pause_aware.
    tts_force_align: bool = False,
    tts_align_model: str = "mms_fa",
    # 9f1b viseme extraction (method M6). Fills `src_visemes[]`/`tts_visemes[]`
    # so `plan_dp`'s 4th cost term (`_d_phoneme`) has something to measure;
    # without it that term returns (0.0, "no_visemes") and `lambda_phoneme` is
    # inert, which makes an M6 arm a silent clone of M7.
    attach_visemes_stage: bool = False,
    viseme_backend: str = "mms_char",
    viseme_fallback_backend: Optional[str] = "allosaurus",
    viseme_source_audio: Optional[Path] = None,
    # 9f2 timing plan, between TTS and mix
    timing_producer: Optional[str] = None,           # None → no timing plan
    timing_params: Optional[Dict[str, Any]] = None,
    timing_plan_json: Optional[Path] = None,
    # 9g final mix + mux
    mix: bool = False,
    mix_sample_rate: int = 48000,
    mix_tts_gain_db: float = 3.0,
    mix_me_gain_db: float = -3.0,
    mix_strategy: str = "global_tempo",
    mix_tempo_cap: float = 1.5,
    mix_clip_source: str = "auto",
    mux_video: bool = True,
    # The tag that goes into `mix/versions/<stem>.v<NNN>.<producer>.dubbed.*`.
    # `None` means "no snapshot" -- the live `<stem>.dubbed.*` is still
    # written, just not copied aside. Callers that want history (the UI's
    # initial run and every edit replay) pass something readable:
    # `"original"` on the first mix, `"edit_NNN"` on a replay.
    mix_producer_tag: Optional[str] = None,
    # 9h lip sync
    lipsync_model: Optional[str] = None,
    lipsync_num_gpus: Optional[int] = None,
    # Stop after the text stages (9b/9c/9c2/9b2) and return before 9d
    # (voices). The gated UI flow runs these in phase one so the operator
    # sees code-mixed, condensed, emotion-annotated text at the cast table;
    # the cast route then picks up from 9d onward.
    stop_before_voices: bool = False,
    # 9i face restoration over the render. A NAMED parameter, not a lookup
    # into a `config` dict — this function has no `config` in scope, and a
    # `config.get(...)` here raised NameError on every run that reached it.
    # Every UI kwarg must be a named pipeline parameter.
    restore_model: Optional[str] = None,
    # Explicit rather than riding the shared `batch_size` adapter kwarg:
    # stage 4 picks that one too, so a lip-sync tuning value would also be
    # handed to the diarizer. Every UI kwarg must be a NAMED pipeline
    # parameter -- anything reaching `**adapter_kwargs` fails deep inside
    # an adapter instead of at the boundary.
    lipsync_batch_size: Optional[int] = None,
    **adapter_kwargs,
) -> PostMTResult:
    """Run stages 9b..9h against one merged ASR+MT JSON.

    Every stage mutates `mt_json` in place and reads the previous stage's
    `extra.*` back out of it, so the only safe way to run two variants of this
    chain is to hand each one its own copy of the merged JSON and its own
    `out_root` — never to thread per-stage output paths through the chain.

    All outputs land under `out_root/{voices,tts,mix,lipsync}`. The
    pipeline passes the run's job folder, so these land in
    `data/jobs/<job_id>/…` alongside the earlier stages; a variant run gets
    its own `exp/<base>/<variant_tag>/` subtree instead.

    `mt_json` may be None. Stage 9d only needs diarization turns, and the
    MT-less warnings below are what the inline version logged.

    `timing_producer` (with `timing_params`) runs `backend.timing.plan` as
    stage 9f2, between TTS and mix. That slot exists because a timing plan is
    measured off the finished TTS clips and consumed by the mixer, so it can
    be neither a pre-pass nor a second call to this function: the 9g guard
    (`if not out.tts_model`) skips the mix unless TTS ran in the same call.
    The plan is written to `timing_plan_json` (default
    `<out_root>/timing/<producer>.plan.json`), stamped into the merged JSON's
    `extra.timing_plan`, and handed to the mixer. Rendering it also needs
    `mix_strategy="timing_plan"`; any other strategy ignores the plan and is
    warned about. Leaving `timing_producer` None skips all of this and does
    not even create the directory.

    `tts_speed_control` and its `tts_speed_*` companions are forwarded to
    `synthesize_segments` as `speed_control` / `speed_bounds` /
    `speed_tolerance` / `speed_max_rounds` (method M1). The defaults here are
    copied from that function, so not touching them is bit-identical to the
    pre-passthrough behaviour.
    """
    out_root = Path(out_root)
    voices_dir = out_root / "voices"
    tts_dir = out_root / "tts"
    mix_dir = out_root / "mix"
    lipsync_dir = out_root / "lipsync"
    for d in (voices_dir, tts_dir, mix_dir, lipsync_dir):
        d.mkdir(parents=True, exist_ok=True)

    video = Path(source_video)
    mt_json = Path(mt_json) if mt_json is not None else None
    out = PostMTResult(mt_json=mt_json)
    if duration_cache_dir is None:
        duration_cache_dir = out_root / "duration_model"
    # Lip-sync writes its own metadata back into whatever JSON it is handed;
    # the live pipeline hands it the run manifest, a replay has only the plan.
    lipsync_merged_json = manifest if manifest is not None else mt_json
    if turns is None:
        turns = _turns_from_merged(
            json.loads(mt_json.read_text()) if mt_json is not None else {}
        )

    # 9b. Code-mix + script normalisation (plan §6). Overwrites the merged
    #     JSON in place with `final_text` and `retain_spans` per segment.
    if translit_model:
        if not mt_json:
            log.warning("codemix requested but MT was skipped; nothing to re-script")
        else:
            log.info("=== stage 9b: code-mix / script-norm (%s) ===", translit_model)
            cm = codemix_segments(
                mt_json,
                target_language=target_language,
                target_script=target_script or "native",
                translit_model=translit_model,
                termbase_path=termbase_path,
                translit_kwargs=_pick(adapter_kwargs,
                                       ("base_url", "api_key", "model",
                                        "beam_width", "topk")),
            )
            out.translit_model = translit_model
            out.codemix_retained_terms = (
                cm.get("extra", {}).get("codemix", {}).get("retained_terms", 0)
            )

    # 9c. Duration-aware text adjustment (plan §7).
    if duration_adjust:
        if not mt_json:
            log.warning("duration-adjust requested but MT was skipped")
        else:
            log.info("=== stage 9c: duration planning ===")
            d = plan_segment_durations(
                mt_json,
                language=target_language,
                voice=duration_voice,
                cache_dir=duration_cache_dir,
                safety_margin_s=duration_safety_margin_s,
            )
            out.duration_model_language = target_language
            out.duration_model_voice = duration_voice
            out.duration_estimated_violations = (
                d.get("extra", {}).get("duration", {}).get("estimated_violations", 0)
            )

            # 9c2. Shorten what will not fit. Runs here because 9c has just
            # worked out WHICH segments overrun and by how much, and before
            # 9e because the point is to synthesise less speech rather than
            # to speed up speech already synthesised.
            #
            # It is the only lever left on this engine. Measured 2026-10-21
            # on svwrTOm55y4: 50.9s of Hindi for 30.6s of slots, needing
            # 1.66x against a 1.5x tempo cap -- 3 of 5 segments clamped and
            # the dub ~8s long. Bodhan has no speed parameter (their API
            # docs list model/input/voice/instructions and nothing else,
            # checked 2026-10-22), so the audio cannot be made shorter after
            # the fact. The sentence has to be.
            #
            # Marker on ONE line: `server/progress.py` scrapes these, so a
            # wrapped marker loses its closing `===` and silently stops
            # matching.
            if condense_overlong_text:
                log.info("=== stage 9c2: shortening over-long lines ===")
                try:
                    from backend.media.condense import condense_overlong
                    c = condense_overlong(
                        mt_json,
                        language=target_language,
                        source_language=source_language,
                        voice=duration_voice,
                        cache_dir=duration_cache_dir,
                        model=condense_model,
                        speed_cap=condense_speed_cap,
                        tolerance=condense_tolerance,
                        batch_size=condense_batch_size,
                    )
                    out.extra.setdefault("condense", {}).update(c)
                except Exception as e:                   # noqa: BLE001
                    # A shortening pass is an optimisation. Losing it costs
                    # the fit, not the dub.
                    log.warning("condense stage failed (%s); keeping the "
                                "translation as it is", e)
                    out.extra.setdefault("condense", {})["error"] = str(e)

    # 9b2. Per-segment delivery style from the source speaker's emotion.
    # Runs after the text is final and before any voice decision, because it
    # only annotates segments -- stage 9e is what acts on them.
    if emotion_model:
        if not mt_json:
            log.warning("emotion requested but MT was skipped; nothing to annotate")
        elif not reference_audio_src:
            log.warning("emotion requested but no source audio in scope; skipping")
        else:
            log.info("=== stage 9b2: source emotion (%s) ===", emotion_model)
            try:
                from backend.emotion.emotion import annotate_segments_with_emotion
                emo = annotate_segments_with_emotion(
                    mt_json,
                    source_audio=reference_audio_src,
                    model=emotion_model,
                    clips_dir=Path(out_root) / "emotion_clips",
                    device=device,
                    min_confidence=emotion_min_confidence,
                    min_run=emotion_min_run,
                    global_style=tts_style,
                )
                out.extra.setdefault("emotion", {}).update(emo)
            except Exception as e:                   # noqa: BLE001
                # A classifier is an enhancement; losing it must not cost the
                # dub. The run continues on the global style.
                log.warning("emotion stage failed (%s); continuing with the "
                            "global style only", e)
                out.extra.setdefault("emotion", {})["error"] = str(e)

    # Gate: the UI's phase one stops here. Everything above (9b, 9c/9c2,
    # 9b2) has annotated the text; everything below (9d onward) needs the
    # operator's cast choices.
    if stop_before_voices:
        return out

    # 9d. Voice assignment per speaker (plan §8).
    voice_engine = _default_voice_engine(voice_engine, tts_model)
    speaker_voice_map: Dict[str, dict] = {}
    speaker_refs: Dict[str, Path] = {}
    if voice_engine or tts_model:
        log.info("=== stage 9d: speaker voice assignment ===")
        # Build a speakers[] list from the diarization turns.
        from collections import defaultdict
        totals: Dict[str, float] = defaultdict(float)
        for t in turns:
            totals[t.speaker] += (t.end - t.start)
        speakers = [{"id": spk, "total_speech_s": tot}
                    for spk, tot in totals.items()]

        # Per-speaker reference clips (extracted from the dialogue stem
        # when available, else the raw 16 kHz mix).
        ref_dir = voices_dir / f"{video.stem}.refs"
        src_for_refs = reference_audio_src
        speaker_refs = extract_speaker_references(
            src_for_refs, turns, out_dir=ref_dir,
        )
        out.speaker_references_dir = ref_dir

        # ASD at the cast gate (relocated stage 6). Run it only when a face
        # vote would actually help -- i.e. a speaker here still needs gender
        # autodetected (not every one is a manual override) -- or when
        # lip-sync runs in THIS call and needs the scores. All-manual gender
        # with a separate lip-sync job skips it here; the lip-sync job runs
        # ASD itself. Idempotent, so a second call reuses the scores.
        _all_speakers = {s["id"] for s in speakers}
        # Stage 6b already voted for these, before the gate. Reuse it: a
        # second ballot would reload 1.2 GB of wav2vec2 weights to reach the
        # same verdict, and would re-run ASD to feed a face vote whose scores
        # are already on disk. Scoped to THIS `out_root`, so an experiment
        # variant (which writes under `exp_root/<stem>/<tag>`) finds no cache
        # and predicts afresh -- which is what variant isolation requires.
        _cached_genders = _read_gender_stage(out_root, video)
        _need = (_all_speakers - set(gender_overrides or {})
                 - set(_cached_genders))
        if _cached_genders:
            log.info("voices: reusing stage 6b gender for %d speaker(s); "
                     "%d still to predict",
                     len(_all_speakers & set(_cached_genders)), len(_need))
        # ASD at the cast gate only buys the face vote, so it is needed only
        # for the speakers still to predict. With 6b having covered everyone
        # this skips entirely -- the scores it wrote are already on disk for
        # the lip-sync job to reuse.
        _need_gender_asd = bool(_need)
        if asd_model and _need_gender_asd:
            _run_asd_stage(
                out_root, video,
                asd_model=asd_model, turns=turns,
                reference_audio=reference_audio_src,
                asd_on_screen_threshold=asd_on_screen_threshold,
                device=device, adapter_kwargs=asd_adapter_kwargs,
                result=out,
            )

        # Gender prediction. THREE methods vote -- wav2vec2 over the
        # reference clip, F0 median pitch over the same clip, and
        # insightface's genderage over the face track ASD attributed to this
        # speaker -- and the majority wins. `backend/gender.py` has the
        # measurements; the short version is that one lecturer in the test
        # set defeats both audio methods at once and only the face gets him
        # right, so no single method is safe here. The face vote is optional
        # by construction: without face tracks + ASD it abstains.
        gender_votes: Dict[str, dict] = {
            spk: dict(rec.get("gender_vote") or {})
            for spk, rec in _cached_genders.items()
        }
        gender_map = {spk: rec["gender"]
                      for spk, rec in _cached_genders.items()}
        if _need:
            face_tracks_by_speaker = _speaker_face_tracks(out_root, video)
            gender_map.update(predict_genders(
                {k: v for k, v in speaker_refs.items() if k in _need},
                device=device or "cpu", video=video,
                speaker_face_tracks=face_tracks_by_speaker,
                details_out=gender_votes,
            ))
        if gender_overrides:
            for sid, g in gender_overrides.items():
                if gender_map.get(sid) != g:
                    log.info("voices: %s gender overridden %s -> %s (edit)",
                             sid, gender_map.get(sid), g)
                gender_map[sid] = g
        for spk in speakers:
            spk["predicted_gender"] = gender_map.get(spk["id"])
            # The whole ballot, so a finished run says WHY it cast that way
            # without re-running three models. An edited gender is stamped
            # over the verdict but leaves the votes visible underneath.
            v = gender_votes.get(spk["id"])
            if v is not None:
                if v["gender"] != spk["predicted_gender"]:
                    v = {**v, "gender": spk["predicted_gender"],
                         "decided_by": "override"}
                spk["gender_vote"] = v
                # `assign_voices` reads this: no evidence means cast a
                # target-language voice of the fallback gender rather than
                # ranking a placeholder by timbre.
                if v["decided_by"] == "fallback":
                    spk["gender_basis"] = "fallback"

        if voice_engine:
            voice_assignments = assign_voices(
                speakers,
                engine=voice_engine,
                target_language=target_language or source_language or "en",
                catalog_json=voice_catalog_json,
                reference_audio=speaker_refs,
                device=device,
                allow_voice_reuse=allow_voice_reuse,
            )
            va_json = voices_dir / f"{video.stem}.{voice_engine}.voices.json"
            va_json.write_text(json.dumps(
                voice_assignments_to_jsonable(voice_assignments), indent=2,
            ))
            out.voice_engine = voice_engine
            out.voice_assignments_json = va_json
            for a in voice_assignments:
                speaker_voice_map[a.speaker] = {
                    "engine": a.engine,
                    "voice": a.voice,
                    "reference_audio": str(a.reference_audio) if a.reference_audio else None,
                    "style": tts_style,
                }
        else:
            # Reachable only when TTS is skipped but 9d was entered anyway,
            # or when a legacy `request.json` names an engine with no
            # catalog. `voice=None` lets the adapter fall back to its own
            # default rather than failing the run; the cloning path this
            # branch used to serve (IndicF5, VC) is gone.
            log.warning("voices: no voice catalog for tts=%r; every speaker "
                        "will read in the adapter's default voice",
                        tts_model)
            for spk in sorted(speaker_refs):
                speaker_voice_map[spk] = {
                    "engine": tts_model,
                    "voice": None,
                    "reference_audio": str(speaker_refs[spk]),
                    "style": tts_style,
                }

        # An edited voice wins over whatever the assignment chose, and is
        # recorded in `extra.speaker_voice_map` like any other assignment so
        # the finished run carries evidence of which voice actually spoke.
        for sid, info in (speaker_voice_overrides or {}).items():
            before = (speaker_voice_map.get(sid) or {}).get("voice")
            merged = dict(speaker_voice_map.get(sid) or {})
            merged.update({k: v for k, v in info.items() if v is not None})
            merged.setdefault("reference_audio",
                              str(speaker_refs[sid]) if sid in speaker_refs else None)
            speaker_voice_map[sid] = merged
            log.info("voices: %s voice overridden %s -> %s (edit)",
                     sid, before, merged.get("voice"))

        # Persist the map whatever branch produced it. The `voices.json`
        # above is written ONLY when `voice_engine` is set; the catalog-less
        # path builds `speaker_voice_map` in memory and leaves no record at
        # all, so nothing downstream could say which voice spoke for which
        # speaker. A separate filename, so `voice_assignments_json` keeps
        # its existing meaning and no manifest key changes.
        # Gender is computed here from the reference clips and then used only
        # to pick a voice — it was never written down, so the UI could not
        # show it or offer same-gender voices without re-running the
        # gender classifier on every request. Record it alongside the voice.
        for _sid, _info in speaker_voice_map.items():
            _g = gender_map.get(_sid)
            if _g and isinstance(_info, dict):
                _info.setdefault("gender", _g)
                _v = gender_votes.get(_sid)
                if _v is not None:
                    _info.setdefault("gender_vote", _v)

        if speaker_voice_map:
            (voices_dir / f"{video.stem}.speaker_voice_map.json").write_text(
                json.dumps(speaker_voice_map, indent=2, ensure_ascii=False))

    # 9e. TTS synthesis (plan §8) — needs MT (and ideally codemix+duration).
    if tts_model:
        if not mt_json:
            log.warning("TTS requested but MT was skipped; nothing to speak")
        else:
            log.info("=== stage 9e: TTS synthesis (%s) ===", tts_model)
            tts_out = tts_dir / f"{video.stem}.{tts_model}"
            synthesize_segments(
                mt_json,
                tts_model=tts_model,
                language=target_language,
                speaker_voice_map=speaker_voice_map,
                out_dir=tts_out,
                sample_rate=tts_sample_rate,
                max_stretch=tts_max_stretch,
                stretch_backend=stretch_backend,
                apply_stretch=tts_apply_stretch,
                speed_control=tts_speed_control,
                speed_bounds=tts_speed_bounds,
                speed_tolerance=tts_speed_tolerance,
                speed_max_rounds=tts_speed_max_rounds,
                tts_kwargs=_pick(adapter_kwargs,
                                   ("base_url", "api_key", "model",
                                    "repo_id", "dtype", "timeout_s",
                                    "max_retries", "rate", "volume",
                                    "pitch")),
            )
            out.tts_model = tts_model
            out.tts_out_dir = tts_out
            tts_meta = json.loads(mt_json.read_text()).get("extra", {}).get("tts", {})
            out.tts_budget_violations = int(tts_meta.get("budget_violations", 0))
            out.tts_errors = int(tts_meta.get("errors", 0))

    # 9f1. Post-TTS forced alignment (method M5). Fills `segments[].tts_words[]`
    #      -- CLIP-RELATIVE word boundaries measured on the RAW TTS clip --
    #      which `plan_word_anchored` (M4) and `plan_dp` (M7) read in 9f2 to cut
    #      that clip at clause boundaries. Without it both silently degrade to
    #      their pause_aware fallback and the results table shows a timing
    #      method that never actually ran, which is why the stage reports its
    #      own health below instead of leaving the degradation invisible.
    if tts_force_align:
        skip = None
        if not mt_json:
            skip = "no_mt"
        elif not out.tts_model and not _merged_has_any_clip(mt_json):
            skip = "no_tts_clips"
        elif not target_language:
            skip = "no_target_language"
        if skip:
            # Recorded, not just logged: a skipped 9f1 means M4/M7 quietly ran
            # on their fallback, and that has to reach the results table.
            log.warning("tts_force_align requested but skipped (%s)", skip)
            out.extra["tts_align"] = {
                "model": tts_align_model, "skipped": skip,
                "segments_with_tts_words": 0, "total": 0, "words_aligned": 0,
                "failures": 0, "failure_reasons": {}, "token_coverage": None,
            }
        else:
            log.info("=== stage 9f1: post-TTS forced alignment (%s) ===",
                     tts_align_model)
            align_work = out_root / "tts_align"
            align_work.mkdir(parents=True, exist_ok=True)
            # `audio_field` stays "tts_raw_audio" on purpose: `tts_audio` may
            # already have been time-stretched by 9e, and word times measured
            # on a stretched clip cannot be used to cut the RAW clip -- which
            # is exactly what `Piece.src_span` does.
            #
            # `out_json=mt_json` is what persists `tts_words[]`:
            # `align_tts_segments` mutates and returns a dict and writes
            # nothing unless it is given a destination, while 9f2 re-reads the
            # merged JSON off disk. Without this the stage would run, cost its
            # GPU minutes, and change nothing downstream.
            #
            # One aligner instance is reused across every segment inside
            # `align_tts_segments` (built lazily on the first segment that has
            # a clip), so this is one construction per call, not per segment.
            # Per-segment failures degrade to `tts_words = []` plus a
            # `qc["tts_align"]` reason in there and the run continues; that is
            # deliberately not wrapped here, because swallowing a whole-stage
            # failure (e.g. the alignment model will not load at all) would put
            # M4/M7 back on the silent fallback this stage exists to remove.
            aligned = align_tts_segments(
                mt_json,
                model=tts_align_model,
                language=target_language,
                device=device,
                audio_field="tts_raw_audio",
                work_dir=align_work / "work",
                out_json=mt_json,
            )
            health = _tts_align_health(aligned, tts_align_model)
            out.extra["tts_align"] = health
            log.info(
                "tts_align: %d/%d segments carry tts_words[], %d words, "
                "%d failures (%s), token coverage %s",
                health["segments_with_tts_words"], health["total"],
                health["words_aligned"], health["failures"],
                health["failure_reasons"] or "none",
                "not measured (nothing attempted)"
                if health["token_coverage"] is None
                else f"{health['token_coverage']:.3f}",
            )

    # 9f1b. Viseme extraction (method M6). Must land AFTER 9f1 -- it reads the
    #       same raw TTS clips -- and BEFORE 9f2, which is the only consumer:
    #       `plan_dp`'s `_d_phoneme` samples `src_visemes[]`/`tts_visemes[]`.
    #       Nothing else in the pipeline reads them, so running this without a
    #       `lambda_phoneme > 0` DP is pure cost for no effect; that is why it
    #       is opt-in rather than on by default.
    if attach_visemes_stage:
        skip = None
        if not mt_json:
            skip = "no_mt"
        elif not out.tts_model and not _merged_has_any_clip(mt_json):
            skip = "no_tts_clips"
        elif not target_language:
            skip = "no_target_language"
        if skip:
            # Same reasoning as 9f1: a silently skipped viseme stage leaves
            # `lambda_phoneme` multiplying a hard 0.0, so an M6 arm would score
            # exactly like M7 while looking enabled. Record it as a row.
            log.warning("attach_visemes requested but skipped (%s)", skip)
            out.extra["visemes"] = {"skipped": skip, "backend": viseme_backend}
        else:
            log.info("=== stage 9f1b: viseme extraction (%s) ===", viseme_backend)
            vis_work = out_root / "visemes"
            vis_work.mkdir(parents=True, exist_ok=True)
            # `out_json=mt_json` for the same load-bearing reason as 9f1:
            # `attach_visemes` mutates and returns a dict and writes nothing
            # unless given a destination, while 9f2 re-reads the merged JSON
            # off disk. `audio` falls back to the merged JSON's own
            # `audio_path` (the source dialogue stem) when not overridden.
            # Per-segment failures degrade to `[]` plus a `qc["visemes_*"]`
            # reason inside `attach_visemes`; not wrapped here, because
            # swallowing a whole-stage failure would put M6 back to being an
            # undetectable M7 clone.
            vis = attach_visemes(
                mt_json,
                language=target_language,
                audio=viseme_source_audio,
                backend=viseme_backend,
                fallback_backend=viseme_fallback_backend,
                device=device,
                audio_field="tts_raw_audio",
                work_dir=vis_work / "work",
                out_json=mt_json,
            )
            health = (vis.get("extra") or {}).get("visemes") or {}
            out.extra["visemes"] = health
            log.info("visemes: %s", health)

    # 9f2. Timing plan (methods M3/M4/M7). Has to land AFTER the TTS clips
    #      exist — every producer measures the raw clip — and BEFORE the mix
    #      reads them, which is why it cannot be a separate `run_post_mt_stages`
    #      call: a second call would have to re-synthesise to get past the
    #      `if not out.tts_model` guard on 9g.
    if timing_producer:
        if not mt_json:
            log.warning("timing plan requested but MT was skipped; nothing to plan")
        elif not out.tts_model and not _merged_has_clips(mt_json):
            log.warning(
                "timing plan requested but there are no TTS clips to place "
                "(TTS did not run here, and %s carries none either)",
                mt_json.name)
        else:
            log.info("=== stage 9f2: timing plan (%s) ===", timing_producer)
            from backend.media import timing as _timing
            tp = _timing.plan(timing_producer, mt_json, **(timing_params or {}))
            if timing_plan_json is not None:
                plan_dest = Path(timing_plan_json)
            else:
                timing_dir = out_root / "timing"
                timing_dir.mkdir(parents=True, exist_ok=True)
                plan_dest = timing_dir / f"{timing_producer}.plan.json"
            tp.write(plan_dest)
            # Also lands in `extra.timing_plan` of the merged JSON, which is
            # where `mix_dubbed_audio` looks when it is not handed a path.
            _timing.apply_plan_to_merged(tp, mt_json)
            out.timing_producer = timing_producer
            out.timing_plan_json = plan_dest
            out.timing_summary = tp.summary
            log.info("timing plan [%s] -> %s", timing_producer, plan_dest)
            if mix and mix_strategy != "timing_plan":
                log.warning(
                    "timing plan %r written to %s, but mix_strategy=%r will "
                    "ignore it; pass mix_strategy='timing_plan' to render it",
                    timing_producer, plan_dest, mix_strategy)

    # 9g. Final audio mix + optional video mux (plan §8).
    if mix:
        if not mt_json:
            log.warning("mix requested but there is no merged JSON to drive it")
        elif not out.tts_model and not _merged_has_any_clip(mt_json):
            # TTS did not run in THIS invocation (e.g. an edit replay that is
            # dirty from `timing`, not `tts`), but the clips from the previous
            # TTS pass are still on disk and recorded in the merged JSON, so
            # the mix is still well-defined. Only skip when there really is
            # nothing to place.
            log.warning(
                "mix requested but TTS was skipped and %s carries no clips",
                mt_json.name)
        else:
            log.info("=== stage 9g: audio mix (ffmpeg) ===")
            mr = mix_dubbed_audio(
                mt_json,
                me_stem=me_stem,
                source_video=video,
                out_dir=mix_dir,
                sample_rate=mix_sample_rate,
                tts_gain_db=mix_tts_gain_db,
                me_gain_db=mix_me_gain_db,
                mux_video=mux_video,
                strategy=mix_strategy,
                tempo_cap=mix_tempo_cap,
                clip_source=mix_clip_source,
                timing_plan=out.timing_plan_json,
            )
            out.mix_wav = mr.audio_wav
            out.mix_mp4 = mr.video_mp4
            # Snapshot the newly written dub into `mix/versions/` so a later
            # edit's replay cannot erase the one that just rendered. Both the
            # wav and the mp4 are copied; neither is a dependency, so a
            # failure here is logged and the pipeline keeps going.
            if mix_producer_tag:
                from backend.media import mix_versions as _mix_versions
                _mix_versions.snapshot(
                    mr.audio_wav, mr.video_mp4, mix_producer_tag)

    # 9h. Lip sync (plan §8). Needs the dubbed audio from the mix stage.
    if lipsync_model:
        if not out.mix_wav:
            log.warning("lipsync requested but mix was skipped; nothing to drive")
        else:
            log.info("=== stage 9h: lip sync (%s) ===", lipsync_model)
            try:
                # ASD runs HERE, just before lip-sync (relocated stage 6):
                # score who is speaking if the cast gate did not already.
                # Idempotent, so a full run that scored at the gate reuses it
                # and a lip-sync-only replay scores it now. Inside the try so
                # a failure is a lip-sync error, not a silent wrong-face dub.
                if asd_model:
                    # The marker lives at THIS call site, not inside
                    # `_run_asd_stage`, and that is deliberate. ASD also runs
                    # in stage 6b (for the gender face vote) before the MT
                    # gate; a marker inside the helper would fire there too
                    # and light the Lip-sync pill on a job that has not been
                    # anywhere near a render. ONE line -- `server/progress.py`
                    # scrapes it and the test greps the literal.
                    log.info("=== stage 6: active speaker detection (%s) ===", asd_model)
                    _run_asd_stage(
                        out_root, video, asd_model=asd_model, turns=turns,
                        reference_audio=reference_audio_src,
                        asd_on_screen_threshold=asd_on_screen_threshold,
                        device=device, adapter_kwargs=asd_adapter_kwargs,
                        result=out)
                # WHICH face to animate. Without this the adapter picks one
                # itself — the first detection — so on a two-person shot the
                # listener's mouth moves and the speaker's does not, at exit
                # code 0 with every artifact looking correct.
                #
                # INSIDE the try, so a failure here is recorded as a
                # lip-sync error instead of spending 8-21x the clip length
                # producing exactly the artifact this exists to prevent.
                _targets = _asd_coord_list(
                    out_root, video, lipsync_model,
                    bbox_shift=int(adapter_kwargs.get("bbox_shift") or 0),
                    device=device)
                _coords = _targets[0] if _targets else None
                if _targets:
                    out.extra["lipsync_targets"] = _targets[1]
                _ls_targets = _asd_targets_npz(
                    out_root, video, lipsync_model, device=device)
                _targets_npz = _ls_targets[0] if _ls_targets else None
                if _ls_targets:
                    out.extra["lipsync_targets"] = _ls_targets[1]
                ls = run_lipsync(
                    lipsync_merged_json,
                    model=lipsync_model,
                    source_video=video,
                    driving_audio=out.mix_wav,
                    out_dir=lipsync_dir,
                    pre_muxed_video=out.mix_mp4,
                    num_gpus=lipsync_num_gpus,
                    window=None,
                    coord_list=_coords,
                    targets_npz=_targets_npz,
                    lipsync_kwargs={
                        **_pick(adapter_kwargs,
                                ("repo_path", "checkpoint", "python_bin",
                                 "unet_config_path", "unet_ckpt_path",
                                 "version", "batch_size", "fps", "bbox_shift",
                                 "pads", "resize_factor", "wav2lip_batch_size",
                                 "face_det_batch_size", "inference_steps",
                                 "guidance_scale", "seed",
                                 "model_arch")),
                        # The named parameter wins over a generic
                        # `batch_size` that happened to be in adapter_kwargs.
                        **({"batch_size": lipsync_batch_size}
                           if lipsync_batch_size else {}),
                    },
                )
                out.lipsync_model = ls.model
                out.lipsync_error = ls.error
                # Only record a render that exists. `ls.out_video` is the
                # path the adapter was ASKED to write, and a failed run still
                # carries it -- which put a nonexistent mp4 in the manifest
                # under `result.lipsync_mp4`, where `edit_agent` and anything
                # else reading the manifest would believe in it.
                if ls.out_video and Path(ls.out_video).is_file():
                    out.lipsync_mp4 = ls.out_video
                elif not ls.error:
                    out.lipsync_error = (
                        f"{ls.model} produced no file at {ls.out_video}")
            except Exception as e:
                log.warning("lipsync failed: %s", e)
                out.lipsync_error = str(e)

    # --- stage 9i: face restoration over the render -----------------------
    #
    # Off unless asked for. It is orthogonal to lip-sync -- it takes a video
    # and returns a video and does not care which adapter drew the mouths --
    # and it is expensive in its own right: measured 2026-10-22 at 1156 s
    # against the MuseTalk render's 447 s, i.e. 2.6x the cost of the thing it
    # cleans. So `--restore gfpgan`, never implied by `--lipsync`.
    #
    # NEVER fails the run. Stage 9h's render is on disk and is a usable dub;
    # losing it because a post-pass failed would be the worse trade, which is
    # the same call `wav2lip_hd`'s stage 2 made.
    if restore_model and out.lipsync_mp4:
        log.info("=== stage 9i: face restoration (%s) ===", restore_model)
        try:
            from backend.restore.restore import run_restore
            _r = run_restore(Path(out_root), model=str(restore_model),
                             render=Path(out.lipsync_mp4),
                             num_gpus=lipsync_num_gpus)
            out.extra["restore"] = _r
            if _r.get("ok") and _r.get("video"):
                out.lipsync_mp4 = _r["video"]
            else:
                log.warning("restore: stage 9i did not produce a video (%s); "
                            "keeping the plain render", _r.get("error"))
        except Exception as e:                            # noqa: BLE001
            log.warning("restore: stage 9i failed (%s); keeping the plain "
                        "render", e)
            out.extra["restore"] = {"ok": False, "error": str(e)}

    out.extra["speaker_voice_map"] = speaker_voice_map
    if speaker_voice_map and mt_json is not None and Path(mt_json).is_file():
        # Mirror it into the merged JSON as well: a post-MT replay reads that
        # file and nothing else, so without this an edited re-run would lose
        # the record of who spoke in which voice.
        try:
            _data = json.loads(Path(mt_json).read_text())
            _data.setdefault("extra", {})["speaker_voice_map"] = speaker_voice_map
            Path(mt_json).write_text(
                json.dumps(_data, indent=2, ensure_ascii=False))
        except (OSError, ValueError) as e:
            log.warning("could not record speaker_voice_map in %s: %s",
                        mt_json, e)
    return out


# Fields a replay cannot re-derive from the merged JSON, carried over from the
# upstream run's manifest. Everything stages 9b..9h own is deliberately absent:
# `run_pipeline` overwrites those from the `PostMTResult` anyway, so a stale
# `mix_wav` or `lipsync_error` can never survive into the rewritten manifest.
_REPLAY_CARRY_OVER = (
    "face_tracks_npz", "num_face_tracks", "face_detector", "face_tracker",
    "asd_model", "assignments_json", "num_on_screen",
    "asr_model", "asr_json",
    "align_model", "align_json", "num_aligned_words",
    "mt_txt", "subtitles", "asr_eval_json", "mt_eval_json",
)
_REPLAY_CARRY_OVER_PATHS = frozenset({
    "face_tracks_npz", "assignments_json", "asr_json", "align_json",
    "mt_txt", "asr_eval_json", "mt_eval_json",
})


def _carry_over_upstream_manifest(result: PipelineResult, manifest: Path) -> int:
    """Copy stages 1..9a's bookkeeping out of `manifest` onto `result`.

    `run_pipeline` ends by rewriting `result.manifest`. On a replay that file
    is the upstream run's own manifest, and the rebuilt `PipelineResult` knows
    nothing about faces, ASD, ASR, alignment, subtitles or the eval reports —
    so without this the replay silently erases all of it. Only fields the
    replay left at their default are filled in.
    """
    if not manifest.exists():
        return 0
    try:
        prior = json.loads(manifest.read_text())
    except (OSError, ValueError) as e:
        log.warning("post-MT replay: unreadable upstream manifest %s (%s); "
                    "its stage 1-9a metadata will not be preserved",
                    manifest, e)
        return 0
    if not isinstance(prior, dict):
        return 0

    restored = 0
    for name in _REPLAY_CARRY_OVER:
        value = prior.get(name)
        if value in (None, 0, "", {}, []):
            continue
        if getattr(result, name, None) not in (None, 0, "", {}, []):
            continue
        if name in _REPLAY_CARRY_OVER_PATHS and isinstance(value, str):
            value = Path(value)
        setattr(result, name, value)
        restored += 1

    # A merged JSON with no speaker labels yields zero turns; keep the real
    # diarization counts rather than writing 0/0 over them.
    if not result.num_turns and prior.get("num_turns"):
        result.num_turns = prior["num_turns"]
        result.num_speakers = prior.get("num_speakers", 0)
        restored += 1
    return restored


def _upstream_root(merged_json: Path, data_dir: Path) -> Path:
    """The job folder whose stages 1..9a produced `merged_json`.

    A replay reads the upstream pass's video/audio/stems instead of spending
    hours regenerating them. Those now live inside that run's own job folder,
    so find it by walking up from the merged JSON to the nearest ancestor
    that actually holds the stage dirs — which works whether the JSON sits in
    `<job>/mt/` or in a variant's `<job>/exp/<base>/<tag>/`. Falls back to
    `data_dir` so a pre-job-layout tree still resolves.
    """
    for anc in list(Path(merged_json).resolve().parents)[:6]:
        if (anc / "audio").is_dir() and (anc / "videos").is_dir():
            return anc
    return data_dir


def _bootstrap_post_mt_only(
    input_: str,
    merged_json: Path,
    *,
    data_dir: Path,
    diarization_model: str,
    stem_model: str,
    skip_stems: bool,
    asr_language: Optional[str],
    mt_model: Optional[str],
    mt_target_language: Optional[str],
    mt_target_script: Optional[str],
    videos_dir: Optional[Path] = None,
) -> Tuple[PipelineResult, Path, Path, Optional[Path], List["_Turn"]]:
    """Rebuild just enough state to replay stages 9b..9h for `merged_json`.

    Stages 1..9a already produced that file and left their other artifacts at
    deterministic paths inside the job folder they ran in, so we point at the
    ones that exist rather than re-running an upstream pass that costs hours
    and gigabytes.
    Returns (result, video, reference_audio_src, me_stem, turns).
    """
    merged_json = Path(merged_json)
    if not merged_json.exists():
        raise FileNotFoundError(
            f"post_mt_only_from: no such merged JSON: {merged_json}"
        )
    data = json.loads(merged_json.read_text())

    # Read the upstream pass's artifacts from the job folder that produced
    # them; anything this replay downloads goes to the NEW job's videos dir.
    upstream = _upstream_root(merged_json, data_dir)
    if upstream != data_dir:
        log.info("post-MT replay: upstream artifacts from %s", upstream)

    video = Path(input_).expanduser()
    if not video.exists():
        video = resolve_input(
            input_, download_dir=(videos_dir or upstream / "videos"))

    audio_16k = upstream / "audio" / f"{video.stem}.16k.mono.wav"
    dia_input = upstream / "audio" / f"{video.stem}.dialogue.16k.mono.wav"
    if not dia_input.exists():
        dia_input = audio_16k

    # Demucs' own layout, mirrored from stem_separation.separate_stems.
    stem_track = upstream / "stems" / stem_model / f"{video.stem}.44k.stereo"
    dialogue_stem = stem_track / "vocals.wav"
    me_stem = stem_track / "no_vocals.wav"
    dialogue_stem = dialogue_stem if dialogue_stem.exists() else None
    me_stem = me_stem if me_stem.exists() else None

    turns = _turns_from_merged(data)
    extra = data.get("extra", {})
    dia_dir = data_dir / "diarization"
    rttm = dia_dir / f"{video.stem}.{diarization_model}.rttm"
    manifest = dia_dir / f"{video.stem}.{diarization_model}.pipeline.json"
    if not rttm.exists():
        # Almost always a diarizer mismatch: the upstream pass used a
        # different one and `diarization_model` still has its default.
        log.warning("post-MT replay: no %s RTTM at %s — check "
                    "--diarization-model matches the upstream run",
                    diarization_model, rttm)
    result = PipelineResult(
        input=input_,
        video=video,
        audio_16k=audio_16k,
        dialogue_stem=dialogue_stem,
        me_stem=me_stem,
        rttm=rttm,
        num_turns=len(turns),
        num_speakers=len({t.speaker for t in turns}),
        diarization_model=diarization_model,
        manifest=manifest,
        stems_skipped=skip_stems,
    )
    result.asr_language = asr_language or data.get("language")
    result.mt_model = mt_model or extra.get("mt_model")
    result.mt_target_language = mt_target_language or extra.get("mt_target_language")
    result.mt_target_script = mt_target_script or extra.get("mt_target_script")
    result.mt_json = merged_json
    # run_pipeline rewrites `manifest` at the end; keep what stages 1..9a put
    # there instead of blanking it.
    carried = _carry_over_upstream_manifest(result, manifest)
    # Report the turns the post-MT chain will actually use, not whatever the
    # upstream manifest's counts were carried back in as.
    log.info("post-MT replay: %s — %d turns, %d speakers, plan=%s%s",
             video.name, len(turns), len({t.speaker for t in turns}),
             merged_json,
             f" (carried {carried} field(s) over from {manifest.name})"
             if carried else "")
    return result, video, dia_input, me_stem, turns


def _adopt_into_job(src: Path, videos_dir: Path) -> Path:
    """Make sure the run's input lives inside the job folder.

    `resolve_input` downloads a URL straight into `videos_dir`, but hands a
    local path back untouched — which would leave the job folder without the
    one file every stage reads from. Hardlink it in (same filesystem, so no
    extra bytes and no risk of a later job mutating a shared copy), falling
    back to a real copy across filesystems. Sidecar subtitles come along too.
    """
    src = Path(src).expanduser().resolve()
    videos_dir = Path(videos_dir)
    try:
        if src.parent.resolve() == videos_dir.resolve():
            return src                      # already ours (URL download)
    except OSError:
        pass

    dst = videos_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst

    for name in (src.name, *(q.name for q in sorted(
            src.parent.glob(f"{src.stem}.*")) if q.suffix.lower() in
            (".vtt", ".srt", ".txt", ".json"))):
        s_path, d_path = src.parent / name, videos_dir / name
        if not s_path.is_file() or d_path.exists():
            continue
        try:
            _os_link(s_path, d_path)
        except OSError:
            shutil.copy2(s_path, d_path)    # cross-device, or link limit
    log.info("input adopted into job: %s", dst)
    return dst if dst.exists() else src


def _run_pipeline(
    input_: str,
    *,
    diarization_model: str = "pyannote",
    data_dir: Path = DEFAULT_DATA_DIR,
    device: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    skip_stems: bool = False,
    # Logical resegmentation (between diarization and ASR).
    resegment_turns: bool = False,
    resegment_split_min_s: float = 8.0,
    resegment_split_pause_s: float = 0.15,
    resegment_merge_max_gap_s: float = 0.50,
    resegment_merge_max_result_s: float = 15.0,
    resegment_max_duration_s: float = 15.0,
    resegment_min_duration_s: float = 0.3,
    # Speaker-label refinement (between resegmentation and ASR).
    refine_speaker_labels: bool = False,
    speaker_purity_floor: float = 0.35,
    speaker_split_distance: float = 0.70,
    speaker_merge_ceiling: float = 0.55,
    speaker_min_speech_s: float = 4.0,
    # face stack
    face_detector: Optional[str] = None,          # None → skip faces entirely
    face_tracker: str = "bytetrack",
    face_every_n: int = 1,
    face_min_score: float = 0.5,
    # ASD + assignment
    asd_model: Optional[str] = None,              # None → skip ASD/assignment
    asd_on_screen_threshold: float = 0.3,
    # Stop phase one after MT, before any audio is synthesised, so the UI can
    # collect per-speaker gender + voice first; the cast route then runs
    # `run_post_mt_stages`. Default False: a CLI run goes end to end.
    stop_after_mt: bool = False,
    # ASR
    asr_model: Optional[str] = None,              # None → skip ASR
    #: What is SPOKEN in the source. None means detect it: the Bodhan
    #: adapter probes the longest turn once and pins that one answer to
    #: every turn, so a 1-5 s turn cannot be labelled on its own.
    #:
    #: Pinning matters more than it looks. Detecting PER TURN instead let
    #: one lecture come back as Telugu, Gujarati, Malayalam and English
    #: across nine turns, with MT then translating each as if its script
    #: were real. The fix is the single probe, not the hardcoded code.
    asr_language: Optional[str] = None,
    asr_full_audio: bool = False,                 # True → transcribe raw 16k
    # Use the uploader's own published transcript instead of running ASR,
    # when a HUMAN-AUTHORED one exists for the ASR language. On by default:
    # it is free, it is usually better than anything inferred from the
    # audio, and every way it can go wrong falls back to ASR rather than
    # failing. `captions_allow_auto` additionally accepts YouTube's
    # auto-generated track -- off by default, because that is YouTube's own
    # ASR and taking it would silently replace the engine we chose.
    captions_as_transcript: bool = True,
    captions_allow_auto: bool = False,
                                                  # mix end-to-end and skip the
                                                  # turn-intersection step
    # Forced alignment (word boundaries). Only runs when ASR did.
    align_model: Optional[str] = None,            # None → skip alignment
    # Machine translation. Only runs when ASR did.
    mt_model: Optional[str] = None,               # None → skip MT
    mt_target_language: Optional[str] = None,     # e.g. "hi"
    mt_target_script: Optional[str] = None,       # "native" | "roman" | None
    # Multi-candidate MT (method M8). These are TRANSLATE-time controls, so
    # they are handed to `translate_asr` explicitly rather than through
    # `_pick`, which carries adapter CONSTRUCTOR kwargs. `translate_asr`
    # filters them against the chosen adapter's translate signature, so an
    # adapter that never heard of them (bodhan) is unaffected.
    # `mt_n_best=1` is the historical single-beam call, unchanged.
    mt_n_best: int = 1,
    mt_length_penalties: Optional[Sequence[float]] = None,
    mt_diversity_penalty: Optional[float] = None,
    # Code-mix / script normalisation (plan §6).
    translit_model: Optional[str] = None,         # None → skip codemix pass
    termbase_path: Optional[Path] = None,
    # Duration-aware text adjustment (plan §7).
    duration_adjust: bool = False,
    duration_voice: str = "default",
    duration_safety_margin_s: float = 0.10,
    # 9c2. LLM rewrite of lines the tempo stretch cannot fit. Off by default
    # on the CLI; the UI turns it ON (`server/presets.py`), since a UI run has no
    # operator watching stage 9c's overrun warnings go past.
    condense_overlong_text: bool = False,
    condense_model: Optional[str] = None,
    condense_speed_cap: float = 1.5,
    condense_tolerance: float = 0.0,
    condense_batch_size: Optional[int] = None,
    # Voice assignment per speaker (plan §8).
    voice_engine: Optional[str] = None,           # "indic-speak"; derived from --tts
    voice_catalog_json: Optional[Path] = None,
    allow_voice_reuse: bool = False,
    # TTS synthesis.
    tts_model: Optional[str] = None,              # None → skip TTS
    tts_style: Optional[str] = None,              # engine delivery hint
    emotion_model: Optional[str] = None,          # None → skip stage 9b2
    emotion_min_confidence: float = 0.60,
    emotion_min_run: int = 2,
    channel_style_overrides: Optional[Dict[str, str]] = None,
    tts_sample_rate: int = 24000,
    tts_max_stretch: float = 0.20,
    stretch_backend: str = "auto",
    tts_apply_stretch: bool = True,
    # Engine-side speed control (method M1). Forwarded straight through to
    # `synthesize_segments`; "off" is the historical behaviour.
    tts_speed_control: str = "off",               # off|oneshot|closed_loop
    tts_speed_bounds: Tuple[float, float] = (0.85, 1.15),
    tts_speed_tolerance: float = 0.05,
    tts_speed_max_rounds: int = 2,
    # Post-TTS forced alignment (method M5), stage 9f1 — fills
    # `segments[].tts_words[]` so the word_anchored (M4) and dp (M7) timing
    # producers have real TTS word boundaries instead of falling back.
    tts_force_align: bool = False,
    tts_align_model: str = "mms_fa",
    # Viseme extraction (method M6), stage 9f1b — fills
    # `segments[].src_visemes[]` / `tts_visemes[]`, the only input to
    # `plan_dp`'s `_d_phoneme` term. Pair it with
    # `timing_params={"lambda_phoneme": <w>}`; with the default weight of 0.0
    # the term is short-circuited and this stage changes nothing.
    attach_visemes_stage: bool = False,
    viseme_backend: str = "mms_char",
    viseme_fallback_backend: Optional[str] = "allosaurus",
    viseme_source_audio: Optional[Path] = None,
    # Timing plan (methods M3/M4/M7), built between TTS and mix.
    timing_producer: Optional[str] = None,        # naive_slot|pause_aware|
                                                  # word_anchored|dp
    timing_params: Optional[Dict[str, Any]] = None,
    # Final mix + mux (requires TTS).
    mix: bool = False,
    mix_sample_rate: int = 48000,
    mix_tts_gain_db: float = 3.0,
    mix_me_gain_db: float = -3.0,
    mix_strategy: str = "global_tempo",
    mix_tempo_cap: float = 1.5,
    mix_clip_source: str = "auto",
    mux_video: bool = True,
    # Snapshot the dub into `mix/versions/` under this tag -- see
    # `run_post_mt_stages`. None skips the snapshot.
    mix_producer_tag: Optional[str] = None,
    # Lip sync (requires mix).
    lipsync_model: Optional[str] = None,           # wav2lip | musetalk | latentsync
    restore_model: Optional[str] = None,          # 9i: gfpgan | None. Never
                                                  # implied by --lipsync: it
                                                  # measured 2.6x the render
                                                  # it cleans (1156s vs 447s)
    lipsync_num_gpus: Optional[int] = None,       # None → ONE gpu, the freest
    lipsync_batch_size: Optional[int] = None,     # None → the adapter's own default
                                                  # device; 1 → single-GPU; N>1 →
                                                  # time-shard across N GPUs
    stem_model: str = "htdemucs",                 # htdemucs | htdemucs_ft |
                                                  # mdx_extra | mdx_extra_q
    # Ground-truth transcripts. If `subtitle_langs` is left None, we still
    # auto-request the ASR source language and the MT target language when
    # those stages are enabled — this is the "just evaluate everything we
    # can" default. Pass an explicit list to override.
    # Experiment isolation. `variant_tag` redirects stages 9b..9h into
    # `<exp_root>/<base>/<variant_tag>/` against a private copy of the merged
    # JSON, so several timing variants can share one upstream pass instead of
    # overwriting each other's in-place mutations. `post_mt_only_from` is the
    # other half: skip stages 1..9a and replay the chain on an existing merge.
    variant_tag: Optional[str] = None,
    exp_root: Optional[Path] = None,              # default: <job>/exp
    post_mt_only_from: Optional[Path] = None,
    # Job folder. Every run gets one under `data/jobs/`; pass an explicit id
    # to name it yourself (e.g. to resume into an existing folder) instead of
    # taking the generated `<timestamp>__<input>__<config>`.
    job_id: Optional[str] = None,
    subtitle_langs: Optional[List[str]] = None,
    **adapter_kwargs,
) -> PipelineResult:
    """Run download → extract → separate → diarize → (faces → ASD → ASR).

    `adapter_kwargs` is a flat bag forwarded to every registry call; each
    adapter picks what it accepts and drops the rest (with a warning). So
    passing `hf_token=…, checkpoint=…, render_mode=…` at once is safe.
    """
    data_dir = Path(data_dir)

    # Every run is a job: `data/jobs/<job_id>/` holds the input and every
    # stage output, so nothing this run produces lands outside its own
    # folder. The stage directories keep their familiar names — only the
    # root they hang off changes, so no stage code needed rewriting.
    job = _jobs.allocate(
        data_dir, input_,
        config={
            "diarization":     diarization_model,
            "asr":             asr_model,
            "align":           align_model,
            "mt":              mt_model,
            "target_language": mt_target_language,
            "target_script":   mt_target_script,
            "asr_language":    asr_language,
            "tts":             tts_model,
            "voice_engine":    voice_engine,
            "lipsync":         lipsync_model,
            # 9i. Recorded because `server/progress.plan_stages` branches on it:
            # a switch the planner reads but the manifest does not hold means
            # a finished job re-plans to a different shape than it ran.
            "restore":         restore_model,
            "stem_model":      stem_model,
            "skip_stems":      skip_stems,
            "timing_producer": timing_producer,
            "variant_tag":     variant_tag,
            # Every OPTIONAL stage's switch, under the key name
            # `server/progress.py::plan_stages` reads. These were missing, and
            # the manifest is the only durable record of what a run did:
            # `_run_pipeline` re-allocates here and OVERWRITES whatever the
            # caller wrote at submission, so the UI's richer config lasted
            # until the pipeline started and no longer. Everything that
            # reads the manifest afterwards — `server/service/spec.py`, `server/service/cost.py`,
            # `_list_jobs`, `edit_agent._config` — therefore saw a run with
            # no resegmentation, no speaker refinement, no register, no
            # emotion, no visemes and no face detector, none of which was
            # true. `plan_stages` on a finished job under-planned by 7
            # stages against a progress doc that had all 25.
            "resegment_turns": bool(resegment_turns),
            "refine_speakers": bool(refine_speaker_labels),
            "face_detector":   face_detector,
            "asd":             asd_model,
            "translit":        translit_model,
            "tts_style":       tts_style,
            "emotion":         emotion_model,
            "duration_adjust": bool(duration_adjust),
            "condense_overlong_text": bool(condense_overlong_text),
            "tts_force_align": bool(tts_force_align),
            "visemes":         bool(attach_visemes_stage),
            "mix":             bool(mix),
        },
        job_id=job_id,
        argv=list(sys.argv[1:]) or None,
    )
    job_root = job.root
    videos_dir = job.stage("videos")
    audio_dir = job.stage("audio")
    stems_dir = job.stage("stems")
    dia_dir = job.stage("diarization")
    faces_dir = job.stage("faces")
    asd_dir = job.stage("asd")
    asr_dir = job.stage("asr")
    align_dir = job.stage("align")
    mt_dir = job.stage("mt")
    eval_dir = job.stage("eval")
    voices_dir = job.stage("voices")
    tts_dir = job.stage("tts")
    mix_dir = job.stage("mix")
    lipsync_dir = job.stage("lipsync")

    exp_root = Path(exp_root) if exp_root is not None else job.stage("exp")

    #: Filled by `resolve_input` for a URL input. A LOCAL file leaves it
    #: empty -- and that is the normal case for a UI run, which pre-cuts the
    #: clip and hands the pipeline a path, so `download_youtube` is never
    #: reached here. `_recover_source_metadata` below picks the uploader's
    #: metadata back up off disk in that case. Also written to the manifest
    #: so a register choice is traceable.
    source_metadata: Dict[str, Any] = {}

    if post_mt_only_from is not None:
        log.info("=== stages 1-9a SKIPPED (post-MT replay) ===")
        result, video, dia_input, me_stem, turns = _bootstrap_post_mt_only(
            input_, post_mt_only_from,
            data_dir=data_dir,
            diarization_model=diarization_model,
            stem_model=stem_model,
            skip_stems=skip_stems,
            asr_language=asr_language,
            mt_model=mt_model,
            mt_target_language=mt_target_language,
            mt_target_script=mt_target_script,
            videos_dir=videos_dir,
        )
        # The merged JSON is the only record of what stages 1..9a decided.
        asr_language = result.asr_language
        mt_target_language = result.mt_target_language
        mt_target_script = result.mt_target_script
        subtitles = {}
        asr_result = None
    else:
        # 1. Resolve input (download if URL) — also fetch ground-truth subtitles
        #    for whichever languages the caller wants to evaluate against.
        log.info("=== stage 1: download / resolve ===")
        requested_langs: List[str] = list(subtitle_langs or [])
        if not requested_langs:
            if asr_language:
                requested_langs.append(asr_language)
            if mt_target_language and mt_target_language not in requested_langs:
                requested_langs.append(mt_target_language)
        subtitles: Dict[str, Dict[str, str]] = {}
        video = resolve_input(
            input_, download_dir=videos_dir,
            subtitle_langs=requested_langs or None,
            subtitles_out=subtitles,
            metadata_out=source_metadata,
        )
        video = _adopt_into_job(video, videos_dir)
        if subtitles:
            log.info("transcripts: fetched %d lang(s): %s",
                     len(subtitles),
                     ", ".join(f"{k}({v['source']})" for k, v in subtitles.items()))

        # 2. Audio extraction — 16 kHz mono WAV, the diarization-friendly format
        log.info("=== stage 2: extract audio ===")
        audio_16k = audio_dir / f"{video.stem}.16k.mono.wav"
        extract_audio(video, audio_16k, sample_rate=16_000, mono=True)

        # 3. Stem separation (HTDemucs wants stereo 44.1 kHz).
        dialogue_stem: Optional[Path] = None
        me_stem: Optional[Path] = None
        if not skip_stems:
            log.info("=== stage 3: stem separation (HTDemucs) ===")
            demucs_input = audio_dir / f"{video.stem}.44k.stereo.wav"
            extract_audio(video, demucs_input, sample_rate=44_100, mono=False)
            stems = separate_stems(demucs_input, stems_dir,
                                   model=stem_model, device=device)
            dialogue_stem = stems.dialogue
            me_stem = stems.music_effects
        else:
            log.info("=== stage 3: SKIPPED ===")

        # Claim the shared C libraries before pyannote does. Must stay ABOVE
        # stage 4: importing it after costs stage 5 the whole face stage with
        # an error that blames a missing package. See the docstring.
        from backend.faces.faces import preimport_face_stack
        preimport_face_stack()

        # 4. Diarization — prefer the dialogue stem (cleaner input).
        log.info("=== stage 4: diarization (%s) ===", diarization_model)
        if dialogue_stem is not None:
            dia_input = audio_dir / f"{video.stem}.dialogue.16k.mono.wav"
            extract_audio(dialogue_stem, dia_input, sample_rate=16_000, mono=True)
        else:
            dia_input = audio_16k

        rttm_path = dia_dir / f"{video.stem}.{diarization_model}.rttm"
        diar_kwargs = _pick(adapter_kwargs, ("hf_token", "checkpoint", "batch_size"))
        diar_result = diarize_stage(
            dia_input,
            model=diarization_model,
            device=device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            rttm_out=rttm_path,
            **diar_kwargs,
        )

        # 4b. Logical resegmentation (optional).
        if resegment_turns:
            log.info("=== stage 4b: logical resegmentation ===")
            diar_result = resegment(
                diar_result,
                audio_path=dia_input,
                split_min_duration_s=resegment_split_min_s,
                split_pause_s=resegment_split_pause_s,
                merge_max_gap_s=resegment_merge_max_gap_s,
                merge_max_result_s=resegment_merge_max_result_s,
                max_duration_s=resegment_max_duration_s,
                min_duration_s=resegment_min_duration_s,
            )
            reseg_rttm = dia_dir / f"{video.stem}.{diarization_model}.resegmented.rttm"
            diar_result.write_rttm(reseg_rttm)
            log.info("resegment: wrote %s", reseg_rttm)

        # 4c. Speaker-label refinement (optional). Diarization labels are
        #     already global -- SPK_00 early is SPK_00 late -- but on noisy
        #     material pyannote under-clusters and one label holds several
        #     people. Since stage 9d assigns one voice PER LABEL, that reads
        #     several characters in one voice while every artifact looks
        #     correct. Audio only, and before face detection, so the stage
        #     order stays acyclic.
        if refine_speaker_labels:
            log.info("=== stage 4c: speaker label refinement ===")
            diar_result = _refine_speaker_labels(
                diar_result,
                audio_path=dia_input,
                device=device,
                purity_floor=speaker_purity_floor,
                split_distance=speaker_split_distance,
                merge_ceiling=speaker_merge_ceiling,
                min_speaker_s=speaker_min_speech_s,
                max_speakers=max_speakers,
            )
            refined_rttm = dia_dir / f"{video.stem}.{diarization_model}.refined.rttm"
            diar_result.write_rttm(refined_rttm)
            log.info("speaker_id: wrote %s", refined_rttm)

        result = PipelineResult(
            input=input_,
            video=video,
            audio_16k=audio_16k,
            dialogue_stem=dialogue_stem,
            me_stem=me_stem,
            rttm=rttm_path,
            num_turns=len(diar_result.turns),
            num_speakers=diar_result.num_speakers,
            diarization_model=diar_result.model_name,
            manifest=dia_dir / f"{video.stem}.{diarization_model}.pipeline.json",
            stems_skipped=skip_stems,
            subtitles=subtitles,
        )

        # 5. Face detection + tracking (optional).
        tracks = None
        if face_detector:
            log.info("=== stage 5: face detect+track (%s + %s) ===",
                     face_detector, face_tracker)
            tracks_npz = faces_dir / f"{video.stem}.{face_detector}_{face_tracker}.npz"
            tracks = detect_and_track(
                video,
                detector=face_detector,
                tracker=face_tracker,
                device=device,
                every_n=face_every_n,
                min_score=face_min_score,
                tracks_out=tracks_npz,
                detector_kwargs=_pick(adapter_kwargs, ("pack", "input_size", "det_thresh",
                                                        "model_selection",
                                                        "min_detection_confidence")),
                tracker_kwargs=_pick(adapter_kwargs, ("track_activation_threshold",
                                                        "lost_track_buffer",
                                                        "minimum_matching_threshold",
                                                        "minimum_consecutive_frames")),
            )
            result.face_tracks_npz = tracks_npz
            result.num_face_tracks = len(tracks.tracks)
            result.face_detector = face_detector
            result.face_tracker = face_tracker

        # 6. ASD + Hungarian speaker↔face assignment -- RELOCATED 2026-10-26.
        # ASD no longer runs mid-pipeline. It runs at the post-MT cast gate
        # (so stage 9d's face-gender vote has an attribution) and at the
        # lip-sync job (so the adapter animates the speaking face), via
        # `_run_asd_stage`. `assignments` stays None here; the per-segment
        # face fold below is therefore empty until the gate re-folds it.
        assignments: Optional[List] = None

        # 6b. Gender for every speaker, on the GATED flow only.
        #
        # A run that stops after MT asks the operator to cast voices, and it
        # was asking with an empty gender column because gender came out of
        # stage 9d, past the gate. Predicting here fills the column while ASR
        # and MT are still running, so it is on screen by the time the
        # question is asked. ASD comes with it -- see `_run_gender_stage` for
        # why that costs a cast job nothing.
        #
        # A CLI run never stops and is untouched: gender stays in 9d, ASD
        # stays where 2026-10-26 put it.
        if stop_after_mt:
            log.info("=== stage 6b: predicting speaker gender ===")
            _run_gender_stage(
                job_root, video,
                turns=diar_result.turns,
                reference_audio=dia_input,
                voices_dir=voices_dir,
                asd_model=asd_model,
                asd_on_screen_threshold=asd_on_screen_threshold,
                device=device, adapter_kwargs=adapter_kwargs,
                result=result,
            )

        # 7. ASR (optional) -- or the uploader's own transcript, when there is
        #    a human-authored one. A published manual caption track is free
        #    and is usually better than anything inferred from the audio, so
        #    the hosted ASR call per turn is pure cost in that case. Automatic
        #    tracks are NOT accepted by default: those are YouTube's own ASR,
        #    so taking them would swap our chosen engine for an unconfigurable
        #    one on exactly the Indic-accented English this repo processes.
        #    `backend/caption_asr.py` returns None for every "no" answer --
        #    wrong language, auto-only, rolling duplication, thin coverage --
        #    so a caption problem costs the saving, never the run.
        caption_result = None
        if asr_model and captions_as_transcript:
            if not asr_language:
                # Caption tracks are selected by language code, and this
                # runs BEFORE ASR, so with detection pending there is no
                # code to select on. Guessing one picks the wrong track on
                # exactly the non-English sources detection exists for.
                log.info("captions: skipped -- the source language is being "
                         "detected, and a track can only be matched by code")
            else:
                caption_result = _captions_from(
                    subtitles, audio_16k, diar_result, asr_full_audio,
                    language=asr_language,
                    allow_auto=captions_allow_auto)
        if caption_result is not None:
            cap = caption_result.extra.get("captions", {})
            # ONE line, deliberately. `server/progress.py` scrapes these markers
            # -- a marker split across two source lines is captured without
            # its closing `===` and silently stops being matched, which
            # freezes the progress bar.
            log.info("=== stage 7: transcript from captions (%s, %s) ===",
                     cap.get("source", "?"), asr_language)
            asr_result = caption_result
        elif asr_model:
            log.info("=== stage 7: ASR (%s, full_audio=%s) ===",
                     asr_model, asr_full_audio)
            # Full-audio mode: transcribe the raw 16 kHz mix end-to-end, so
            # anything the stem separator or diarizer missed (intros under
            # music, quiet asides) still lands in the transcript. Turn-sliced
            # mode preserves the historical per-turn segment manifest.
            asr_source = audio_16k if asr_full_audio else dia_input
            asr_turns = None if asr_full_audio else diar_result.turns
            asr_result = asr_transcribe(
                asr_source, model=asr_model, device=device,
                language=asr_language, turns=asr_turns,
                **_pick(adapter_kwargs, ("base_url", "api_key", "model",
                                          "render_mode", "mode",
                                          "timeout_s", "max_retries",
                                          "checkpoint", "compute_type",
                                          "batch_size", "language", "align",
                                          "condition_on_previous_text",
                                          "no_speech_threshold",
                                          "vad_onset", "vad_offset",
                                          "repo_id")),
            )

        # Shared by BOTH stage-7 paths. This used to live inside the ASR
        # branch; a caption-sourced transcript would then have produced no
        # `asr.json`, no transcript.txt and no `result.asr_*` fields, so the
        # manifest would describe a run with no transcript while stage 8
        # happily aligned one. The artifacts are the durable record -- they
        # cannot depend on which path produced the text.
        if asr_model and asr_result is not None:
            asr_tag = ("captions" if caption_result is not None else asr_model)
            # Fold speaker labels + face assignments into each segment.
            asd_by_seg = {a.seg_id: a for a in (assignments or [])}
            payload = asr_result.to_jsonable()
            for seg in payload["segments"]:
                a = asd_by_seg.get(seg["id"])
                if a is not None:
                    seg["face_track"] = a.face_track
                    seg["asd_score"] = (a.asd_score if a.asd_score != float("-inf") else None)
                    seg["on_screen"] = a.on_screen
            asr_json = asr_dir / f"{video.stem}.{asr_tag}.asr.json"
            asr_json.write_text(json.dumps(payload, indent=2))
            asr_txt = asr_dir / f"{video.stem}.{asr_tag}.transcript.txt"
            asr_txt.write_text(
                "\n".join(s["text"].strip() for s in payload["segments"] if s.get("text")) + "\n",
                encoding="utf-8",
            )
            result.asr_model = asr_tag
            result.asr_language = asr_result.language
            result.asr_json = asr_json

        # 8. Forced alignment — fills in word-level timings. Requires ASR output
        #    and the audio the ASR saw (dialogue stem when present).
        if align_model:
            if not asr_model:
                log.warning("alignment requested but ASR was skipped; nothing to align")
            else:
                log.info("=== stage 8: forced alignment (%s) ===", align_model)
                align_result = align_asr(
                    dia_input, asr_result,
                    model=align_model, device=device,
                    language=asr_result.language or asr_language,
                )
                align_json = align_dir / f"{video.stem}.{align_model}.align.json"
                align_json.write_text(json.dumps(
                    align_result.to_jsonable(), indent=2, ensure_ascii=False,
                ))
                # Re-write the ASR JSON with word timings folded in — downstream
                # stages read one file, not two.
                asr_result = apply_alignment_to_asr(asr_result, align_result)
                asr_payload = asr_result.to_jsonable()
                # Preserve speaker/face metadata we folded in above.
                prev = json.loads(asr_json.read_text())
                prev_by_id = {s["id"]: s for s in prev.get("segments", [])}
                for seg in asr_payload["segments"]:
                    prev_seg = prev_by_id.get(seg["id"], {})
                    for k in ("speaker", "face_track", "asd_score", "on_screen"):
                        if k in prev_seg and prev_seg[k] is not None:
                            seg[k] = prev_seg[k]
                asr_json.write_text(json.dumps(asr_payload, indent=2,
                                                ensure_ascii=False))
                result.align_model = align_model
                result.align_json = align_json
                result.num_aligned_words = sum(len(s.words) for s in align_result.segments)

        # 9. Machine translation — one call per ASR segment.
        if mt_model:
            if not asr_model:
                log.warning("translation requested but ASR was skipped; nothing to translate")
            elif not mt_target_language:
                log.warning("translation requested but mt_target_language is empty; skipping")
            else:
                log.info("=== stage 9: machine translation (%s → %s, script=%s) ===",
                         mt_model, mt_target_language, mt_target_script or "default")
                mt_result = translate_asr(
                    asr_result,
                    target_language=mt_target_language,
                    target_script=mt_target_script,
                    model=mt_model, device=device,
                    source_language=asr_result.language,
                    # Translate-time, so NOT via `_pick` — that allowlist feeds
                    # `get_mt()` constructor kwargs and would raise on these.
                    n_best=mt_n_best,
                    length_penalties=mt_length_penalties,
                    diversity_penalty=mt_diversity_penalty,
                    **_pick(adapter_kwargs, ("base_url", "api_key", "model",
                                              "timeout_s", "max_retries",
                                              "batch_size", "dtype",
                                              "max_new_tokens")),
                )
                mt_json = mt_dir / f"{video.stem}.{mt_model}.{mt_target_language}.mt.json"
                mt_txt = mt_dir / f"{video.stem}.{mt_model}.{mt_target_language}.transcript.txt"
                write_mt_outputs(mt_json, mt_txt, asr_result, mt_result)
                # v000. The pristine translation, before any post-MT stage
                # rewrites it in place. Everything after this point edits
                # the same file, so without this snapshot the original
                # stops existing as soon as 9b runs.
                from .mt_versions import snapshot as _mt_snapshot
                _mt_snapshot(mt_json, "original")
                result.mt_model = mt_model
                result.mt_target_language = mt_target_language
                result.mt_target_script = mt_target_script
                result.mt_json = mt_json
                result.mt_txt = mt_txt
        turns = diar_result.turns

    # Variant isolation, applied between 9a and 9b: hand the post-MT chain its
    # own copy of the merged JSON and its own output tree. Each stage keeps
    # mutating one file in place — it is just no longer the shared one.
    post_mt_out_root = job_root
    if variant_tag:
        variant_dir = exp_root / video.stem / variant_tag
        variant_dir.mkdir(parents=True, exist_ok=True)
        if result.mt_json is not None:
            plan_json = variant_dir / "plan.mt.json"
            shutil.copy2(result.mt_json, plan_json)
            result.mt_json = plan_json
            log.info("variant %r: merged JSON → %s", variant_tag, plan_json)
        # The run manifest moves too, or every variant would rewrite the one
        # shared diarization manifest (and feed it to lip-sync).
        result.manifest = variant_dir / "pipeline.json"
        post_mt_out_root = variant_dir

    # The UI downloads and cuts before calling us, so `source_metadata` is
    # empty on exactly the runs that have the richest metadata sitting next
    # to the source video. Recover it rather than letting every UI run fall
    # through to the LLM tier and answer "unknown".
    if tts_style == "auto" and not source_metadata:
        recovered = _recover_source_metadata(job_root, video)
        if recovered:
            source_metadata.update(recovered)

    # `--tts-style auto` — pick the programme's register from the uploader's
    # own metadata, falling back to an LLM read of the transcript. Resolved
    # here because this is the first point where BOTH the metadata (from the
    # download) and a transcript (from ASR/MT) exist. A None answer means
    # "no global style", i.e. the voice's neutral reading, which is also what
    # happens when the video came from a local file with no metadata.
    if tts_style == "auto":
        from backend.translate.translate import classify_register, transcript_sample
        sample = ""
        try:
            if result.mt_json and Path(result.mt_json).exists():
                sample = transcript_sample(
                    json.loads(Path(result.mt_json).read_text()))
        except Exception as e:                       # noqa: BLE001
            log.warning("classify: could not sample transcript (%s)", e)
        tts_style, tier = classify_register(
            source_metadata, transcript=sample,
            channel_overrides=channel_style_overrides,
        )
        log.info("=== stage 9b1: content register = %r (via %s) ===",
                 tts_style, tier)
        # `PipelineResult` has no `extra` field — it is a flat, JSON-
        # serialised dataclass — so this used to raise AttributeError and
        # fail the whole job at 9b1, after ASR, alignment and MT had
        # already been paid for. The register is carried on the result the
        # same way every other stage's output is: as a named field.
        result.content_register = tts_style
        result.content_register_tier = tier

    # 9b..9h — code-mix → duration → condense → emotion → voices → TTS →
    # mix → lip-sync. The UI's gated flow (`stop_after_mt`) runs the TEXT
    # stages (9b, 9c/9c2, 9b2) in phase one so the operator sees final text
    # at the cast table; the cast route picks up from 9d (voices) onward.
    # A CLI run never stops (stop_after_mt False).
    post = run_post_mt_stages(
        result.mt_json,
        source_video=video,
        me_stem=me_stem,
        reference_audio_src=dia_input,
        out_root=post_mt_out_root,
        restore_model=restore_model,
        target_language=mt_target_language,
        target_script=mt_target_script,
        source_language=asr_language,
        device=device,
        turns=turns,
        manifest=result.manifest,
        asd_model=asd_model,
        asd_on_screen_threshold=asd_on_screen_threshold,
        asd_adapter_kwargs=adapter_kwargs,
        translit_model=translit_model,
        termbase_path=termbase_path,
        duration_adjust=duration_adjust,
        duration_voice=duration_voice,
        duration_safety_margin_s=duration_safety_margin_s,
        duration_cache_dir=DURATION_MODEL_DIR,
        condense_overlong_text=condense_overlong_text,
        condense_model=condense_model,
        condense_speed_cap=condense_speed_cap,
        condense_tolerance=condense_tolerance,
        condense_batch_size=condense_batch_size,
        voice_engine=voice_engine,
        voice_catalog_json=voice_catalog_json,
        allow_voice_reuse=allow_voice_reuse,
        tts_model=tts_model,
        tts_style=tts_style,
        emotion_model=emotion_model,
        emotion_min_confidence=emotion_min_confidence,
        emotion_min_run=emotion_min_run,
        tts_sample_rate=tts_sample_rate,
        tts_max_stretch=tts_max_stretch,
        stretch_backend=stretch_backend,
        tts_apply_stretch=tts_apply_stretch,
        tts_speed_control=tts_speed_control,
        tts_speed_bounds=tts_speed_bounds,
        tts_speed_tolerance=tts_speed_tolerance,
        tts_speed_max_rounds=tts_speed_max_rounds,
        tts_force_align=tts_force_align,
        tts_align_model=tts_align_model,
        attach_visemes_stage=attach_visemes_stage,
        viseme_backend=viseme_backend,
        viseme_fallback_backend=viseme_fallback_backend,
        viseme_source_audio=viseme_source_audio,
        timing_producer=timing_producer,
        timing_params=timing_params,
        mix=mix,
        mix_sample_rate=mix_sample_rate,
        mix_tts_gain_db=mix_tts_gain_db,
        mix_me_gain_db=mix_me_gain_db,
        mix_strategy=mix_strategy,
        mix_tempo_cap=mix_tempo_cap,
        mix_clip_source=mix_clip_source,
        mux_video=mux_video,
        mix_producer_tag=mix_producer_tag,
        lipsync_model=lipsync_model,
        lipsync_num_gpus=lipsync_num_gpus,
        lipsync_batch_size=lipsync_batch_size,
        stop_before_voices=stop_after_mt,
        **adapter_kwargs,
    )
    if stop_after_mt:
        # Phase one ran the text stages (9b/9c/9c2/9b2) and stopped before
        # voices. Copy their results, then mark the job for the cast table.
        result.translit_model = post.translit_model
        result.codemix_retained_terms = post.codemix_retained_terms
        result.duration_model_language = post.duration_model_language
        result.duration_model_voice = post.duration_model_voice
        result.duration_estimated_violations = post.duration_estimated_violations
        result.awaiting_cast = True
    else:
        result.mt_json = post.mt_json
        result.translit_model = post.translit_model
        result.codemix_retained_terms = post.codemix_retained_terms
        result.duration_model_language = post.duration_model_language
        result.duration_model_voice = post.duration_model_voice
        result.duration_estimated_violations = post.duration_estimated_violations
        result.voice_engine = post.voice_engine
        result.voice_assignments_json = post.voice_assignments_json
        result.speaker_references_dir = post.speaker_references_dir
        result.tts_model = post.tts_model
        result.tts_out_dir = post.tts_out_dir
        result.tts_budget_violations = post.tts_budget_violations
        result.tts_errors = post.tts_errors
        result.timing_producer = post.timing_producer
        result.timing_plan_json = post.timing_plan_json
        result.timing_summary = post.timing_summary
        result.mix_wav = post.mix_wav
        result.mix_mp4 = post.mix_mp4
        result.lipsync_model = post.lipsync_model
        result.lipsync_mp4 = post.lipsync_mp4
        result.lipsync_error = post.lipsync_error
        # ASD now runs inside the post-MT stages (relocated stage 6), so its
        # outputs come back on `post` -- carry them onto the sealed result so
        # the manifest and the edit-replay both see which model scored.
        result.asd_model = post.asd_model or result.asd_model
        result.assignments_json = post.assignments_json or result.assignments_json
        result.num_on_screen = post.num_on_screen or result.num_on_screen

    # 10. Evaluate ASR against the source-language subtitle (if we have one).
    #     Produces WER / CER so a user can see at a glance how far the ASR
    #     model's transcript drifted from the publisher's own captions.
    if result.asr_json and subtitles:
        src_lang = (asr_result.language if asr_result is not None
                    else None) or asr_language
        sub = _pick_subtitle(subtitles, src_lang)
        if sub is not None:
            lang_key, sub_meta = sub
            log.info("=== stage 10a: ASR vs transcript (%s, source=%s) ===",
                     lang_key, sub_meta["source"])
            try:
                report = evaluate_asr(
                    result.asr_json, Path(sub_meta["path"]),
                    language=lang_key, source=sub_meta["source"],
                )
                asr_eval = eval_dir / f"{video.stem}.{asr_model}.{lang_key}.asr_eval.json"
                write_report(report, asr_eval)
                result.asr_eval_json = asr_eval
                log.info("ASR WER=%.3f CER=%.3f (vs %d ref words)",
                         report["wer"]["wer"], report["cer"]["cer"],
                         report["wer"]["n_ref_words"])
            except Exception as e:
                log.warning("ASR evaluation failed: %s", e)
        else:
            log.info("ASR eval skipped: no subtitle for language %r "
                     "(have: %s)", src_lang, ", ".join(subtitles) or "none")

    # 11. Evaluate MT against the target-language dubbed subtitle (if we have one).
    if result.mt_json and subtitles:
        sub = _pick_subtitle(subtitles, mt_target_language)
        if sub is not None:
            lang_key, sub_meta = sub
            log.info("=== stage 10b: MT vs dubbed transcript (%s, source=%s) ===",
                     lang_key, sub_meta["source"])
            try:
                report = evaluate_mt(
                    result.mt_json, Path(sub_meta["path"]),
                    target_language=lang_key, source=sub_meta["source"],
                )
                mt_eval = eval_dir / f"{video.stem}.{mt_model}.{lang_key}.mt_eval.json"
                write_report(report, mt_eval)
                result.mt_eval_json = mt_eval
                log.info("MT BLEU=%.3f chrF=%.3f ROUGE-L=%.3f METEOR=%.3f",
                         report["bleu"]["bleu"], report["chrf"]["chrf"],
                         report["rougeL"]["f1"], report["meteor"]["meteor"])
            except Exception as e:
                log.warning("MT evaluation failed: %s", e)
        else:
            log.info("MT eval skipped: no subtitle for target language %r "
                     "(have: %s)", mt_target_language,
                     ", ".join(subtitles) or "none")

    result.to_json(result.manifest)

    # Seal the job: stamp the terminal status and inventory everything that
    # landed under `data/jobs/<job_id>/`. The inventory is a walk of the job
    # tree, so it reports what the run actually wrote, not what it intended.
    result.job_id = job.job_id
    result.job_dir = job_root
    try:
        job_manifest = job.finalize(result, status="ok")
        log.info("job manifest → %s", job_manifest)
    except Exception as e:
        log.warning("failed to write job manifest: %s", e)

    log.info(
        "pipeline done: %d turns, %d speakers, %d face tracks, %d on-screen → %s",
        result.num_turns, result.num_speakers,
        result.num_face_tracks, result.num_on_screen,
        result.rttm,
    )
    return result


@functools.wraps(_run_pipeline)
def run_pipeline(*args, **kwargs) -> PipelineResult:
    """`_run_pipeline`, but a crash still closes out the job folder.

    Without this a failed run leaves `manifest.json` reading "running"
    forever, which is indistinguishable from a job that is still going.
    `functools.wraps` keeps the real signature visible to callers that
    introspect it.
    """
    try:
        return _run_pipeline(*args, **kwargs)
    except BaseException as e:
        job = _jobs.current()
        if job is not None and job.read().get("status") == "running":
            try:
                job.finalize(status="failed", error=f"{type(e).__name__}: {e}")
                log.error("job %s marked failed → %s", job.job_id, job.root)
            except Exception:
                log.exception("could not mark job %s failed", job.job_id)
        raise


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _captions_from(subtitles, audio_16k, diar_result, asr_full_audio,
                   *, language: str, allow_auto: bool):
    """The uploader's own transcript as an ASRResult, or None to run ASR.

    Kept as a wrapper so stage 7 reads as one decision. Everything that can
    say "no" -- wrong language, automatic-only, rolling duplication, thin
    turn coverage, an unparseable file -- answers None rather than raising,
    because the caller's fallback is to run ASR and a caption problem must
    cost the saving, not the run.

    `turns` is passed ONLY in turn-sliced mode. In full-audio mode the
    pipeline deliberately keeps segments off the diarization grid, and
    forcing captions onto turns there would quietly re-impose it.
    """
    try:
        from backend.asr.asr import transcript_from_captions
    except Exception as e:                               # noqa: BLE001
        log.debug("captions: helper unavailable (%s)", e)
        return None
    turns = None if asr_full_audio else getattr(diar_result, "turns", None)
    return transcript_from_captions(
        subtitles, audio_path=audio_16k, turns=turns,
        language=language, allow_auto=allow_auto)


def _recover_source_metadata(job_root: Path, video: Optional[Path]) -> Dict[str, Any]:
    """Find the uploader metadata a previous download left in the job folder.

    `backend.downloader.download_youtube` writes `<stem>.meta.json` beside
    whatever it fetched. When the UI pre-cuts a clip that file lands under
    `<job>/source/`, the pipeline is handed the cut and never calls the
    downloader, and `source_metadata` stays empty -- so stage 9b1 skipped
    tiers 1-3 and asked an LLM to read a transcript, which answered
    "unknown". Measured on the W3L9 lecture: `categories: ["Education"]` was
    on disk the whole time and the run still recorded `register = None`.

    Returns {} when there is nothing to find, which keeps a genuinely
    metadata-less local file on the LLM tier where it belongs.
    """
    candidates: List[Path] = []
    if video is not None:
        # The cut keeps the source stem, so prefer a sibling of the input.
        candidates.append(Path(video).with_suffix(".meta.json"))
    for sub_dir in ("source", "videos"):
        d = Path(job_root) / sub_dir
        if d.is_dir():
            candidates.extend(sorted(d.glob("*.meta.json")))
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text())
        except Exception as e:                       # noqa: BLE001
            log.warning("classify: could not read %s (%s)", path.name, e)
            continue
        if isinstance(data, dict) and data:
            log.info("classify: recovered source metadata from %s", path.name)
            return data
    return {}


def _merged_has_clips(merged_json: Path) -> bool:
    """True when every segment of `merged_json` already points at a real clip.

    Lets the timing hook run on a merged JSON whose TTS pass happened in an
    earlier call (a replay, or a variant that reuses a donor's clips) instead
    of insisting that stage 9e ran in this very invocation.
    """
    try:
        data = json.loads(Path(merged_json).read_text())
    except (OSError, ValueError):
        return False
    segs = data.get("segments") or []
    return bool(segs) and all(
        seg.get("tts_raw_audio") and Path(seg["tts_raw_audio"]).exists()
        for seg in segs)


def _merged_has_any_clip(merged_json: Path) -> bool:
    """True when AT LEAST ONE segment points at a real raw TTS clip.

    Stage 9f1's guard, deliberately weaker than `_merged_has_clips`: the
    aligner skips a clipless segment on its own and records why, so demanding
    a complete set here would turn one missing clip into "no segment gets
    tts_words[]", which is the silent M4/M7 fallback the stage exists to stop.
    """
    try:
        data = json.loads(Path(merged_json).read_text())
    except (OSError, ValueError):
        return False
    return any(
        seg.get("tts_raw_audio") and Path(seg["tts_raw_audio"]).exists()
        for seg in (data.get("segments") or [])
    )


def _tts_align_health(data: dict, model: str) -> dict:
    """Stage-health counters for 9f1.

    A regression to the pause_aware fallback produces output that still looks
    plausible, so the only way it shows up is as numbers in the results table:
    how many segments actually carry `tts_words[]`, out of how many, and how
    much of the text the aligner managed to place.

    `token_coverage` is aligned TTS word tokens over whitespace tokens of the
    text handed to the aligner, counted only across segments where alignment
    was attempted (a clip was present and the text non-empty). It is None --
    reported as "not measured" -- when nothing was attempted, because 0.0 there
    would read as "aligned nothing" rather than "tried nothing".
    """
    segments = data.get("segments", []) or []
    stage = (data.get("extra", {}) or {}).get("tts_align", {}) or {}
    failures = stage.get("failures", []) or []
    # A segment skipped for no_audio/audio_missing/no_text was never offered to
    # the aligner, so its tokens are not coverage the aligner lost.
    not_attempted = {f.get("id") for f in failures
                     if f.get("reason") in ("no_audio", "audio_missing", "no_text")}
    with_words = words = expected = 0
    for seg in segments:
        tts_words = seg.get("tts_words") or []
        if tts_words:
            with_words += 1
            words += len(tts_words)
        if seg.get("id") in not_attempted:
            continue
        text = (seg.get("final_text") or seg.get("mt_text")
                or seg.get("text") or "")
        expected += len(text.split())
    return {
        "model": model,
        "segments_with_tts_words": with_words,
        "total": len(segments),
        "words_aligned": words,
        "failures": len(failures),
        "failure_reasons": dict(Counter(f.get("reason", "unknown")
                                        for f in failures)),
        "token_coverage": (round(words / expected, 4) if expected else None),
    }


def _pick(d: Dict[str, Any], keys) -> dict:
    """Return a sub-dict with only `keys` present in `d`."""
    return {k: d[k] for k in keys if k in d and d[k] is not None}


def _pick_subtitle(
    subs: Dict[str, Dict[str, str]], lang: Optional[str]
) -> Optional[tuple]:
    """Find the best subtitle entry for `lang`.

    Exact-match wins. Failing that, match on the primary subtag (so a
    request for 'en' lands on 'en-US' or 'en-GB'). Prefer "manual" and
    "local" sources over "auto" when several hits qualify, since
    human-authored captions are the honest reference for WER.
    """
    if not lang or not subs:
        return None
    if lang in subs:
        return lang, subs[lang]
    primary = lang.split("-")[0].lower()
    cands = [(k, v) for k, v in subs.items()
             if k.split("-")[0].lower() == primary]
    if not cands:
        return None
    cands.sort(key=lambda kv: 0 if kv[1].get("source") != "auto" else 1)
    return cands[0]
