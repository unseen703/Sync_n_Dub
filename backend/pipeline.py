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
        ▼   backend.asd.score_asd + assign_faces_to_turns (TalkNet|TS-TalkNet + Hungarian)
    SegmentAssignment[]
        │
        ▼   backend.asr.transcribe (Bodhan|WhisperX|IndicConformer)
    ASRResult (per-turn segments)
        │
        ▼   backend.align.align_asr (MMS-FA|WhisperX)   — word boundaries
    ASRResult with words[]
        │
        ▼   backend.translate.translate_asr (Bodhan|NLLB)
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
from .align import (align_asr, align_tts_segments, apply_alignment_to_asr,
                    attach_visemes)
from .asd import assign_faces_to_turns, assignments_to_jsonable, score_asd
from .asr import transcribe as asr_transcribe
from .codemix import codemix_segments
from .diarize import diarize as diarize_stage
from .segment import resegment
from .speaker_id import refine_speakers as _refine_speaker_labels
from .downloader import resolve_input
from .duration import plan_segment_durations
from .faces import detect_and_track
from .lipsync import run_lipsync
from .media_io import extract_audio
from .mix import mix_dubbed_audio
from .stem_separation import separate_stems
from .transcripts import evaluate_asr, evaluate_mt, write_report
from .translate import translate_asr, write_mt_outputs
from .tts import synthesize_segments
from .voice_convert import convert_segments
from .gender import predict_genders
from .voices import (
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
DURATION_MODEL_DIR = REPO_ROOT / "models" / "duration"


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

    # Voice conversion (plan §8 / E14).
    vc_model: Optional[str] = None
    vc_out_dir: Optional[Path] = None
    vc_errors: int = 0

    # Timing plan.
    timing_producer: Optional[str] = None
    timing_plan_json: Optional[Path] = None
    timing_summary: dict = field(default_factory=dict)

    # Final mix.
    mix_wav: Optional[Path] = None
    mix_mp4: Optional[Path] = None

    # Lip-sync (plan §8).
    lipsync_model: Optional[str] = None
    lipsync_mp4: Optional[Path] = None
    lipsync_error: Optional[str] = None
    # Face restoration over the render, when one was asked for. The
    # un-restored render stays on disk beside it so the two are comparable.
    restore_model: Optional[str] = None
    restore_error: Optional[str] = None

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
    vc_model: Optional[str] = None
    vc_out_dir: Optional[Path] = None
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
    vc_errors: int = 0
    lipsync_model: Optional[str] = None
    restore_model: Optional[str] = None
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
#: source speaker. These are the ones stage 9d can actually cast.
_CATALOG_ENGINES = {"bodhan": "indic-speak"}


def _default_voice_engine(voice_engine: Optional[str],
                          tts_model: Optional[str]) -> Optional[str]:
    """Give a stock-voice engine its catalog even when nobody asked.

    `--voice-engine` and `--tts` are separate flags, and the gap between
    them was a silent wrong output: `--tts bodhan` on its own skipped stage
    9d's casting entirely, every speaker got `voice=None`, and
    `bodhan_adapter._post_one` substituted its hardcoded `"voice": voice or
    "Amit"` — one Hindi male voice reading every line of every speaker, with
    the predicted genders computed and discarded. Nothing errored.

    There is no reason to run a catalog engine without its catalog, so the
    pairing is now implied. Passing `--voice-engine` explicitly still wins.
    """
    if voice_engine:
        return voice_engine
    implied = _CATALOG_ENGINES.get((tts_model or "").lower())
    if implied:
        log.info("voices: --tts %s implies --voice-engine %s; casting from "
                 "the catalog rather than leaving every speaker on the "
                 "adapter's single default voice", tts_model, implied)
    return implied


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
    # 9b code-mix / script normalisation
    translit_model: Optional[str] = None,
    termbase_path: Optional[Path] = None,
    # 9c duration-aware text adjustment
    duration_adjust: bool = False,
    duration_voice: str = "default",
    duration_safety_margin_s: float = 0.10,
    duration_cache_dir: Optional[Path] = None,
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
    tts_max_stretch: float = 0.10,
    stretch_backend: str = "auto",
    tts_apply_stretch: bool = True,
    # 9e engine-side speed control (method M1). Defaults are copied verbatim
    # from `synthesize_segments`, so leaving them alone is bit-identical to
    # not passing them at all.
    tts_speed_control: str = "off",                  # off|oneshot|closed_loop
    tts_speed_bounds: Tuple[float, float] = (0.85, 1.15),
    tts_speed_tolerance: float = 0.05,
    tts_speed_max_rounds: int = 2,
    # 9f voice conversion
    vc_model: Optional[str] = None,
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
    # 9h lip sync
    lipsync_model: Optional[str] = None,
    lipsync_num_gpus: Optional[int] = None,
    restore_model: Optional[str] = None,
    # Render only this (start_s, end_s) slice and splice it back into the
    # full timeline. None -- the default -- renders the whole video, exactly
    # as before. `backend.edits.lipsync_window` is what sets it.
    lipsync_window: Optional[Tuple[float, float]] = None,
    **adapter_kwargs,
) -> PostMTResult:
    """Run stages 9b..9h against one merged ASR+MT JSON.

    Every stage mutates `mt_json` in place and reads the previous stage's
    `extra.*` back out of it, so the only safe way to run two variants of this
    chain is to hand each one its own copy of the merged JSON and its own
    `out_root` — never to thread per-stage output paths through the chain.

    All outputs land under `out_root/{voices,tts,vc,mix,lipsync}`. The
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
    vc_dir = out_root / "vc"
    mix_dir = out_root / "mix"
    lipsync_dir = out_root / "lipsync"
    for d in (voices_dir, tts_dir, vc_dir, mix_dir, lipsync_dir):
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
                from .emotion import annotate_segments_with_emotion
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

    # 9d. Voice assignment per speaker (plan §8).
    voice_engine = _default_voice_engine(voice_engine, tts_model)
    speaker_voice_map: Dict[str, dict] = {}
    speaker_refs: Dict[str, Path] = {}
    if voice_engine or tts_model or vc_model:
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

        # Gender prediction from reference clips (ECAPA primary, F0 fallback).
        gender_map = predict_genders(speaker_refs, device=device or "cpu")
        if gender_overrides:
            for sid, g in gender_overrides.items():
                if gender_map.get(sid) != g:
                    log.info("voices: %s gender overridden %s -> %s (edit)",
                             sid, gender_map.get(sid), g)
                gender_map[sid] = g
        for spk in speakers:
            spk["predicted_gender"] = gender_map.get(spk["id"])

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
            # No stock-voice catalog in use. Since Edge was removed this is
            # the CLONING path only — IndicF5 or a VC model — where the
            # speaker's own reference clip carries the timbre and there is no
            # voice name to assign. `voice=None` is the correct answer here,
            # not a missing one.
            #
            # A stock-voice engine must never reach this branch: it would get
            # `voice=None` for every speaker and the adapter would fall back
            # to its own single default, reading the whole dub in one voice.
            # That is what `--tts bodhan` without `--voice-engine` used to do
            # — 45 voices in the catalog and `Amit` on every line — which is
            # why `_default_voice_engine` above now resolves it.
            for spk in sorted(speaker_refs):
                speaker_voice_map[spk] = {
                    "engine": tts_model or vc_model,
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
        # above is written ONLY when `voice_engine` is set; the Edge /
        # IndicF5 path builds `speaker_voice_map` in memory and left no
        # record at all, so nothing downstream could say which voice spoke
        # for which speaker. A separate filename, so `voice_assignments_json`
        # keeps its existing meaning and no manifest key changes.
        # Gender is computed here from the reference clips and then used only
        # to pick a voice — it was never written down, so the UI could not
        # show it or offer same-gender voices without re-running ECAPA on
        # every request. Record it alongside the voice.
        for _sid, _info in speaker_voice_map.items():
            _g = gender_map.get(_sid)
            if _g and isinstance(_info, dict):
                _info.setdefault("gender", _g)

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

    # 9f. Voice conversion (plan §8 / E14) — chains after TTS.
    if vc_model:
        if not out.tts_model:
            log.warning("VC requested but TTS was skipped; nothing to convert")
        elif not speaker_refs:
            log.warning("VC requested but no speaker references extracted")
        else:
            log.info("=== stage 9f: voice conversion (%s) ===", vc_model)
            vc_out = vc_dir / f"{video.stem}.{vc_model}"
            convert_segments(
                mt_json,
                vc_model=vc_model,
                language=target_language,
                reference_audio=speaker_refs,
                out_dir=vc_out,
                vc_kwargs=_pick(adapter_kwargs,
                                   ("repo_path", "checkpoint_dir",
                                    "language", "task")),
            )
            out.vc_model = vc_model
            out.vc_out_dir = vc_out
            vc_meta = json.loads(mt_json.read_text()).get("extra", {}).get("vc", {})
            out.vc_errors = int(vc_meta.get("errors", 0))

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
            from . import timing as _timing
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
        if not out.tts_model:
            log.warning("mix requested but TTS was skipped; nothing to mix")
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

    # 9h. Lip sync (plan §8). Needs the dubbed audio from the mix stage.
    if lipsync_model:
        if not out.mix_wav:
            log.warning("lipsync requested but mix was skipped; nothing to drive")
        else:
            log.info("=== stage 9h: lip sync (%s) ===", lipsync_model)
            try:
                ls = run_lipsync(
                    lipsync_merged_json,
                    model=lipsync_model,
                    source_video=video,
                    driving_audio=out.mix_wav,
                    out_dir=lipsync_dir,
                    pre_muxed_video=out.mix_mp4,
                    num_gpus=lipsync_num_gpus,
                    window=lipsync_window,
                    restore=restore_model,
                    restore_kwargs=_pick(adapter_kwargs,
                        ("fidelity", "upscale", "detection_model",
                         "only_center_face", "face_upsample", "bg_upsampler",
                         "keep_frames")),
                    lipsync_kwargs=_pick(adapter_kwargs,
                        ("repo_path", "checkpoint", "python_bin",
                         "unet_config_path", "unet_ckpt_path",
                         "version", "batch_size", "fps", "bbox_shift",
                         "pads", "resize_factor", "wav2lip_batch_size",
                         "face_det_batch_size", "inference_steps",
                         "guidance_scale", "seed",
                         "model_arch", "lipgan_batch_size")),
                )
                out.lipsync_model = ls.model
                out.lipsync_mp4 = ls.out_video
                out.lipsync_error = ls.error
                rinfo = ls.extra.get("restore")
                if rinfo:
                    out.restore_model = rinfo.get("model_name")
                    out.restore_error = rinfo.get("error") or rinfo.get("warning")
            except Exception as e:
                log.warning("lipsync failed: %s", e)
                out.lipsync_error = str(e)

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
    # ASR
    asr_model: Optional[str] = None,              # None → skip ASR
    asr_language: Optional[str] = None,
    asr_full_audio: bool = False,                 # True → transcribe raw 16k
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
    # adapter that never heard of them (bodhan, nllb) is unaffected.
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
    # Voice assignment per speaker (plan §8).
    voice_engine: Optional[str] = None,           # "indic-speak" | "indicf5"
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
    tts_max_stretch: float = 0.10,
    stretch_backend: str = "auto",
    tts_apply_stretch: bool = True,
    # Engine-side speed control (method M1). Forwarded straight through to
    # `synthesize_segments`; "off" is the historical behaviour.
    tts_speed_control: str = "off",               # off|oneshot|closed_loop
    tts_speed_bounds: Tuple[float, float] = (0.85, 1.15),
    tts_speed_tolerance: float = 0.05,
    tts_speed_max_rounds: int = 2,
    # Voice conversion.
    vc_model: Optional[str] = None,               # None → skip VC
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
    # Lip sync (requires mix).
    lipsync_model: Optional[str] = None,           # wav2lip | musetalk | latentsync
    lipsync_num_gpus: Optional[int] = None,       # None → use every visible CUDA
    restore_model: Optional[str] = None,          # codeformer — post-lipsync
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
            "vc":              vc_model,
            "voice_engine":    voice_engine,
            "lipsync":         lipsync_model,
            "restore":         restore_model,
            "stem_model":      stem_model,
            "skip_stems":      skip_stems,
            "timing_producer": timing_producer,
            "variant_tag":     variant_tag,
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
    vc_dir = job.stage("vc")
    mix_dir = job.stage("mix")
    lipsync_dir = job.stage("lipsync")

    exp_root = Path(exp_root) if exp_root is not None else job.stage("exp")

    #: Filled by `resolve_input` for a URL input; stays empty for a local
    #: file, which is exactly when the classifier falls through to its LLM
    #: tier. Also written to the manifest so a register choice is traceable.
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
        from backend.faces import preimport_face_stack
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

        # 6. ASD + Hungarian speaker↔face assignment (optional, needs tracks).
        assignments: Optional[List] = None
        if asd_model:
            if tracks is None:
                log.warning("asd requested but no face tracks produced; skipping")
            else:
                log.info("=== stage 6: ASD (%s) + Hungarian assignment ===", asd_model)
                asd_result = score_asd(
                    video, dia_input, tracks,
                    model=asd_model, device=device,
                    **_pick(adapter_kwargs, ("repo_path", "checkpoint",
                                              "speaker_encoder_checkpoint",
                                              "crop_size", "target_fps")),
                )
                assignments = assign_faces_to_turns(
                    diar_result.turns, tracks, asd_result,
                    on_screen_threshold=asd_on_screen_threshold,
                )
                assignments_json = asd_dir / f"{video.stem}.{asd_model}.assignments.json"
                assignments_json.write_text(
                    json.dumps(assignments_to_jsonable(assignments), indent=2)
                )
                result.asd_model = asd_model
                result.assignments_json = assignments_json
                result.num_on_screen = sum(1 for a in assignments if a.on_screen)

        # 7. ASR (optional).
        if asr_model:
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
            # Fold speaker labels + face assignments into each segment.
            asd_by_seg = {a.seg_id: a for a in (assignments or [])}
            payload = asr_result.to_jsonable()
            for seg in payload["segments"]:
                a = asd_by_seg.get(seg["id"])
                if a is not None:
                    seg["face_track"] = a.face_track
                    seg["asd_score"] = (a.asd_score if a.asd_score != float("-inf") else None)
                    seg["on_screen"] = a.on_screen
            asr_json = asr_dir / f"{video.stem}.{asr_model}.asr.json"
            asr_json.write_text(json.dumps(payload, indent=2))
            asr_txt = asr_dir / f"{video.stem}.{asr_model}.transcript.txt"
            asr_txt.write_text(
                "\n".join(s["text"].strip() for s in payload["segments"] if s.get("text")) + "\n",
                encoding="utf-8",
            )
            result.asr_model = asr_model
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
                                              "model_id", "num_beams",
                                              "batch_size", "dtype",
                                              "max_new_tokens")),
                )
                mt_json = mt_dir / f"{video.stem}.{mt_model}.{mt_target_language}.mt.json"
                mt_txt = mt_dir / f"{video.stem}.{mt_model}.{mt_target_language}.transcript.txt"
                write_mt_outputs(mt_json, mt_txt, asr_result, mt_result)
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

    # `--tts-style auto` — pick the programme's register from the uploader's
    # own metadata, falling back to an LLM read of the transcript. Resolved
    # here because this is the first point where BOTH the metadata (from the
    # download) and a transcript (from ASR/MT) exist. A None answer means
    # "no global style", i.e. the voice's neutral reading, which is also what
    # happens when the video came from a local file with no metadata.
    if tts_style == "auto":
        from .classify import classify_register, transcript_sample
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
        result.extra.setdefault("classify", {}).update(
            {"register": tts_style, "tier": tier,
             "categories": source_metadata.get("categories")})

    # 9b..9h — code-mix → duration → voices → TTS → VC → mix → lip-sync.
    post = run_post_mt_stages(
        result.mt_json,
        source_video=video,
        me_stem=me_stem,
        reference_audio_src=dia_input,
        out_root=post_mt_out_root,
        target_language=mt_target_language,
        target_script=mt_target_script,
        source_language=asr_language,
        device=device,
        turns=turns,
        manifest=result.manifest,
        translit_model=translit_model,
        termbase_path=termbase_path,
        duration_adjust=duration_adjust,
        duration_voice=duration_voice,
        duration_safety_margin_s=duration_safety_margin_s,
        duration_cache_dir=DURATION_MODEL_DIR,
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
        vc_model=vc_model,
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
        lipsync_model=lipsync_model,
        lipsync_num_gpus=lipsync_num_gpus,
        # Dropping this line meant `run_post_mt_stages` fell back to its own
        # `restore_model=None`, so `run_lipsync`'s `if restore and ...` never
        # fired -- while line ~1292 wrote `"restore": restore_model` into the
        # manifest regardless. Every standard/best UI job on disk claims a
        # restoration pass that never ran, with no error anywhere. This call
        # forwards ~60 kwargs by hand and restore_model was the only
        # lip-sync-related one missing; `tests/test_ui_stage_wiring.py` now
        # checks the whole set rather than this one name.
        restore_model=restore_model,
        **adapter_kwargs,
    )
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
    result.vc_model = post.vc_model
    result.vc_out_dir = post.vc_out_dir
    result.vc_errors = post.vc_errors
    result.timing_producer = post.timing_producer
    result.timing_plan_json = post.timing_plan_json
    result.timing_summary = post.timing_summary
    result.mix_wav = post.mix_wav
    result.mix_mp4 = post.mix_mp4
    result.lipsync_model = post.lipsync_model
    result.lipsync_mp4 = post.lipsync_mp4
    result.lipsync_error = post.lipsync_error

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
    `functools.wraps` keeps the real signature visible to
    `backend.experiment`, which validates variant knobs against it.
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
