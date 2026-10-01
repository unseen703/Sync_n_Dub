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

Each post-diarization stage is behind a flag; the baseline pipeline still
stops at diarization for anyone who only wants that.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .asd import assign_faces_to_turns, assignments_to_jsonable, score_asd
from .asr import transcribe as asr_transcribe
from .diarize import diarize as diarize_stage
from .downloader import resolve_input
from .faces import detect_and_track
from .media_io import extract_audio
from .stem_separation import separate_stems

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


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


def run_pipeline(
    input_: str,
    *,
    diarization_model: str = "pyannote",
    data_dir: Path = DEFAULT_DATA_DIR,
    device: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    skip_stems: bool = False,
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
    stem_model: str = "htdemucs",                 # htdemucs | htdemucs_ft |
                                                  # mdx_extra | mdx_extra_q
    **adapter_kwargs,
) -> PipelineResult:
    """Run download → extract → separate → diarize → (faces → ASD → ASR).

    `adapter_kwargs` is a flat bag forwarded to every registry call; each
    adapter picks what it accepts and drops the rest (with a warning). So
    passing `hf_token=…, checkpoint=…, render_mode=…` at once is safe.
    """
    data_dir = Path(data_dir)
    videos_dir = data_dir / "videos"
    audio_dir = data_dir / "audio"
    stems_dir = data_dir / "stems"
    dia_dir = data_dir / "diarization"
    faces_dir = data_dir / "faces"
    asd_dir = data_dir / "asd"
    asr_dir = data_dir / "asr"
    for d in (videos_dir, audio_dir, stems_dir, dia_dir, faces_dir, asd_dir, asr_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Resolve input (download if URL)
    log.info("=== stage 1: download / resolve ===")
    video = resolve_input(input_, download_dir=videos_dir)

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
                                      "render_mode", "timeout_s", "max_retries",
                                      "checkpoint", "compute_type",
                                      "batch_size", "language", "align",
                                      "condition_on_previous_text",
                                      "no_speech_threshold",
                                      "vad_onset", "vad_offset")),
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
        result.asr_model = asr_model
        result.asr_language = asr_result.language
        result.asr_json = asr_json

    result.to_json(result.manifest)
    log.info(
        "pipeline done: %d turns, %d speakers, %d face tracks, %d on-screen → %s",
        result.num_turns, result.num_speakers,
        result.num_face_tracks, result.num_on_screen,
        rttm_path,
    )
    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pick(d: Dict[str, Any], keys) -> dict:
    """Return a sub-dict with only `keys` present in `d`."""
    return {k: d[k] for k in keys if k in d and d[k] is not None}
