"""CLI entry point:

    python -m backend.cli --input <url_or_path> \
        [--model pyannote|diarizen] \
        [--face-detector retinaface|mediapipe --face-tracker bytetrack] \
        [--asd talknet|ts_talknet] \
        [--asr bodhan|whisperx|indicconformer --asr-language hi]
"""
from __future__ import annotations

import argparse
import logging
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv

from models.asd import available_models as _asd_models
from models.asr import available_models as _asr_models
from models.diarization import available_models as _diar_models
from models.faces.detect import available_detectors as _detectors
from models.faces.track import available_trackers as _trackers

from .pipeline import DEFAULT_DATA_DIR, REPO_ROOT, run_pipeline


def _check_env_permissions() -> None:
    """Refuse to start if `.env` is group- or world-readable.

    On a shared box (e.g. /data1/), a mode-644 .env leaks whatever HF /
    Bodhan token lives in it to every account on the host.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    mode = env_path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise SystemExit(
            f"Refusing to start: {env_path} is group/world accessible "
            f"(mode {oct(mode & 0o777)}) and likely contains a secret token. "
            f"Fix with: chmod 600 {env_path}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Dubbing Studio pipeline — download → extract → separate → "
            "diarize → (faces → ASD → ASR)."
        ),
    )
    ap.add_argument(
        "--input", "-i", required=True,
        help="YouTube URL (or any yt-dlp-supported URL) or path to a local video file.",
    )
    ap.add_argument(
        "--model", "-m", default="pyannote", choices=_diar_models(),
        help="Diarization model.",
    )
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), type=Path)
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--skip-stems", action="store_true",
                    help="Skip stem separation; diarize the raw audio instead.")
    ap.add_argument("--stem-model", default="htdemucs",
                    choices=("htdemucs", "htdemucs_ft", "mdx_extra",
                             "mdx_extra_q", "htdemucs_6s"),
                    help="Demucs model. htdemucs_ft gives cleaner vocals on "
                         "music beds (~4x slower). Default: htdemucs.")

    # Face stack --------------------------------------------------------
    ap.add_argument("--face-detector", default=None, choices=_detectors(),
                    help="Enable face detection with this model.")
    ap.add_argument("--face-tracker", default="bytetrack", choices=_trackers())
    ap.add_argument("--face-every-n", type=int, default=1,
                    help="Detect on every nth frame (tracker interpolates). "
                         "Set 2 for lecture footage to roughly halve detect cost.")
    ap.add_argument("--face-min-score", type=float, default=0.5)

    # ASD + assignment --------------------------------------------------
    ap.add_argument("--asd", default=None, choices=_asd_models(),
                    help="Active-speaker-detection model. Requires --face-detector.")
    ap.add_argument("--asd-on-screen-threshold", type=float, default=0.3,
                    help="ASD mean-score cutoff below which a turn is tagged off-screen.")

    # ASR ---------------------------------------------------------------
    ap.add_argument("--asr", default=None, choices=_asr_models(),
                    help="Automatic-speech-recognition model.")
    ap.add_argument("--asr-language", default=None,
                    help="ISO-639 code (en, hi, ta, …). None = detect.")
    ap.add_argument("--asr-full-audio", action="store_true",
                    help="Transcribe the entire 16 kHz mix end-to-end instead "
                         "of re-slicing to diarization turns. Keeps intro/outro "
                         "and over-music speech that would otherwise be dropped.")
    ap.add_argument("--render-mode", default=None,
                    choices=("native", "mixed", "roman"),
                    help="Bodhan ASR render mode (plan §6).")

    # Adapter-specific kwargs — forwarded via **adapter_kwargs. Each adapter
    # drops what it does not recognise, so a wide union is safe.
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (pyannote / diarizen adapters).")
    ap.add_argument("--checkpoint", default=None,
                    help="Override checkpoint for whichever adapter uses one.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Batch size (indicconformer / whisperx).")
    ap.add_argument("--allow-insecure-env", action="store_true",
                    help="Skip the .env file-permission check on startup.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_insecure_env:
        _check_env_permissions()
    load_dotenv()

    # pyannote.audio 4.x torchcodec needs system ffmpeg libs; shim via PyAV.
    if args.model == "pyannote":
        from models.diarization.pyannote_adapter import (
            ensure_ffmpeg_libs_on_ld_path_or_reexec,
        )
        ensure_ffmpeg_libs_on_ld_path_or_reexec()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    )

    if args.asd and not args.face_detector:
        raise SystemExit("--asd requires --face-detector (nothing to track).")

    adapter_kwargs = {
        k: v for k, v in (
            ("hf_token", args.hf_token),
            ("checkpoint", args.checkpoint),
            ("batch_size", args.batch_size),
            ("render_mode", args.render_mode),
        ) if v is not None
    }

    result = run_pipeline(
        args.input,
        diarization_model=args.model,
        data_dir=args.data_dir,
        device=args.device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        skip_stems=args.skip_stems,
        stem_model=args.stem_model,
        face_detector=args.face_detector,
        face_tracker=args.face_tracker,
        face_every_n=args.face_every_n,
        face_min_score=args.face_min_score,
        asd_model=args.asd,
        asd_on_screen_threshold=args.asd_on_screen_threshold,
        asr_model=args.asr,
        asr_language=args.asr_language,
        asr_full_audio=args.asr_full_audio,
        **adapter_kwargs,
    )

    print("\n=== PIPELINE RESULT ===")
    print(f"video          : {result.video}")
    print(f"audio (16k)    : {result.audio_16k}")
    print(f"dialogue stem  : {result.dialogue_stem}")
    print(f"M&E stem       : {result.me_stem}")
    print(f"rttm           : {result.rttm}")
    print(f"turns          : {result.num_turns}")
    print(f"speakers       : {result.num_speakers}")
    print(f"diar model     : {result.diarization_model}")
    if result.face_detector:
        print(f"face detector  : {result.face_detector} + {result.face_tracker}")
        print(f"face tracks    : {result.num_face_tracks} ({result.face_tracks_npz})")
    if result.asd_model:
        print(f"asd model      : {result.asd_model}")
        print(f"on-screen turns: {result.num_on_screen}/{result.num_turns}")
        print(f"assignments    : {result.assignments_json}")
    if result.asr_model:
        print(f"asr model      : {result.asr_model} (lang={result.asr_language})")
        print(f"asr json       : {result.asr_json}")
    print(f"manifest       : {result.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
