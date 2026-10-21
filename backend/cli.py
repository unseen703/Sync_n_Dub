"""CLI entry point:

    python -m backend.cli --input <url_or_path> \
        [--model pyannote|diarizen] \
        [--face-detector retinaface|mediapipe --face-tracker bytetrack] \
        [--asd talknet|ts_talknet] \
        [--asr bodhan|whisperx|indicconformer|indictranscribe_core|indictranscribe_flex --asr-language hi] \
        [--align mms_fa|whisperx] \
        [--mt bodhan|nllb --mt-target-language hi
              --mt-target-script codemix|native|roman --mt-model-id <hf repo id>]
        [--timing-method m2|m4|m7 [--timing-tempo-cap 1.50]]
"""
from __future__ import annotations

import argparse
import logging
from collections import namedtuple
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv

from models.align import available_models as _align_models
from models.asd import available_models as _asd_models
from models.asr import available_models as _asr_models
from models.diarization import available_models as _diar_models
from models.faces.detect import available_detectors as _detectors
from models.faces.track import available_trackers as _trackers
from models.lipsync import available_models as _lipsync_models


#: Bodhan is the house implementation for the three hosted stages, so a bare
#: `--asr` / `--mt` / `--tts` picks it. This is the DEFAULT IMPLEMENTATION, not
#: a default-on stage: the flag being absent still means "skip", because the
#: whole dependency chain in `_validate` keys off None, and defaulting these
#: to a model would force every run to supply --mt-target-language.
#: Falls back to the registry's first entry if bodhan is not installed.
_PREFERRED_IMPL = "bodhan"


def _speed_bounds(s: str) -> tuple:
    """`"0.85,1.15"` -> `(0.85, 1.15)`, validated.

    argparse cannot express a bounded pair, and a silently reversed or
    inverted range would clamp every rate to one end without complaining.
    """
    try:
        lo, hi = (float(x) for x in str(s).split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--tts-speed-bounds wants 'lo,hi' (e.g. 0.85,1.15), got {s!r}")
    if not (0 < lo <= hi):
        raise argparse.ArgumentTypeError(
            f"--tts-speed-bounds needs 0 < lo <= hi, got {lo},{hi}")
    return (lo, hi)


def _speed_control_modes():
    from backend.tts import SPEED_CONTROL_MODES
    return SPEED_CONTROL_MODES


def _emotion_models():
    try:
        from models.emotion import available_models
        return available_models()
    except ImportError:
        return []


def _default_impl(stage: str) -> str:
    avail = {"asr": _asr_models, "mt": _mt_models, "tts": _tts_models}[stage]()
    return _PREFERRED_IMPL if _PREFERRED_IMPL in avail else (avail[0] if avail else None)


def _tts_styles():
    """Styles the Bodhan TTS engine accepts, plus `auto`.

    `auto` is not a Bodhan value — it asks `backend.classify` to pick the
    register from the video's own metadata (and, failing that, an LLM read
    of the transcript). It is resolved to a real style, or to None, before
    anything is synthesized.
    """
    try:
        from models.tts.bodhan_adapter import INDIC_SPEAK_STYLES
        return ("auto",) + tuple(INDIC_SPEAK_STYLES)
    except ImportError:
        return ("auto",)
from models.mt import available_models as _mt_models
from models.translit import available_models as _translit_models
from models.tts import available_models as _tts_models
from models.vc import available_models as _vc_models

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


# Characters that only ever appear in a *list* of inputs, never in one path
# or URL we would accept. Catching them turns "the pipeline silently ran on
# one of my three files" into an error at parse time.
_MULTI_INPUT_HINTS = (",", "\n", "\t", "|", ";")

#: Shell globs that reached us unexpanded (quoted, or no match) — each stands
#: for a set of files, not one.
_GLOB_CHARS = ("*", "?", "[")


class _SingleInput(argparse.Action):
    """Accept exactly one video/audio input per run.

    The pipeline is built around a single input: one job folder, one video
    stem, one set of stage outputs. Several inputs would quietly collide in
    that one folder, so every way of naming more than one is rejected here
    rather than halfway through stage 9.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(
                f"{option_string} given more than once. The pipeline runs on "
                f"exactly one video/audio input per invocation — run it once "
                f"per file, each gets its own job folder under data/jobs/."
            )
        text = str(value).strip()
        if not text:
            parser.error(f"{option_string} is empty.")
        for ch in _MULTI_INPUT_HINTS:
            if ch in text:
                parser.error(
                    f"{option_string} looks like a list of inputs "
                    f"(contains {ch!r}). Pass exactly one video/audio file "
                    f"or URL; run the pipeline once per input."
                )
        if not text.startswith(("http://", "https://")):
            if any(c in text for c in _GLOB_CHARS):
                parser.error(
                    f"{option_string} contains a shell glob ({text!r}) that "
                    f"matched nothing or was quoted. Pass one concrete file."
                )
            path = Path(text).expanduser()
            if path.is_dir():
                parser.error(
                    f"{option_string} is a directory ({path}). Pass a single "
                    f"video/audio file, not a folder of them."
                )
            if not path.exists():
                parser.error(f"{option_string} not found: {path}")
        elif "list=" in text or "/playlist" in text:
            parser.error(
                f"{option_string} is a playlist URL. Pass the URL of one "
                f"video; run the pipeline once per video."
            )
        setattr(namespace, self.dest, text)


def _reject_extra_inputs(argv: list[str]) -> None:
    """Catch bare paths left over from an unquoted glob.

    `--input *.mp4` expands before argparse sees it, so the first file binds
    to --input and the rest arrive as positionals. argparse would reject them
    with "unrecognized arguments"; this says what actually went wrong.
    """
    media = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4a", ".wav", ".mp3", ".flac")
    # Only true positionals count. A value that follows an option belongs to
    # that option, so `--input A --input B` is the action's repeat error, not
    # this one — otherwise it would be blamed on a glob it never saw.
    strays = [a for i, a in enumerate(argv)
              if not a.startswith("-")
              and not (i and argv[i - 1].startswith("-"))
              and a.lower().endswith(media)]
    if strays:
        raise SystemExit(
            f"Refusing to start: {len(strays)} extra media file(s) passed "
            f"outside --input ({', '.join(strays[:3])}"
            f"{'…' if len(strays) > 3 else ''}). The pipeline runs on exactly "
            f"one input per invocation — most likely an unquoted glob "
            f"expanded. Quote it, or run the pipeline once per file."
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    _reject_extra_inputs(list(argv if argv is not None else sys.argv[1:]))
    ap = argparse.ArgumentParser(
        description=(
            "Dubbing Studio pipeline — download → extract → separate → "
            "diarize → (faces → ASD → ASR)."
        ),
    )
    ap.add_argument(
        "--input", "-i", required=True, action=_SingleInput, default=None,
        help="YouTube URL (or any yt-dlp-supported URL) or path to a local "
             "video file. Exactly one input per run — each run becomes its "
             "own job folder under data/jobs/.",
    )
    ap.add_argument(
        "--job-id", default=None,
        help="Name for this run's folder under data/jobs/ "
             "(default: <timestamp>__<input>__<config>).",
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

    # Resegmentation ----------------------------------------------------
    ap.add_argument("--resegment-turns", action="store_true",
                    help="Refine diarization turns: split long monologues at "
                         "silences, merge tiny turns, cap duration.")
    ap.add_argument("--resegment-max-duration", type=float, default=15.0,
                    help="Hard cap on segment duration (seconds).")
    ap.add_argument("--refine-speaker-labels", action="store_true",
                    help="Stage 4c: split diarization labels that hold more "
                         "than one person, by ECAPA similarity between turns. "
                         "Each label gets ONE voice downstream, so an impure "
                         "label reads several people in one voice.")
    ap.add_argument("--speaker-purity-floor", type=float, default=0.35,
                    help="Mean within-label cosine below which a label is "
                         "split (default 0.35; measured single-speaker floor).")
    ap.add_argument("--speaker-min-speech-s", type=float, default=4.0,
                    help="A split piece with less speech than this is folded "
                         "back into the sibling it most resembles.")

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
    ap.add_argument("--asr", nargs="?", const=_default_impl("asr"),
                    default=None, choices=_asr_models(),
                    help="Automatic-speech-recognition model. Bare --asr "
                         f"selects {_default_impl('asr')!r}; omit the flag "
                         "entirely to skip the stage.")
    ap.add_argument("--asr-language", default="en",
                    help="ISO-639 code of what is SPOKEN in the source "
                         "(en, hi, ta, …). Default 'en' — the dubbing "
                         "direction here is English -> Indic. Pass "
                         "'auto' to let the ASR detect it instead, which "
                         "it does per diarization turn and therefore "
                         "unreliably on short ones.")
    ap.add_argument("--asr-full-audio", action="store_true",
                    help="Transcribe the entire 16 kHz mix end-to-end instead "
                         "of re-slicing to diarization turns. Keeps intro/outro "
                         "and over-music speech that would otherwise be dropped.")
    ap.add_argument("--no-captions-transcript", action="store_true",
                    help="Always run ASR, even when the source has a "
                         "human-authored caption track. By default the "
                         "pipeline uses the uploader's own published "
                         "transcript for the ASR language when one exists, "
                         "which skips stage 7's per-turn API calls entirely. "
                         "Every failure mode (wrong language, auto-only, "
                         "rolling duplication, thin coverage) already falls "
                         "back to ASR on its own.")
    ap.add_argument("--captions-allow-auto", action="store_true",
                    help="Also accept YouTube's AUTO-generated captions as "
                         "the transcript. Off by default: those are "
                         "YouTube's own ASR output, so using them swaps the "
                         "engine chosen here for an unconfigurable one -- on "
                         "Indic-accented English that is usually a downgrade, "
                         "not a saving. Only manual tracks are used unless "
                         "this is passed.")
    ap.add_argument("--render-mode", default=None,
                    choices=("native", "mixed", "roman"),
                    help="Bodhan ASR render mode (plan §6).")

    # Forced alignment ---------------------------------------------------
    ap.add_argument("--align", default=None, choices=_align_models(),
                    help="Forced-alignment model. Requires --asr.")

    # Machine translation ------------------------------------------------
    ap.add_argument("--mt", nargs="?", const=_default_impl("mt"),
                    default=None, choices=_mt_models(),
                    help="Machine-translation model. Requires --asr. Bare "
                         f"--mt selects {_default_impl('mt')!r}.")
    ap.add_argument("--mt-target-language", default=None,
                    help="Target language code (e.g. hi, ta, bn). Required with --mt.")
    ap.add_argument("--mt-target-script", default="codemix",
                    choices=("native", "roman", "codemix"),
                    help="Target script for Indic languages (default: codemix). "
                         "codemix keeps English terms in Latin inside the "
                         "target-script sentence, which matches how this "
                         "material is spoken and stays consistent across "
                         "segments; native re-scripts, but Bodhan leaves Latin "
                         "remnants at random when it does.")
    ap.add_argument("--mt-model-id", default=None,
                    help="HuggingFace model id to force (nllb). "
                         "Default: adapter picks (NLLB-600M-distilled).")
    ap.add_argument("--mt-num-beams", type=int, default=None,
                    help="Beam count for local MT models (nllb).")

    # Code-mix / script normalisation -----------------------------------
    ap.add_argument("--translit", default=None, choices=_translit_models(),
                    help="Transliteration backend for the code-mix pass.")
    ap.add_argument("--termbase", default=None, type=Path,
                    help="Two-column TSV of source→replacement terms (plan §6).")

    # Duration-aware text adjustment ------------------------------------
    ap.add_argument("--duration-adjust", action="store_true",
                    help="Rank MT candidates and plan WSOLA stretch per segment.")
    ap.add_argument("--duration-voice", default="default",
                    help="Which TTS voice the fitted duration model is for.")
    ap.add_argument("--duration-safety-margin", type=float, default=0.10,
                    help="Seconds trimmed from each turn's budget.")

    # Voice assignment per speaker --------------------------------------
    ap.add_argument("--voice-engine", default=None,
                    choices=("indic-speak", "indicf5"),
                    help="TTS engine the voice catalog is drawn from.")
    ap.add_argument("--voice-catalog", default=None, type=Path,
                    help="Override JSON of voice rows (overrides defaults).")
    ap.add_argument("--allow-voice-reuse", action="store_true",
                    help="Permit the same voice on multiple speakers.")

    # TTS ---------------------------------------------------------------
    ap.add_argument("--tts", nargs="?", const=_default_impl("tts"),
                    default=None, choices=_tts_models(),
                    help="Text-to-speech model. Requires --mt. Bare --tts "
                         f"selects {_default_impl('tts')!r}.")
    ap.add_argument("--tts-style", default=None, choices=_tts_styles(),
                    metavar="STYLE",
                    help="Delivery style for the TTS engine (bodhan "
                         "indic-speak only; rides in instructions.style). "
                         "Omit for the voice's neutral reading. "
                         "'educational lecture' suits lecture material.")
    ap.add_argument("--emotion", default=None, choices=_emotion_models(),
                    help="Per-segment delivery style from the SOURCE "
                         "speaker's emotion (stage 9b2). Overrides "
                         "--tts-style on the segments it is confident "
                         "about. Requires --tts.")
    ap.add_argument("--emotion-min-confidence", type=float, default=0.60,
                    help="Below this the segment keeps the global style "
                         "(default: 0.60).")
    ap.add_argument("--emotion-min-run", type=int, default=2,
                    help="A label must hold for this many consecutive "
                         "segments before it is applied — suppresses "
                         "single-segment classifier noise, which is audible "
                         "as style jitter (default: 2).")
    ap.add_argument("--tts-speed-control", default="off",
                    choices=_speed_control_modes(),
                    help="Ask the ENGINE to speak faster/slower instead of "
                         "stretching the rendered clip. 'oneshot' predicts "
                         "the length from the text and sets the rate once; "
                         "'closed_loop' synthesizes, measures and re-asks "
                         "(each round is a real TTS call). Needs an engine "
                         "with supports_rate — today Bodhan only; it no-ops "
                         "with a warning elsewhere. Default: off (payload "
                         "byte-identical to not passing it).")
    ap.add_argument("--tts-speed-bounds", type=_speed_bounds,
                    default=(0.85, 1.15), metavar="LO,HI",
                    help="Clamp for the requested rate (default: 0.85,1.15). "
                         "Wider ranges get audibly synthetic fast.")
    ap.add_argument("--tts-speed-tolerance", type=float, default=0.05,
                    help="closed_loop stops once within this fraction of the "
                         "target duration (default: 0.05).")
    ap.add_argument("--tts-speed-max-rounds", type=int, default=2,
                    help="Extra closed_loop synthesis rounds per segment "
                         "(default: 2). Each one is a billed TTS call; the "
                         "count lands in extra.tts.extra_synthesis_rounds.")
    ap.add_argument("--tts-sample-rate", type=int, default=24000)
    ap.add_argument("--tts-max-stretch", type=float, default=0.20,
                    help="Time-stretch cap, as a +/- FRACTION of natural "
                         "pace: 0.20 clamps each clip to 0.8x-1.2x speed "
                         "(backend/stretch.py applies "
                         "`max(1-cap, min(1+cap, ratio))`). Raised from "
                         "0.10 on 2026-10-20 by request.")
    ap.add_argument("--stretch-backend", default="auto",
                    choices=("auto", "rubberband", "ffmpeg_atempo",
                             "librosa", "linear"),
                    help="Time-stretch implementation. 'auto' picks the best "
                         "available (rubberband here). 'linear' is a "
                         "pitch-SHIFTING resample kept only as an ablation "
                         "baseline -- it is what the old silent fallback did.")
    ap.add_argument("--no-tts-stretch", action="store_true",
                    help="Synthesise raw clips only, skipping the per-segment "
                         "stretch, so a downstream timing plan owns all "
                         "stretching instead of compounding with it.")

    # Voice conversion --------------------------------------------------
    ap.add_argument("--vc", default=None, choices=_vc_models(),
                    help="Voice-conversion model. Requires --tts.")

    # Timing method — top 3 from the eight-method sweep -------------------
    ap.add_argument("--speed-mode", default="global",
                    choices=("global", "segment"),
                    help="How the dub is re-paced to fit the source timeline. "
                         "`global` (DEFAULT, unchanged behaviour): every clip "
                         "plays at its natural pace, placed shift-forward, "
                         "then ONE atempo over the whole file so the dub "
                         "lands on the source duration. `segment`: each clip "
                         "is fitted to its own slot -- compressed when the "
                         "dub line is longer, STRETCHED OUT when it is "
                         "shorter, so a short line does not leave a pause in "
                         "the middle of its slot. Bounded by --speed-cap; "
                         "whatever the bound could not absorb is reported as "
                         "unfilled_slot_s in the timing plan.")
    ap.add_argument("--speed-cap", type=float, default=1.5,
                    help="With --speed-mode segment: the symmetric pace "
                         "bound, applied as max_tempo=cap and "
                         "min_tempo=1/cap. Default 1.5 -- the widest cap "
                         "measured (the m2 arm), and the only one wide "
                         "enough to matter, since Hindi TTS runs 1.21-1.65x "
                         "longer than the English it covers.")
    ap.add_argument("--speed-slot-margin", type=float, default=0.0,
                    help="With --speed-mode segment: seconds of slot left "
                         "deliberately empty at the end of each segment. "
                         "Default 0.0 -- any margin IS a pause, which is the "
                         "thing this mode exists to remove. Raise it if "
                         "clips are running into each other.")

    ap.add_argument("--timing-method", default=None,
                    choices=("m2", "m4", "m7"),
                    help="Timing-alignment method for the dubbed audio. "
                         "m2: post-TTS time stretching — naive_slot placement "
                         "with a wide rubberband cap (1.50x), best on overrun "
                         "and pause structure. "
                         "m4: word-anchored + forced alignment — clause-level "
                         "placement on source speech runs with TTS word timings "
                         "(M4+M5), most promising placement method. "
                         "m7: DP optimiser + forced alignment — two-level dynamic "
                         "programming over split points and tempo (M7+M5), only "
                         "statistically significant arm. "
                         "Default (omitted): global_tempo (one atempo for the "
                         "whole file).")

    # Per-method fine-tuning knobs
    ap.add_argument("--timing-tempo-cap", type=float, default=None,
                    help="Override the per-piece tempo cap for the chosen "
                         "timing method. Default: 1.50 (m2), 1.15 (m4/m7).")

    # Final mix ---------------------------------------------------------
    ap.add_argument("--mix", action="store_true",
                    help="Mix TTS/VC segments over M&E stem and mux onto video.")
    ap.add_argument("--mix-sample-rate", type=int, default=48000)
    ap.add_argument("--mix-tts-gain-db", type=float, default=3.0)
    ap.add_argument("--mix-me-gain-db", type=float, default=-3.0)
    ap.add_argument("--mix-strategy", default="global_tempo",
                    choices=("global_tempo", "no_tempo", "timing_plan"),
                    help="How the mixer resolves TTS overruns against the "
                         "source timeline. `global_tempo` (default) places "
                         "every clip at natural pace via shift-forward, then "
                         "applies ONE global atempo so the dub matches the "
                         "source-video length. `no_tempo` keeps every clip at "
                         "its natural pace with no stretching; the dub may be "
                         "longer than the video (mux will trim with -shortest). "
                         "`timing_plan` renders per-piece placement from a "
                         "timing plan (set automatically by --timing-method).")
    ap.add_argument("--mix-clip-source", default="auto",
                    choices=("auto", "raw", "stretched", "vc"),
                    help="Which per-segment clip the mix reads. 'auto' keeps "
                         "the historical vc_audio-or-tts_audio behaviour; "
                         "'raw' takes the unstretched clip, which is what you "
                         "want when the mix applies its own tempo.")
    ap.add_argument("--mix-tempo-cap", type=float, default=1.5,
                    help="Upper bound for the global atempo factor "
                         "(global_tempo strategy). Default 1.5x.")
    ap.add_argument("--no-mux", action="store_true",
                    help="With --mix: only write the dubbed WAV, skip MP4 mux.")

    # Lip sync ----------------------------------------------------------
    ap.add_argument("--lipsync", default=None, choices=_lipsync_models(),
                    help="Lip-sync model (lipgan | wav2lip | musetalk | latentsync). "
                         "Requires --mix.")
    ap.add_argument("--lipsync-batch-size", type=int, default=None,
                    help="MuseTalk / Wav2Lip / LipGAN batch size.")
    ap.add_argument("--lipsync-inference-steps", type=int, default=None,
                    help="LatentSync diffusion steps.")
    ap.add_argument("--lipsync-num-gpus", type=int, default=None,
                    help="Shard lip-sync across N GPUs (one adapter per GPU, "
                         "video+audio split in time, outputs concatenated). "
                         "DEFAULT: one GPU -- whichever has the most free "
                         "memory. Sharding is opt-in because each shard does "
                         "its own face-detection smoothing (the join can show "
                         "as a one-frame mouth jump) and both cards then run "
                         "near the ceiling, where another tenant OOMs a "
                         "shard. Clips shorter than N*15s fall back to one "
                         "GPU regardless.")

    # Ground-truth transcripts (for ASR/MT evaluation) -------------------
    ap.add_argument("--subtitle-langs", default=None,
                    help="Comma-separated ISO-639 codes (e.g. 'en,hi') to "
                         "download alongside the video for ASR/MT scoring. "
                         "Default: fetch the ASR source + MT target languages "
                         "automatically.")
    ap.add_argument("--test", action="store_true",
                    help="After the pipeline finishes, score ASR (WER/CER + "
                         "BLEU/chrF/ROUGE/METEOR) and MT (BLEU/chrF/ROUGE/"
                         "METEOR) against the fetched transcripts and fail "
                         "if no reference transcript was found for a stage.")
    ap.add_argument("--no-eval", action="store_true",
                    help="Skip transcript download and ASR/MT evaluation.")

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


# ---------------------------------------------------------------------------
# Speed / timing resolution
# ---------------------------------------------------------------------------

#: What `resolve_speed_setup` returns. Named so `backend.edit_agent` can
#: replay a run's timing setup instead of re-deriving it and getting the
#: producer's defaults — which would quietly turn a `--speed-mode segment`
#: run into a 1.10x-capped, 0.10 s-margin one on the first edit.
SpeedSetup = namedtuple("SpeedSetup", [
    "timing_producer", "timing_params", "tts_apply_stretch",
    "tts_force_align", "mix_strategy", "mix_tempo_cap", "mix_clip_source",
])


def resolve_speed_setup(args) -> "SpeedSetup":
    """Turn --speed-mode / --timing-method into the pipeline's timing knobs.

    Both the CLI and the edit agent's re-run path go through here, so an
    edited run re-paces itself exactly the way the original run did.
    """
    # -- Resolve --timing-method into pipeline parameters ------------------
    timing_producer = None
    timing_params: dict = {}
    tts_apply_stretch = not args.no_tts_stretch
    tts_force_align = False
    mix_strategy = args.mix_strategy
    mix_tempo_cap = args.mix_tempo_cap
    mix_clip_source = args.mix_clip_source

    # --speed-mode is the plain-language front end to the same machinery the
    # research arms use: `segment` IS naive_slot rendered through the timing
    # plan, with a zero margin and a symmetric cap. `global` leaves every
    # default exactly where it was.
    if args.speed_mode == "segment":
        if args.timing_method:
            raise SystemExit(
                "--speed-mode segment and --timing-method set the same "
                "knobs. Use one: --speed-mode for the two-option choice, "
                "--timing-method for the measured research arms (m2 is the "
                "same producer with a 0.10 s slot margin).")
        if args.speed_cap <= 1.0:
            raise SystemExit(
                f"--speed-cap must be > 1.0 (got {args.speed_cap}); it is a "
                f"symmetric bound, applied as max {args.speed_cap}x and min "
                f"1/{args.speed_cap}x.")
        timing_producer = "naive_slot"
        timing_params = {
            "max_tempo": args.speed_cap,
            "min_tempo": 1.0 / args.speed_cap,
            "safety_margin_s": args.speed_slot_margin,
            # Honour a per-segment target written by stage 9c or by
            # `backend.edits.set_speed`; without this they are silently
            # ignored in this mode.
            "use_target_duration": True,
        }
        # The TTS stage must NOT stretch as well, or every clip is re-timed
        # twice -- once into its slot here, once into the same slot there.
        tts_apply_stretch = False
        mix_strategy = "timing_plan"
        mix_clip_source = "raw"

    tm = args.timing_method
    if tm == "m2":
        timing_producer = "naive_slot"
        cap = args.timing_tempo_cap if args.timing_tempo_cap is not None else 1.50
        timing_params = {"max_tempo": cap, "min_tempo": 1.0 / cap}
        tts_apply_stretch = False
        mix_strategy = "timing_plan"
        mix_clip_source = "raw"
    elif tm == "m4":
        timing_producer = "word_anchored"
        cap = args.timing_tempo_cap if args.timing_tempo_cap is not None else 1.15
        timing_params = {
            "max_tempo": cap,
            "min_tempo": 0.90,
            "require_tts_words": True,
        }
        tts_apply_stretch = False
        tts_force_align = True
        mix_strategy = "timing_plan"
        mix_clip_source = "raw"
    elif tm == "m7":
        timing_producer = "dp"
        cap = args.timing_tempo_cap if args.timing_tempo_cap is not None else 1.15
        timing_params = {
            "max_tempo": cap,
            "min_tempo": 0.90,
            "lambda_duration": 1.0,
            "lambda_pause": 1.0,
            "lambda_speechrate": 0.5,
            "lambda_phoneme": 0.0,
        }
        tts_apply_stretch = False
        tts_force_align = True
        mix_strategy = "timing_plan"
        mix_clip_source = "raw"
    return SpeedSetup(timing_producer, timing_params, tts_apply_stretch,
                      tts_force_align, mix_strategy, mix_tempo_cap,
                      mix_clip_source)


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
    if args.align and not args.asr:
        raise SystemExit("--align requires --asr (nothing to align).")
    if args.mt and not args.asr:
        raise SystemExit("--mt requires --asr (nothing to translate).")
    if args.mt and not args.mt_target_language:
        raise SystemExit("--mt requires --mt-target-language (e.g. hi).")
    if args.translit and not args.mt:
        raise SystemExit("--translit requires --mt (nothing to re-script).")
    if args.duration_adjust and not args.mt:
        raise SystemExit("--duration-adjust requires --mt.")
    if args.tts and not args.mt:
        raise SystemExit("--tts requires --mt.")
    if args.vc and not args.tts:
        raise SystemExit("--vc requires --tts.")
    if args.emotion and not args.tts:
        raise SystemExit("--emotion sets a per-segment TTS style; "
                         "it requires --tts.")
    if args.mix and not args.tts:
        raise SystemExit("--mix requires --tts.")
    if args.lipsync and not args.mix:
        raise SystemExit("--lipsync requires --mix.")
    # Language-gate the ASR choice up-front: argparse's `choices` is built
    # before --asr-language is known, so we re-check here and refuse with a
    # suggestion rather than letting the adapter raise mid-pipeline.
    if args.asr and args.asr_language:
        compatible = _asr_models(language=args.asr_language)
        if args.asr not in compatible:
            raise SystemExit(
                f"--asr {args.asr!r} does not support --asr-language "
                f"{args.asr_language!r}. "
                f"Options for this language: {compatible or '(none)'}."
            )

    adapter_kwargs = {
        k: v for k, v in (
            ("hf_token", args.hf_token),
            ("checkpoint", args.checkpoint),
            ("batch_size", args.batch_size),
            ("render_mode", args.render_mode),
            ("model_id", args.mt_model_id),
            ("num_beams", args.mt_num_beams),
        ) if v is not None
    }
    # Lip-sync-specific knobs are forwarded via the same adapter_kwargs
    # bag; the adapters pick what they recognise.
    if args.lipsync_batch_size is not None:
        adapter_kwargs["batch_size"] = args.lipsync_batch_size
    if args.lipsync_inference_steps is not None:
        adapter_kwargs["inference_steps"] = args.lipsync_inference_steps
    mux_video = not args.no_mux

    if args.no_eval and args.test:
        raise SystemExit("--test and --no-eval are mutually exclusive.")
    if args.no_eval:
        subtitle_langs = []
    elif args.subtitle_langs:
        subtitle_langs = [s.strip() for s in args.subtitle_langs.split(",") if s.strip()]
    else:
        subtitle_langs = None   # pipeline auto-picks asr/mt languages
    if args.test and not (args.asr or args.mt):
        raise SystemExit("--test requires --asr and/or --mt (nothing to score).")

    setup = resolve_speed_setup(args)
    timing_producer = setup.timing_producer
    timing_params = setup.timing_params
    tts_apply_stretch = setup.tts_apply_stretch
    tts_force_align = setup.tts_force_align
    mix_strategy = setup.mix_strategy
    mix_tempo_cap = setup.mix_tempo_cap
    mix_clip_source = setup.mix_clip_source

    if args.timing_method and not args.tts:
        raise SystemExit("--timing-method requires --tts.")
    if args.timing_method and not args.mix:
        raise SystemExit("--timing-method requires --mix.")
    if args.speed_mode == "segment" and not (args.tts and args.mix):
        raise SystemExit("--speed-mode segment requires --tts and --mix "
                         "(there is nothing to re-pace otherwise).")
    if args.speed_mode == "segment" and args.no_tts_stretch:
        logging.getLogger(__name__).info(
            "--no-tts-stretch is implied by --speed-mode segment")

    result = run_pipeline(
        args.input,
        diarization_model=args.model,
        data_dir=args.data_dir,
        job_id=args.job_id,
        device=args.device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        skip_stems=args.skip_stems,
        resegment_turns=args.resegment_turns,
        resegment_max_duration_s=args.resegment_max_duration,
        refine_speaker_labels=args.refine_speaker_labels,
        speaker_purity_floor=args.speaker_purity_floor,
        speaker_min_speech_s=args.speaker_min_speech_s,
        stem_model=args.stem_model,
        face_detector=args.face_detector,
        face_tracker=args.face_tracker,
        face_every_n=args.face_every_n,
        face_min_score=args.face_min_score,
        asd_model=args.asd,
        asd_on_screen_threshold=args.asd_on_screen_threshold,
        asr_model=args.asr,
        # 'auto' is the explicit way back to detection; argparse cannot
        # express "a string, or None" on its own.
        asr_language=(None if (args.asr_language or "").lower()
                      in ("auto", "detect", "none", "") else args.asr_language),
        asr_full_audio=args.asr_full_audio,
        captions_as_transcript=not args.no_captions_transcript,
        captions_allow_auto=args.captions_allow_auto,
        align_model=args.align,
        mt_model=args.mt,
        mt_target_language=args.mt_target_language,
        mt_target_script=args.mt_target_script,
        translit_model=args.translit,
        termbase_path=args.termbase,
        duration_adjust=args.duration_adjust,
        duration_voice=args.duration_voice,
        duration_safety_margin_s=args.duration_safety_margin,
        voice_engine=args.voice_engine,
        voice_catalog_json=args.voice_catalog,
        allow_voice_reuse=args.allow_voice_reuse,
        tts_model=args.tts,
        tts_style=args.tts_style,
        tts_speed_control=args.tts_speed_control,
        tts_speed_bounds=args.tts_speed_bounds,
        tts_speed_tolerance=args.tts_speed_tolerance,
        tts_speed_max_rounds=args.tts_speed_max_rounds,
        emotion_model=args.emotion,
        emotion_min_confidence=args.emotion_min_confidence,
        emotion_min_run=args.emotion_min_run,
        tts_sample_rate=args.tts_sample_rate,
        tts_max_stretch=args.tts_max_stretch,
        stretch_backend=args.stretch_backend,
        tts_apply_stretch=tts_apply_stretch,
        tts_force_align=tts_force_align,
        vc_model=args.vc,
        mix=args.mix,
        mix_sample_rate=args.mix_sample_rate,
        mix_tts_gain_db=args.mix_tts_gain_db,
        mix_me_gain_db=args.mix_me_gain_db,
        mix_strategy=mix_strategy,
        mix_tempo_cap=mix_tempo_cap,
        mix_clip_source=mix_clip_source,
        timing_producer=timing_producer,
        timing_params=timing_params or None,
        mux_video=mux_video,
        lipsync_model=args.lipsync,
        lipsync_num_gpus=args.lipsync_num_gpus,
        subtitle_langs=subtitle_langs,
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
    if result.align_model:
        print(f"align model    : {result.align_model} ({result.num_aligned_words} words)")
        print(f"align json     : {result.align_json}")
    if result.mt_model:
        print(f"mt model       : {result.mt_model} → {result.mt_target_language} "
              f"({result.mt_target_script or 'default script'})")
        print(f"mt json        : {result.mt_json}")
        print(f"mt transcript  : {result.mt_txt}")
    if result.translit_model:
        print(f"translit       : {result.translit_model} "
              f"({result.codemix_retained_terms} retained terms)")
    if result.duration_model_language:
        print(f"duration       : lang={result.duration_model_language} "
              f"voice={result.duration_model_voice} "
              f"(est. violations: {result.duration_estimated_violations})")
    if result.voice_engine:
        print(f"voice engine   : {result.voice_engine}")
        print(f"voices         : {result.voice_assignments_json}")
    if result.speaker_references_dir:
        print(f"speaker refs   : {result.speaker_references_dir}")
    if result.tts_model:
        print(f"tts            : {result.tts_model} → {result.tts_out_dir}  "
              f"(errors={result.tts_errors}, cap_hits={result.tts_budget_violations})")
    if result.vc_model:
        print(f"vc             : {result.vc_model} → {result.vc_out_dir}  "
              f"(errors={result.vc_errors})")
    if result.timing_producer:
        print(f"timing         : {result.timing_producer} → {result.timing_plan_json}")
        if result.timing_summary:
            s = result.timing_summary
            print(f"  pieces={s.get('pieces', '?')}  "
                  f"cap_hits={s.get('tempo_cap_hits', '?')}  "
                  f"drift={s.get('drift_pct', '?')}%")
    if result.mix_wav:
        print(f"dubbed audio   : {result.mix_wav}")
    if result.mix_mp4:
        print(f"dubbed video   : {result.mix_mp4}")
    if result.lipsync_model:
        print(f"lipsync        : {result.lipsync_model} → {result.lipsync_mp4}"
              + (f"  (err={result.lipsync_error})" if result.lipsync_error else ""))
    if result.subtitles:
        print(f"subtitles      : "
              + ", ".join(f"{k}:{v['source']}" for k, v in result.subtitles.items()))
    import json as _json
    if result.asr_eval_json:
        rep = _json.loads(Path(result.asr_eval_json).read_text())
        print(f"asr eval       : vs {rep['transcript_source']} {rep['language']} "
              f"→ {result.asr_eval_json}")
        print(f"  WER={rep['wer']['wer']:.3f}  CER={rep['cer']['cer']:.3f}  "
              f"BLEU={rep['bleu']['bleu']:.3f}  chrF={rep['chrf']['chrf']:.3f}")
        print(f"  ROUGE-1={rep['rouge1']['f1']:.3f}  "
              f"ROUGE-2={rep['rouge2']['f1']:.3f}  "
              f"ROUGE-L={rep['rougeL']['f1']:.3f}  "
              f"METEOR={rep['meteor']['meteor']:.3f}")
    if result.mt_eval_json:
        rep = _json.loads(Path(result.mt_eval_json).read_text())
        print(f"mt eval        : vs {rep['transcript_source']} {rep['target_language']} "
              f"→ {result.mt_eval_json}")
        print(f"  BLEU={rep['bleu']['bleu']:.3f}  chrF={rep['chrf']['chrf']:.3f}  "
              f"ROUGE-1={rep['rouge1']['f1']:.3f}  "
              f"ROUGE-2={rep['rouge2']['f1']:.3f}  "
              f"ROUGE-L={rep['rougeL']['f1']:.3f}  "
              f"METEOR={rep['meteor']['meteor']:.3f}")
    if args.test:
        missing = []
        if args.asr and not result.asr_eval_json:
            missing.append("ASR (no source-language transcript found)")
        if args.mt and not result.mt_eval_json:
            missing.append("MT (no target-language transcript found)")
        if missing:
            raise SystemExit("--test failed: " + "; ".join(missing))
    print(f"manifest       : {result.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
