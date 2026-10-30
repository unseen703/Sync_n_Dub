"""CLI entry point:

    python -m backend.cli --input <url_or_path> \
        [--model pyannote] \
        [--face-detector retinaface|mediapipe --face-tracker bytetrack] \
        [--asd loconet] \
        [--asr bodhan --asr-language hi] \
        [--align mms_fa] \
        [--mt bodhan --mt-target-language hi
              --mt-target-script codemix|native|roman]
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

from backend import speed as _speed

from models.align import available_models as _align_models
from models.asd import available_models as _asd_models
from backend.asr import available_models as _asr_models
from backend.diarize import available_models as _diar_models
from models.faces.detect import available_detectors as _detectors
from models.faces.track import available_trackers as _trackers
from models.lipsync import available_models as _lipsync_models


#: Default IMPLEMENTATION for a bare `--asr` / `--mt` / `--tts`, not a
#: default-on stage: an absent flag still means skip, because `_validate`'s
#: dependency chain keys off None and a real default here would force every
#: run to supply --mt-target-language. Falls back to the registry's first
#: entry when bodhan is not installed.
_PREFERRED_IMPL = "bodhan"


def _speed_bounds(s: str) -> tuple:
    """`"0.85,1.15"` -> `(0.85, 1.15)`, validated.

    argparse cannot express a bounded pair, and a reversed or inverted range
    clamps every rate to one end without complaining.
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
    from backend.tts.tts import SPEED_CONTROL_MODES
    return SPEED_CONTROL_MODES


def _emotion_models():
    try:
        from backend.emotion.emotion import available_models
        return available_models()
    except ImportError:
        return []


def _default_impl(stage: str) -> str:
    avail = {"asr": _asr_models, "mt": _mt_models, "tts": _tts_models}[stage]()
    return _PREFERRED_IMPL if _PREFERRED_IMPL in avail else (avail[0] if avail else None)


def _tts_styles():
    """Styles the Bodhan TTS engine accepts, plus `auto` and `neutral`.

    Neither extra is a Bodhan value:

      * `auto` asks `backend.classify` to pick the register from the video's
        metadata, or an LLM read of the transcript, and resolves to a real
        style or None before synthesis.
      * `neutral` sends no style at all -- the voice's plain reading.
        `models.tts.resolve_style` maps it to None.

    Listed here so the CLI offers exactly what the UI dropdowns do.
    """
    from backend.tts.tts import NEUTRAL_STYLE
    try:
        from backend.tts.bodhan_adapter import INDIC_SPEAK_STYLES
        return ("auto", NEUTRAL_STYLE) + tuple(INDIC_SPEAK_STYLES)
    except ImportError:
        return ("auto", NEUTRAL_STYLE)
from backend.translate import available_models as _mt_models
from backend.translate.translit import available_models as _translit_models
from backend.tts.tts import available_models as _tts_models

from .pipeline import DEFAULT_DATA_DIR, REPO_ROOT, run_pipeline


def _check_env_permissions() -> None:
    """Refuse to start if `.env` is group- or world-readable.

    On a shared box a mode-644 .env leaks its HF / Bodhan tokens to every
    account on the host.
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


# Characters that only appear in a LIST of inputs, never in one acceptable
# path or URL. Catching them turns a silent run on one of three files into a
# parse-time error.
_MULTI_INPUT_HINTS = (",", "\n", "\t", "|", ";")

#: Shell globs that arrived unexpanded (quoted, or no match): each stands for
#: a set of files, not one.
_GLOB_CHARS = ("*", "?", "[")


class _SingleInput(argparse.Action):
    """Accept exactly one video/audio input per run.

    One job folder, one video stem, one set of stage outputs -- several
    inputs collide in that folder, so every way of naming more than one is
    rejected at parse time rather than halfway through stage 9.
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
    to --input and the rest arrive as positionals. argparse would only say
    "unrecognized arguments".
    """
    media = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4a", ".wav", ".mp3", ".flac")
    # Only true positionals count: a value following an option belongs to
    # that option, so `--input A --input B` is the action's repeat error.
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
    ap.add_argument("--asr-language", default="auto",
                    help="ISO-639 code of what is SPOKEN in the source "
                         "(en, hi, ta, …). Default 'auto': the ASR probes "
                         "the longest diarization turn once and pins that "
                         "answer to the whole run. Pass a code to skip the "
                         "probe when you already know the language.")
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
                         "codemix keeps Latin-script source terms in Latin "
                         "inside the target-script sentence, which matches how this "
                         "material is spoken and stays consistent across "
                         "segments; native re-scripts, but Bodhan leaves Latin "
                         "remnants at random when it does.")

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
    ap.add_argument("--condense-overlong", action="store_true",
                    help="Stage 9c2. Ask an LLM to shorten lines that "
                         "--duration-adjust predicts will overrun their "
                         "slot, BEFORE they are spoken. The only lever "
                         "left when the tempo cap is already maxed: Bodhan "
                         "has no speed parameter, so audio cannot be "
                         "shortened after the fact. Requires "
                         "--duration-adjust (it reads 9c's estimates) and "
                         "OPENROUTER_API_KEY. A rewrite that does not "
                         "measure shorter is discarded, and any failure "
                         "keeps the original wording.")
    ap.add_argument("--condense-model", default=None,
                    help="Model for 9c2 (default: $CONDENSE_MODEL, else "
                         "backend.condense.DEFAULT_MODEL). A small fast "
                         "model suits the task -- it is 'say this shorter', "
                         "not reasoning.")
    ap.add_argument("--condense-speed-cap", type=float, default=1.5,
                    help="Rewrite a line only once it needs MORE speed-up "
                         "than this to fit (default 1.5). Below it the mix "
                         "can still stretch the clip, and rewriting text "
                         "that already renders correctly is a translation "
                         "risk for nothing. Deliberately separate from "
                         "--speed-cap (1.8): that is how hard the mixer may "
                         "push, this is when to re-word instead.")
    ap.add_argument("--condense-tolerance", type=float, default=0.0,
                    help="Extra slack above --condense-speed-cap before a "
                         "rewrite is worth a call (default 0.0, i.e. the "
                         "cap is the cap).")
    ap.add_argument("--condense-batch-size", type=int, default=None,
                    help="Segments per 9c2 request (default 2, or "
                         "$CONDENSE_BATCH_SIZE). The ~2.9k-token system "
                         "prompt is sent once per request, not once per "
                         "segment. Kept small on purpose: a model holding "
                         "several lines at once can blend one into "
                         "another, which is short, fluent and wrong, and "
                         "the duration check cannot see it. 1 sends each "
                         "segment alone.")

    # Voice assignment per speaker --------------------------------------
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
                         "`max(1-cap, min(1+cap, ratio))`).")
    ap.add_argument("--stretch-backend", default="auto",
                    choices=("auto", "rubberband", "ffmpeg_atempo",
                             "librosa", "linear"),
                    help="Time-stretch implementation. 'auto' picks the best "
                         "available (rubberband here). 'linear' is a "
                         "pitch-SHIFTING resample kept only as an ablation "
                         "baseline.")
    ap.add_argument("--no-tts-stretch", action="store_true",
                    help="Synthesise raw clips only, skipping the per-segment "
                         "stretch, so a downstream timing plan owns all "
                         "stretching instead of compounding with it.")

    # Timing method — top 3 from the eight-method sweep -------------------
    # DEFAULT None, not "segment": `resolve_speed_setup` refuses
    # --speed-mode segment alongside --timing-method, so a literal default
    # kills every research-arm invocation at startup. None distinguishes "the
    # operator asked for segment" from "nobody said".
    ap.add_argument("--speed-mode", default=None,
                    choices=("global", "segment", "warp"),
                    help="How the dub is re-paced to fit the source timeline. "
                         "`segment` (the default): each clip "
                         "is fitted to its own slot -- compressed when the "
                         "dub line is longer, STRETCHED OUT when it is "
                         "shorter, so a short line does not leave a pause in "
                         "the middle of its slot. Bounded by --speed-cap; "
                         "whatever the bound could not absorb is reported as "
                         "unfilled_slot_s in the timing plan. `global`: "
                         "every clip plays at its natural "
                         "pace, placed shift-forward, then ONE atempo over "
                         "the whole file so the dub lands on the source "
                         "duration. Passing --timing-method implies `global`, "
                         "since the arm sets these knobs itself.")
    ap.add_argument("--speed-cap", type=float, default=1.8,
                    help="With --speed-mode segment: the SPEED-UP bound, "
                         "applied as max_tempo=cap. Hindi TTS runs "
                         "1.21-1.65x longer than the English it covers, so a "
                         "cap of 1.5 leaves no headroom above the top of that "
                         "range. The slowdown floor is NOT 1/cap; see "
                         "--speed-floor.")
    ap.add_argument("--speed-floor", type=float, default=0.75,
                    help="With --speed-mode segment: the SLOW-DOWN bound, "
                         "applied as min_tempo. Default 0.75. Deliberately "
                         "not 1/--speed-cap: a symmetric 1.8 would stretch a "
                         "short line to 0.56x -- nearly double length -- to "
                         "fill its slot, which is audible as drawl. Whatever "
                         "this floor cannot absorb stays a real pause and is "
                         "reported as unfilled_slot_s rather than hidden.")
    ap.add_argument("--speed-slot-margin", type=float, default=0.0,
                    help="With --speed-mode segment: seconds of slot left "
                         "deliberately empty at the end of each segment. "
                         "Default 0.0 -- any margin IS a pause, which is the "
                         "thing this mode exists to remove. Raise it if "
                         "clips are running into each other.")
    ap.add_argument("--speed-gap-barrier", type=float, default=1.0,
                    help="With --speed-mode segment: a source gap at least "
                         "this long after a cue is treated as a WALL, so an "
                         "overrunning line is sped up to end at its slot "
                         "rather than running on over whatever fills the gap "
                         "— usually music. Default 1.0 s; 0 disables it and "
                         "restores the plain shift-forward cascade.")
    ap.add_argument("--speed-borrow-gap", type=float,
                    default=_speed.DEFAULT_BORROW_GAP,
                    help="With --speed-mode segment: seconds of the gap "
                         "AFTER a cue that the cue may take rather than "
                         "speed up. Turns --speed-gap-barrier from a wall "
                         "into a budget. DEFAULT 0.0 (off), because any "
                         "other value re-paces every cue that has a gap "
                         "after it. 0.3-0.5 is the range the prosodic-"
                         "alignment literature works in. A nested cue and "
                         "the last cue never borrow, and a cue whose slot "
                         "was shortened by `set_speed` does not either — "
                         "borrowing would cancel that edit. Reported per "
                         "segment as borrowed_s, with the tempo it would "
                         "have had as tempo_unborrowed.")
    ap.add_argument("--speed-borrow-frac", type=float,
                    default=_speed.DEFAULT_BORROW_FRAC,
                    help="With --speed-borrow-gap: the share of the "
                         "following gap a cue may take. Default 0.5 — the "
                         "gap is shared, and taking all of it leaves the "
                         "next cue no cushion.")
    ap.add_argument("--absorb-pauses", action="store_true",
                    help="With --speed-mode segment: spend a cue's duration "
                         "deficit on the SILENCE inside its own TTS clip "
                         "before speeding the speech up. The clip is VADed, "
                         "its gaps are shrunk to tiered floors (sentence "
                         "boundaries keep more than breaths), and only what "
                         "the silence cannot cover reaches the speech — so a "
                         "line with sentence breaks in it often renders at "
                         "exactly 1.0x. Runs in reverse too: a clip shorter "
                         "than its slot grows its pauses instead of drawling "
                         "at --speed-floor. The cue becomes one piece per "
                         "speech run; a clip the VAD cannot use falls back "
                         "to the single-clip tempo. Read speech_tempo_max in "
                         "the plan summary, not tempo_max.")
    ap.add_argument("--absorb-max-frac", type=float,
                    default=_speed.DEFAULT_ABSORB_MAX_FRAC,
                    help="With --absorb-pauses: the most any ONE pause may "
                         "give up, even where its floor would allow more. "
                         "Default 0.60. A dub with every pause floored is "
                         "relentless.")
    ap.add_argument("--absorb-vad-backend",
                    default=_speed.DEFAULT_ABSORB_VAD_BACKEND,
                    choices=("energy", "pyannote", "auto"),
                    help="With --absorb-pauses: VAD used to find the clip's "
                         "pauses. Default 'energy' — a TTS clip is clean "
                         "audio with near-digital silence in its gaps, which "
                         "is energy's good case, and pyannote would load a "
                         "model and hold VRAM into stage 9h.")

    # --speed-mode warp ---------------------------------------------------
    ap.add_argument("--speed-step", type=float,
                    default=_speed.DEFAULT_WARP_STEP,
                    help="With --speed-mode warp: the largest RELATIVE tempo "
                         "change allowed between adjacent cues. Default 0.20. "
                         "The just-noticeable difference for speech tempo is "
                         "about 10%% (Quene 2007), so 0.20 is roughly twice "
                         "the detection threshold — audible but not a lurch. "
                         "Enforced per island, symmetric in the log domain, "
                         "and VERIFIED on what was rendered: read "
                         "tempo_step_max / tempo_step_ok in the plan summary.")
    ap.add_argument("--speed-tempo-max", type=float,
                    default=_speed.DEFAULT_WARP_TEMPO_MAX,
                    help="With --speed-mode warp: the soft SPEECH-rate "
                         "ceiling. Default 1.45. This is a speech rate, not "
                         "the blob rate --speed-cap bounds: the engine's own "
                         "silence is compressed separately and much harder.")
    ap.add_argument("--speed-tempo-hard-max", type=float,
                    default=_speed.DEFAULT_WARP_TEMPO_HARD_MAX,
                    help="With --speed-mode warp: the rate a cue may be "
                         "pushed to when nothing else fits. Default 1.80. "
                         "Reaching it is flagged budget_violation — the "
                         "answer there is to shorten the line "
                         "(--condense-overlong), not to push harder.")
    ap.add_argument("--speed-island-gap", type=float,
                    default=_speed.DEFAULT_WARP_ISLAND_GAP,
                    help="With --speed-mode warp: a gap at least this long "
                         "breaks the smoothing chain. Each island's total "
                         "duration is conserved exactly, so drift can only "
                         "live inside one. Default 1.0 s.")
    ap.add_argument("--speed-ramp", type=float,
                    default=_speed.DEFAULT_WARP_RAMP_S,
                    help="With --speed-mode warp: width of the smoothstep "
                         "rate transition at each cue joint. Default 0.18 s. "
                         "Slid into a nearby pause when there is one — a rate "
                         "change inside silence is far less audible than one "
                         "mid-phoneme.")
    ap.add_argument("--warp-backend", default="auto",
                    choices=("auto", "rubberband", "pytsmod"),
                    help="With --speed-mode warp: the time-map stretcher. "
                         "'auto' prefers rubberband (pyrubberband + the "
                         "rubberband CLI) and falls back to pytsmod's WSOLA. "
                         "Note the ffmpeg `rubberband` FILTER in the "
                         "imageio build is a different thing and cannot take "
                         "a time map.")

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
                    choices=("auto", "raw", "stretched"),
                    help="Which per-segment clip the mix reads. 'auto' takes "
                         "the stretched clip the TTS stage wrote; 'raw' takes "
                         "the unstretched one, which is what you want when the "
                         "mix applies its own tempo.")
    ap.add_argument("--mix-tempo-cap", type=float, default=1.5,
                    help="Upper bound for the global atempo factor "
                         "(global_tempo strategy). Default 1.5x.")
    ap.add_argument("--no-mux", action="store_true",
                    help="With --mix: only write the dubbed WAV, skip MP4 mux.")

    # Lip sync ----------------------------------------------------------
    ap.add_argument("--lipsync", default=None, choices=_lipsync_models(),
                    help="Lip-sync model (wav2lip | musetalk | latentsync15). "
                         "Requires --mix.")
    ap.add_argument("--restore", default=None, choices=("gfpgan",),
                    help="Stage 9i: face restoration over the finished "
                         "render. Orthogonal to --lipsync -- it takes a "
                         "video and returns a video and does not care which "
                         "adapter drew the mouths, so any model can be "
                         "paired with it. NEVER implied: measured on this "
                         "host it costs 1156 s against the MuseTalk render's "
                         "447 s, i.e. 2.6x the thing it cleans. Restores "
                         "only the frames lip-sync changed (found by "
                         "diffing) and never fails the run -- a post-pass "
                         "that falls over leaves stage 9h's usable dub.")
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

    # Adapter-specific kwargs, forwarded via **adapter_kwargs. Each adapter
    # drops what it does not recognise, so a wide union is safe.
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (pyannote adapter).")
    ap.add_argument("--checkpoint", default=None,
                    help="Override checkpoint for whichever adapter uses one.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Batch size, for whichever adapter uses one.")
    ap.add_argument("--allow-insecure-env", action="store_true",
                    help="Skip the .env file-permission check on startup.")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Speed / timing resolution
# ---------------------------------------------------------------------------

#: What `resolve_speed_setup` returns. Named so `backend.edit_agent` replays
#: a run's timing setup rather than re-deriving it and landing on the
#: producer's defaults, which turns a `--speed-mode segment` run into a
#: 1.10x-capped, 0.10 s-margin one on the first edit.
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

    # --speed-mode is the plain-language front end to the research arms'
    # machinery: `segment` IS naive_slot through the timing plan, with a zero
    # margin and a symmetric cap; `global` leaves every default alone; `warp`
    # is the default. `--speed-mode` parses to None when nobody passed it, so
    # an EXPLICIT mode beside `--timing-method` is refused as a
    # contradiction, while the default yields to the arm, which sets
    # mix_strategy / tts_apply_stretch / mix_clip_source itself.
    speed_mode = args.speed_mode or ("global" if args.timing_method else "warp")

    if speed_mode == "segment":
        if args.timing_method:
            raise SystemExit(
                "--speed-mode segment and --timing-method set the same "
                "knobs. Use one: --speed-mode for the two-option choice, "
                "--timing-method for the measured research arms (m2 is the "
                "same producer with a 0.10 s slot margin).")
        _speed.validate_segment_bounds(args.speed_cap, args.speed_floor)
        _speed.validate_absorb_bounds(
            args.speed_borrow_gap, args.speed_borrow_frac,
            args.absorb_max_frac)
        # The values live in `backend/speed.py`, which `frontend/presets` also
        # calls, so the two cannot drift.
        _seg = _speed.segment_timing_kwargs(
            speed_cap=args.speed_cap,
            speed_floor=args.speed_floor,
            slot_margin=args.speed_slot_margin,
            gap_barrier=args.speed_gap_barrier,
            borrow_gap=args.speed_borrow_gap,
            borrow_frac=args.speed_borrow_frac,
            absorb_pauses=args.absorb_pauses,
            absorb_max_frac=args.absorb_max_frac,
            absorb_vad_backend=args.absorb_vad_backend,
        )
        timing_producer = _seg["timing_producer"]
        timing_params = _seg["timing_params"]
        tts_apply_stretch = _seg["tts_apply_stretch"]
        mix_strategy = _seg["mix_strategy"]
        mix_clip_source = _seg["mix_clip_source"]

    if speed_mode == "warp":
        if args.timing_method:
            raise SystemExit(
                "--speed-mode warp and --timing-method set the same knobs. "
                "Use one: --speed-mode warp for the continuous-rate "
                "producer, --timing-method for the measured research arms.")
        _speed.validate_warp_bounds(args.speed_step, args.speed_tempo_max,
                                    args.speed_tempo_hard_max)
        _speed.validate_segment_bounds(args.speed_cap, args.speed_floor)
        _warp = _speed.warp_timing_kwargs(
            speed_floor=args.speed_floor,
            step=args.speed_step,
            tempo_max=args.speed_tempo_max,
            tempo_hard_max=args.speed_tempo_hard_max,
            island_gap=args.speed_island_gap,
            ramp_s=args.speed_ramp,
            absorb_max_frac=args.absorb_max_frac,
            vad_backend=args.absorb_vad_backend,
            warp_backend=args.warp_backend,
        )
        timing_producer = _warp["timing_producer"]
        timing_params = _warp["timing_params"]
        tts_apply_stretch = _warp["tts_apply_stretch"]
        mix_strategy = _warp["mix_strategy"]
        mix_clip_source = _warp["mix_clip_source"]

    # Both knobs reach the pipeline ONLY through `segment_timing_params` and
    # are inert anywhere else, so asking for them is refused rather than
    # ignored: `global` builds no timing plan, the research arms set
    # `timing_params` wholesale below, and `warp` does both by construction.
    if speed_mode != "segment" and (args.absorb_pauses
                                    or args.speed_borrow_gap > 0):
        asked = [n for n, on in (("--absorb-pauses", args.absorb_pauses),
                                 ("--speed-borrow-gap", args.speed_borrow_gap > 0))
                 if on]
        why = {
            "warp": " — warp absorbs pauses and borrows across boundaries as "
                    "part of its solve; use --absorb-max-frac and "
                    "--speed-island-gap to steer it",
            "global": " (--speed-mode global builds no timing plan at all)",
        }.get(speed_mode, " (--timing-method sets its own timing_params)")
        raise SystemExit(
            f"{' and '.join(asked)} only applies to --speed-mode segment, "
            f"and this run resolved to {speed_mode!r}{why}.")

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
        from backend.diarize.pyannote_adapter import (
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
    if args.condense_overlong and not args.duration_adjust:
        raise SystemExit(
            "--condense-overlong requires --duration-adjust: it shortens "
            "the lines stage 9c predicts will overrun, so without 9c there "
            "are no estimates to act on.")
    if args.tts and not args.mt:
        raise SystemExit("--tts requires --mt.")
    if args.emotion and not args.tts:
        raise SystemExit("--emotion sets a per-segment TTS style; "
                         "it requires --tts.")
    if args.mix and not args.tts:
        raise SystemExit("--mix requires --tts.")
    if args.lipsync and not args.mix:
        raise SystemExit("--lipsync requires --mix.")
    # "auto" means detect, which is the absence of a language rather than
    # a language. Normalise once, here, so the gate below and the pipeline
    # call further down cannot disagree about what it means.
    if (args.asr_language or "").lower() in ("auto", "detect", "none", ""):
        args.asr_language = None
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
    # Both keyed on the EXPLICIT flag (`args.speed_mode`, not the resolved
    # mode), and deliberately so now that `segment` is the default: refusing
    # a transcribe-only run because of a default nobody typed would be a
    # regression. These stay what they always were -- guards against an
    # operator asking for something that cannot happen.
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
        asd_model=args.asd  or "loconet",
        asd_on_screen_threshold=args.asd_on_screen_threshold,
        asr_model=args.asr,
        # 'auto' is the explicit way back to detection; argparse cannot
        # express "a string, or None" on its own.
        asr_language=args.asr_language,
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
        condense_overlong_text=args.condense_overlong,
        condense_model=args.condense_model,
        condense_speed_cap=args.condense_speed_cap,
        condense_tolerance=args.condense_tolerance,
        condense_batch_size=args.condense_batch_size,
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
        restore_model=args.restore,
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
