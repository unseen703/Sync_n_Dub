"""The one UI flow, the kwargs it sets, and what it costs.

`backend.pipeline._run_pipeline` takes about sixty keyword arguments. This
module sets only the ones the UI means; everything else keeps the pipeline
default.

Which ASR and MT to use are environment-overridable rather than hard-coded,
because they depend on which credentials an install has.

`estimate_runtime` backs the number the form shows before the operator
commits. Lip-sync factors are measured on this host
(`.claude/skills/clip-run`); the rest of the pipeline is a rough band and is
labelled approximate.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from backend import speed as _speed
from . import lipsync_models as _ls_models


def _scaled_lipsync_batch() -> int:
    """Default lipsync batch for the UI preset, scaled from VRAM.

    Imported lazily inside the function to keep `backend.scaling`'s torch
    probe off the import graph when the UI boots (torch is heavy and the
    probe is only wanted at submit time, after the main process has
    already paid for it elsewhere).
    """
    from backend.scaling import musetalk_batch_default
    return musetalk_batch_default()

#: Wall time as a multiple of clip length, on this host's 2x16 GiB Blackwell
#: cards. Derived from `ui/lipsync_models.py`; kept as a name here because
#: `routes_create` and the tests read it.
LIPSYNC_FACTORS: Dict[str, Tuple[float, float]] = {
    k: v["rate"] for k, v in _ls_models.FACTS.items() if v.get("rate")
}

#: Everything before lip-sync -- download, stems, diarization, ASR, MT, TTS,
#: mix. A rough band, not a measured factor, and reported as approximate.
BASE_FACTOR: Tuple[float, float] = (1.5, 5.0)
BASE_FLOOR_S: float = 60.0          # model load + download dominate short clips

#: One flow. Every run dubs the audio and stops; lip-sync is a separate,
#: explicit second step against a model the operator names at
#: `POST .../lipsync`, using the face size shown by
#: `GET /api/jobs/{id}/lipsync/readiness`.
PRESET_ID = "dub"

#: Old ids keep working: a stale browser bundle still posts
#: `preset: "standard"`, and jobs on disk name one of these.
PRESET_ALIASES: Dict[str, str] = {
    "draft": PRESET_ID, "standard": PRESET_ID, "best": PRESET_ID,
}

PRESET_IDS = (PRESET_ID,)

PRESETS: Dict[str, Dict[str, Any]] = {
    PRESET_ID: {
        "label": "Dub the audio",
        "note": ("Transcribes, translates and speaks the dub, then stops so "
                 "you can review the cast, the transcript and the face size "
                 "before committing GPU time to lip-sync."),
        "lipsync": None,
        # Stems are separated: this is the only pass there is, and its mix
        # is what the review judges.
        "skip_stems": False,
    },
}


def resolve_preset(name: Optional[str]) -> str:
    """Map any accepted preset id, including the retired ones, onto the
    single flow. Unknown ids raise, so a typo is not silently dubbed."""
    key = (name or PRESET_ID).strip().lower()
    key = PRESET_ALIASES.get(key, key)
    if key not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected {PRESET_ID}")
    return key

def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip()


def default_source_language() -> Optional[str]:
    """What the UI assumes is spoken in the source.

    Defaults to ``auto``: the probe endpoint detects the language from
    yt-dlp metadata or a Gemini Flash Lite audio analysis and pre-selects
    it. ``DUBBING_UI_SOURCE_LANGUAGE=en`` pins English instead.
    """
    v = _env("DUBBING_UI_SOURCE_LANGUAGE", "auto").lower()
    return None if v in ("auto", "detect", "none", "") else v


def default_asr() -> str:
    """Bodhan indic-transcribe.

    It takes no `asr_language`: the adapter sends `language` only when given
    one and reads the detected code back off the response.

    Two things to know:

    1. ASR, MT, TTS and the transliterator are all Bodhan, so an outage
       takes the whole run -- the registries hold Bodhan alone, with no
       local adapter to fall back to.
    2. It needs `BODHAN_ASR_KEY`, a different key from the MT and TTS ones.
       An install with only `BODHAN_MT_KEY` fails at stage 7.

    Word timings are not a consideration: stage 8 runs `mms_fa` regardless.
    """
    return _env("DUBBING_UI_ASR_MODEL", "bodhan")


def default_mt() -> str:
    return _env("DUBBING_UI_MT_MODEL", "bodhan")


def default_tts() -> str:
    """Bodhan indic-speak -- the only engine there is.

    45 voices over the 22 scheduled languages, matched per speaker by
    `voice_engine_for` turning on stage 9d's catalog, and the one engine
    that reads `--tts-style`. Needs `BODHAN_TTS_KEY`.
    """
    return _env("DUBBING_UI_TTS_MODEL", "bodhan")


def voice_engine_for(tts: str) -> Optional[str]:
    """Stage 9d's voice catalog, which is named differently from the engine.

    Still a lookup rather than a constant so that an unknown `tts` returns
    None and 9d skips casting, instead of casting from a catalog the engine
    cannot speak.
    """
    return {"bodhan": "indic-speak"}.get((tts or "").lower())


#: Defaults for the stages the UI used to leave unset. `plan_stages` keys off
#: this dict, so an absent key reads as "the operator chose not to".
UI_STAGE_DEFAULTS: Dict[str, Any] = {
    "resegment_turns": True,            # 4b  logical resegmentation
    # 4c  Split diarization labels holding more than one person. pyannote
    #     under-clusters noisy material, and 9d assigns one voice per label,
    #     so an impure label reads several characters in one voice while
    #     every artifact still looks right.
    "refine_speaker_labels": True,
    # loconet is the default: the only joint multi-speaker model, so it can
    # tell competing faces apart rather than scoring each in isolation. Its
    # weights are AVA-only and over-fire on dubbed footage, which is why the
    # on-screen threshold below is 0.5 rather than the CLI default 0.3.
    "asd_model": "loconet",             # 6   ASD + Hungarian assignment
    "asd_on_screen_threshold": 0.5,
    # No `render_mode`: the hosted indic-transcribe endpoint returns
    #    `HTTP 422 unexpected field(s): ['render_mode']` and fails the job
    #    at stage 7. Re-add it only after one rated call returns 200.
    "tts_style": "auto",                # 9b1 content register from metadata
    "emotion_model": "emotion2vec",     # 9b2 source emotion
    "tts_force_align": True,            # 9f1 post-TTS forced alignment
    "tts_align_model": "mms_fa",
    "attach_visemes_stage": True,       # 9f1b viseme extraction
    "viseme_backend": "mms_char",
}

#: Warp-mode pacing (the default): the continuous-rate producer plans a
#: per-cue tempo under a slew limit and a boundary ramp, borrows silence
#: rather than only speeding speech, and the mix reads the pre-stretched
#: warped clip. Built by `backend/speed.py`, called rather than copied.
#:
#: The three companion keys are not optional. `backend/tts.py` already
#: stretches each clip, so leaving `tts_apply_stretch` on stretches twice;
#: `mix_clip_source` must be `"warped"` or the mix places unstretched clips
#: at warped positions and the dub drifts.
#:
#: It takes no arguments: the UI does not expose these bounds, so it gets
#: the same defaults `--speed-step` / `--speed-tempo-max` carry.
UI_TIMING_DEFAULTS: Dict[str, Any] = _speed.warp_timing_kwargs()


def build_kwargs(preset: str, *, target_language: str,
                 tts: Optional[str] = None,
                 lipsync_model: Optional[str] = None,
                 num_speakers: Optional[int] = None,
                 skip_stems: Optional[bool] = None,
                 style: Optional[str] = None,
                 source_language: Optional[str] = None,
                 code_mix: bool = True) -> Dict[str, Any]:
    """The `run_pipeline` kwargs for one submission.

    `lipsync_model` is the resolved model -- `"auto"` is not a pipeline
    value, so `ui/runner.py` substitutes the face-picked one first. None
    means no lip-sync.
    """
    spec = PRESETS[resolve_preset(preset)]
    engine = (tts or default_tts()).lower()

    # No lip-sync here, ever. Phase one dubs the audio and stops; the model
    # is named at `POST /api/jobs/{id}/lipsync`. `lipsync_model` is still
    # accepted on the submit schema so an older bundle does not 422.
    #
    # Stages 5 (faces) and 6 (ASD) run anyway: the review screen reads the
    # per-speaker face size, and `assignments.json` feeds the gender vote
    # and the speaking-face targeting a later lip-sync uses.
    kw: Dict[str, Any] = {
        # who speaks when
        "diarization_model": "pyannote",
        "num_speakers": num_speakers,
        "skip_stems": bool(spec["skip_stems"] if skip_stems is None else skip_stems),
        # what they said, and what it becomes
        "asr_model": default_asr(),
        "align_model": "mms_fa",
        "mt_model": default_mt(),
        "mt_target_language": target_language,
        # Code-mixing is the MT API's own job, not a second pass, so
        # `translit_model` is absent (not None) and stage 9b is not planned.
        # Driven by the UI's "Code-mix" dropdown: "Code-mix" → "codemix"
        # (default — keeps English technical terms in Latin so Indic TTS does
        # not mispronounce a transliterated remnant); "Native" → "native"
        # (fully re-scripted translation).
        "mt_target_script": "codemix" if code_mix else "native",
        "duration_adjust": True,
        # 9c2, on for every UI run. The CLI leaves it off; a UI run has
        # nobody watching 9c's overrun warnings scroll past.
        #
        # It only fires past `condense_speed_cap`, so an ordinary run costs
        # zero LLM calls -- the gate is evaluated from 9c's estimates before
        # any request is made.
        #
        # Requires `duration_adjust` (it reads 9c's estimates) and
        # OPENROUTER_API_KEY. Without the key it logs and changes nothing.
        "condense_overlong_text": True,
        # 1.5, not the 1.8 the mixer may reach -- see `backend/condense.py`.
        "condense_speed_cap": 1.5,
        # saying it
        "voice_engine": voice_engine_for(engine),
        "tts_model": engine,
        "mix": True,
        "mux_video": True,
        # Always, not only when lip-sync is wanted — see the note above.
        "face_detector": "retinaface",
        # Recorded at submit even though lip-sync runs later: the replay
        # rebuilds its kwargs from this manifest config. Leave it out and
        # the adapter takes MuseTalk's own default of 20, measured at a
        # 15496 MiB peak on a 16311 MiB card -- an OOM once onnxruntime's
        # CUDA EP is resident.
        #
        # Batch size moves throughput and VRAM only, not output pixels.
        #
        # Named parameter, not the shared `batch_size` adapter kwarg, which
        # stage 4 would also hand to the diarizer. Scales with VRAM via
        # backend.scaling.musetalk_batch_default() -- 8 on the 16 GiB
        # reference host, more on bigger cards. DUBBING_LIPSYNC_BATCH wins.
        "lipsync_batch_size": _scaled_lipsync_batch(),
    }
    kw.update(UI_STAGE_DEFAULTS)
    # An explicit delivery type from the form replaces the `"auto"` default
    # (stage 9b1 classifying the register from metadata). `plan_stages` keys
    # 9b1 off `tts_style == "auto"`, so the progress bar follows.
    if style:
        kw["tts_style"] = style
    # Stated beats the install default, which beats detection. An explicit
    # "auto" from the form reaches here as the string and means detect.
    if (source_language or "").lower() in ("auto", "detect"):
        kw["asr_language"] = None
    else:
        kw["asr_language"] = source_language or default_source_language()
    kw.update(UI_TIMING_DEFAULTS)
    return kw


def plan_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Manifest-shaped config.

    Read by `progress.plan_stages` to decide which stages to draw, and by
    `backend.edit_agent.rerun` to re-pace a replay.

    The pacing block at the bottom exists for the second reader: a UI job
    has no pipeline argv for `recover_args` to re-parse, so without these
    keys a replay falls through to CLI defaults this submission never used.
    `use_target_duration` is the one that bites -- `edits.set_speed` works
    entirely by writing `target_duration_s`, so with it off a speed edit
    re-renders at the original speed.
    """
    return {
        "diarization":     kwargs.get("diarization_model"),
        "asr":             kwargs.get("asr_model"),
        "asr_language":    kwargs.get("asr_language"),
        "align":           kwargs.get("align_model"),
        "mt":              kwargs.get("mt_model"),
        "target_language": kwargs.get("mt_target_language"),
        "translit":        kwargs.get("translit_model"),
        "duration_adjust": kwargs.get("duration_adjust"),
        "voice_engine":    kwargs.get("voice_engine"),
        "tts":             kwargs.get("tts_model"),
        "face_detector":   kwargs.get("face_detector"),
        "asd":             kwargs.get("asd_model"),
        "lipsync":         kwargs.get("lipsync_model"),
        # 9i is never part of a submission -- the UI reaches it through
        # `POST .../enhance` on a finished job. Recorded as None so the key
        # exists for `plan_stages`.
        "restore":         kwargs.get("restore_model"),
        "mix":             kwargs.get("mix"),
        "skip_stems":      kwargs.get("skip_stems"),
        "timing_producer": kwargs.get("timing_producer"),
        # `plan_stages` keys off these; without them the stages run but are
        # not planned, so the bar under-counts and appends nodes late.
        "resegment_turns": kwargs.get("resegment_turns"),
        "refine_speakers": kwargs.get("refine_speaker_labels"),
        # 9c2. `plan_stages` reads this together with `duration_adjust`;
        # without it the stage runs as an unplanned node.
        "condense_overlong_text": kwargs.get("condense_overlong_text"),
        "tts_style":       kwargs.get("tts_style"),
        "emotion":         kwargs.get("emotion_model"),
        "tts_force_align": kwargs.get("tts_force_align"),
        "visemes":         kwargs.get("attach_visemes_stage"),
        # -- how this run was PACED, for the replay --------------------------
        # Read by `edit_agent._setup_from_config` when there is no argv to
        # re-parse.
        "timing_params":     kwargs.get("timing_params"),
        "tts_apply_stretch": kwargs.get("tts_apply_stretch"),
        "mix_strategy":      kwargs.get("mix_strategy"),
        "mix_clip_source":   kwargs.get("mix_clip_source"),
        "mix_tempo_cap":     kwargs.get("mix_tempo_cap"),
        "tts_max_stretch":   kwargs.get("tts_max_stretch"),
    }


# ---------------------------------------------------------------------------
# estimates
# ---------------------------------------------------------------------------

def estimate_runtime(duration_s: float, *,
                     lipsync_model: Optional[str] = None) -> Dict[str, Any]:
    """`{low_s, high_s, text, exact, lipsync_measured}` for a clip.

    `exact` is always False: the base band is not measured the way the
    lip-sync factors are, and lip-sync itself spans 8-21x on one model.

    `lipsync_measured` is False when the chosen model has no measured band.
    Callers must surface that -- an unmeasured model otherwise returns the
    dub-only estimate and advertises as the fastest option on the list.
    """
    d = max(0.0, float(duration_s or 0.0))
    low = max(BASE_FLOOR_S, d * BASE_FACTOR[0])
    high = max(BASE_FLOOR_S * 2, d * BASE_FACTOR[1])

    parts = ["the dub itself"]
    asked = bool((lipsync_model or "").strip())
    ls = _ls_models.normalise(lipsync_model)
    band = _ls_models.rate_band(ls) if ls else None
    if band:
        lo, hi = band
        low += d * lo
        high += d * hi
        parts.append(f"{ls} at {lo:g}-{hi:g}x")

    text = f"about {human_duration(low)} to {human_duration(high)}"
    if asked and not band:
        # Never a bare number for a model nobody timed.
        text = (f"{text} for the lipsyncing.")

    return {
        "low_s": int(round(low)),
        "high_s": int(round(high)),
        "text": text,
        "basis": "Measured on this host: " + ", ".join(parts) + ".",
        "exact": False,
        "lipsync_measured": bool(band),
    }


def human_duration(seconds: float) -> str:
    s = int(round(max(0.0, float(seconds))))
    if s < 90:
        return f"{s} seconds"
    m = s // 60
    if m < 90:
        return f"{m} minutes"
    h, rem = divmod(m, 60)
    return f"{h}h{rem:02d}m" if rem else f"{h} hours"
