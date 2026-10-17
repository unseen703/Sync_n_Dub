"""Three ways to run the pipeline, and what each one costs.

`backend.pipeline._run_pipeline` takes about sixty keyword arguments. A form
that exposed them would be a worse CLI; a form that exposed none of them
could not produce a lip-synced dub. So the UI offers three named points on
that surface and sets **only** the arguments it means — everything else keeps
the pipeline's own default, which is where the knowledge about it lives.

    draft     no lip-sync. Audio dub only, stems skipped. Minutes, not hours;
              this is the "is the translation any good" run.
    standard  lip-sync with the auto-picked model + CodeFormer restoration.
    best      MuseTalk + CodeFormer, regardless of face size.

Model choices that are not a preset decision (which ASR, which MT) are
environment-overridable rather than hard-coded, because they depend on which
credentials this install has — Bodhan needs keys, and CLAUDE.md records that
its endpoints return 5xx during upstream outages. Since 2026-10-17 ASR is
Bodhan too, so every hosted stage of a default run is one vendor: an outage
there is total, and `DUBBING_UI_ASR_MODEL=whisperx` is the way back to a
local transcriber.

Estimates
---------
`estimate_runtime` exists because the form shows a number for every video
before the operator commits. The lip-sync factors are measured on this host
(`.claude/skills/clip-run`): wav2lip 3-5x clip length, musetalk 8-21x,
latentsync ~83x, CodeFormer a further ~13x. The non-lip-sync part of the
pipeline is NOT measured that way — the range below is a rough band and is
labelled as such, because quoting a precise number nobody measured is how a
UI teaches people to distrust it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

#: Measured wall time as a multiple of clip length, on this host's 2x16 GiB
#: Blackwell cards. From `.claude/skills/clip-run`.
LIPSYNC_FACTORS: Dict[str, Tuple[float, float]] = {
    "wav2lip":    (3.0, 5.0),
    "musetalk":   (8.0, 21.0),
    "latentsync": (75.0, 90.0),     # measured "~83x"
    "lipgan":     (5.0, 15.0),      # torch port, ~10 it/s; least characterised
}

#: CodeFormer is a second pass, not a cheaper one: ~13x clip length.
RESTORE_FACTORS: Dict[str, Tuple[float, float]] = {
    "codeformer": (10.0, 16.0),
}

#: Everything before lip-sync — download, stems, diarization, ASR, MT, TTS,
#: mix. Not measured as a factor the way the GPU stages were; this is a band
#: wide enough to be honest and is reported as approximate.
BASE_FACTOR: Tuple[float, float] = (1.5, 5.0)
BASE_FLOOR_S: float = 60.0          # model load + download dominate short clips

PRESET_IDS = ("draft", "standard", "best")

PRESETS: Dict[str, Dict[str, Any]] = {
    "draft": {
        "label": "Draft — audio only",
        "note": "Dubs the audio and skips lip-sync entirely. Minutes, not hours.",
        "lipsync": None,
        "restore": None,
        "skip_stems": True,
    },
    "standard": {
        "label": "Standard — lip-synced",
        "note": ("Lip-syncs with the model picked from the face size in the "
                 "video."),
        "lipsync": "auto",
        "restore": None,
        "skip_stems": False,
    },
    "best": {
        "label": "Best — MuseTalk",
        "note": ("Always MuseTalk at 256x256. Slowest; use it when the "
                 "speaker fills the frame."),
        "lipsync": "musetalk",
        "restore": None,
        "skip_stems": False,
    },
}

#: Restoration is OPT-IN and no preset turns it on. It was a preset default
#: until 2026-10-17, which was a promise the pipeline did not keep: the UI
#: wrote `restore: codeformer` into every standard/best manifest while
#: `_run_pipeline` never forwarded `restore_model` to the stage that runs it,
#: so no UI job has ever been restored. Rather than switch a silent no-op into
#: a silent doubling of GPU time, it now has to be asked for by name
#: (`restore_model` on the request). The forwarding bug is fixed either way.
DEFAULT_RESTORE: Optional[str] = None


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip()


def default_asr() -> str:
    """Bodhan indic-transcribe. Changed from `whisperx` on 2026-10-17.

    It takes no `asr_language`: the adapter sends `language` only when one is
    given and reads the detected code back off the response, so the UI's
    "don't ask the operator what language the source is" behaviour is
    unchanged.

    Two consequences worth knowing, neither of which blocks the choice:

    1. **A Bodhan outage now takes the whole run, not most of it.** ASR, MT,
       TTS and the transliterator are all Bodhan here, and CLAUDE.md records
       that these endpoints answer `internal_error` 5xx during upstream
       outages rather than rate-limiting. whisperx ran locally, so it used
       to be the one stage an outage could not touch. There is no automatic
       fallback — `DUBBING_UI_ASR_MODEL=whisperx` is the manual one.
    2. **It needs `BODHAN_ASR_KEY`**, which is a different key from the MT
       and TTS ones (the plan's "credentials are per service"). An install
       with only `BODHAN_MT_KEY` set will fail at stage 7 rather than
       falling back.

    Word timings are not a consideration: stage 8 runs `mms_fa` for
    alignment regardless of which ASR produced the text.
    """
    return _env("DUBBING_UI_ASR_MODEL", "bodhan")


def default_mt() -> str:
    return _env("DUBBING_UI_MT_MODEL", "bodhan")


def default_tts() -> str:
    """Bodhan indic-speak. The UI offers no other engine.

    Since Edge was removed it is the only stock-voice engine: 45 voices over
    the 22 scheduled languages, matched per speaker by `voice_engine_for`
    turning on stage 9d's catalog, and the only engine that reads
    `--tts-style`. The alternative, IndicF5, clones the source speaker from
    their own clip and has no catalog to pick from at all.

    Needs `BODHAN_TTS_KEY`. `DUBBING_UI_TTS_MODEL` can still name `indicf5`
    for an install with no credentials, at the cost of cloning rather than
    casting.
    """
    return _env("DUBBING_UI_TTS_MODEL", "bodhan")


def voice_engine_for(tts: str) -> Optional[str]:
    """Stage 9d's engine, which is not the same thing as the TTS engine.

    `voice_engine` selects a voice *catalog*. A cloning engine has none --
    IndicF5's "voices" ARE reference clips -- so stage 9d takes its `else`
    branch there and the speaker's own clip carries the timbre instead.
    """
    return {"bodhan": "indic-speak", "indicf5": "indicf5"}.get((tts or "").lower())


#: Defaults for the stages the UI used to leave unset. Every one of these was
#: a stage the pipeline supports, the progress bar already has a slot for, and
#: that simply never ran because `build_kwargs` did not name it -- `plan_stages`
#: keys off this dict, so an absent key reads as "the operator chose not to".
#:
#: Two of them are why stages 5 and 8 existed without a consumer: ASD is the
#: only reader of the face tracks, and the timing plan is the only reader of
#: `words[]`. Turning them on is what makes that work count for something.
UI_STAGE_DEFAULTS: Dict[str, Any] = {
    "resegment_turns": True,            # 4b  logical resegmentation
    # 4c  Split diarization labels that hold more than one person. pyannote
    #     under-clusters noisy material, and since 9d assigns one voice PER
    #     LABEL, an impure label reads several characters in one voice while
    #     every artifact still looks right. Measured on a 279 s mob scene:
    #     4 labels -> 11, and labels below the single-speaker purity floor
    #     went 3-of-4 to 1-of-11.
    "refine_speaker_labels": True,
    "asd_model": "talknet",             # 6   ASD + Hungarian assignment
    "asd_on_screen_threshold": 0.3,
    # 7  Indic-Transcribe renders in native script, mixed (English and
    #    numerals left in Latin) or fully romanized — plan.md section 5.
    #    `mixed` because this transcript is MT INPUT: `native` would
    #    re-script "weight" to "वेट" before Bodhan MT ever sees it, which
    #    destroys exactly the English terms `mt_target_script="codemix"`
    #    then asks to be preserved. Left unset it is whatever the server
    #    defaults to, which is not a thing to leave to chance in the one
    #    stage that decides what every later stage reads.
    "render_mode": "mixed",
    "tts_style": "auto",                # 9b1 content register from metadata
    "emotion_model": "emotion2vec",     # 9b2 source emotion
    "tts_force_align": True,            # 9f1 post-TTS forced alignment
    "tts_align_model": "mms_fa",
    "attach_visemes_stage": True,       # 9f1b viseme extraction
    "viseme_backend": "mms_char",
}

#: Segment-wise pacing (9f2): each clip is fitted to its own slot in both
#: directions rather than the whole file getting one `atempo`. This is exactly
#: what `backend/cli.py::resolve_speed_setup` builds for `--speed-mode segment`,
#: copied deliberately rather than imported because the CLI resolver takes an
#: argparse namespace this path does not have.
#:
#: The three companion keys are NOT optional. `backend/tts.py` stretches each
#: clip to its target and the timing plan fits it again in the mix; leaving
#: `tts_apply_stretch` on would stretch every clip twice to the same slot.
UI_TIMING_DEFAULTS: Dict[str, Any] = {
    "timing_producer": "naive_slot",
    "timing_params": {
        "max_tempo": 1.5,
        "min_tempo": 1.0 / 1.5,
        "safety_margin_s": 0.0,         # any margin is itself a pause
        "use_target_duration": True,    # honour stage 9c / speed edits
    },
    "tts_apply_stretch": False,
    "mix_strategy": "timing_plan",
    "mix_clip_source": "raw",
}


def build_kwargs(preset: str, *, target_language: str,
                 tts: Optional[str] = None,
                 lipsync_model: Optional[str] = None,
                 num_speakers: Optional[int] = None,
                 skip_stems: Optional[bool] = None,
                 restore_model: Optional[str] = None) -> Dict[str, Any]:
    """The `run_pipeline` kwargs for one submission.

    `lipsync_model` is the *resolved* model: `"auto"` is not a pipeline value,
    so `ui/runner.py` substitutes the face-picked model before calling. A
    `None` here means no lip-sync, whatever the preset said.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of "
                         f"{', '.join(PRESET_IDS)}")
    spec = PRESETS[preset]
    engine = (tts or default_tts()).lower()

    # "Will there be lip-sync" and "which model" are two different questions,
    # and conflating them silently dropped restoration: with `standard` the
    # model is deferred to the face probe, so `lipsync_model` is absent here
    # — but restoration still happens, and keying it off the model meant it
    # was dropped for exactly the preset that is supposed to have it.
    # Three distinct inputs, and collapsing any two of them is a bug:
    #   "off"  -> no lip-sync, whatever the preset says
    #   None   -> unspecified; take the preset's choice
    #   "auto" -> lip-sync, model deferred to the face probe
    # `None` used to be the only way to say "no lip-sync", which made it
    # impossible to turn lip-sync OFF on a preset that has it on: the form's
    # "Off" option resolved back to the preset's `auto`.
    if (lipsync_model or "").lower() == "off":
        requested = None
    elif lipsync_model is not None:
        requested = lipsync_model
    else:
        requested = spec["lipsync"]
    wants_lipsync = requested is not None
    ls = None if requested == "auto" else requested

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
        # Code-mixing is the MT API's own job, not a second pass.
        #
        # This used to ask for `native` and then run stage 9b (the
        # transliterator) to put the English terms back — a detour that cost
        # an extra Bodhan call per retained span and could disagree with the
        # translation it was editing. `codemix` is `bodhan_adapter`'s own
        # default for a measured reason: `native` re-scripts inconsistently
        # (the same "WT" came back three different ways inside half a minute
        # on one clip), and a bare Latin remnant is what an Indic TTS voice
        # then mispronounces.
        #
        # `translit_model` is therefore absent, not None — `plan_stages`
        # keys off presence, so stage 9b is simply not part of the run.
        "mt_target_script": "codemix",
        "duration_adjust": True,
        # saying it
        "voice_engine": voice_engine_for(engine),
        "tts_model": engine,
        "mix": True,
        "mux_video": True,
    }
    # Voice conversion (9f) is deliberately absent. The adapters are here but
    # neither upstream is installed on this host -- `third_party/OpenVoice` and
    # `third_party/Amphion` are both missing, so `get_vc(...)` raises on load.
    # It is also an ALTERNATIVE to stage 9d, not an addition: VC re-voices the
    # output and overrides the stock voice just assigned. Enable it only with
    # a deliberate `vc_model`, once a repo exists to run it.
    kw.update(UI_STAGE_DEFAULTS)
    kw.update(UI_TIMING_DEFAULTS)
    if wants_lipsync:
        kw["face_detector"] = "retinaface"
        # Opt-in only: the caller's explicit choice, else the preset's (which
        # is None for every preset). Never a silent default.
        kw["restore_model"] = (restore_model if restore_model is not None
                               else spec["restore"])
        if ls:
            kw["lipsync_model"] = ls
        # No `lipsync_model` key when the choice is deferred: `ui/runner.py`
        # sets it after the face probe. Writing None here would read as
        # "lip-sync is off" to `plan_stages` and to the pipeline alike.
    return kw


def plan_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Manifest-shaped config, which is what `progress.plan_stages` reads."""
    return {
        "diarization":     kwargs.get("diarization_model"),
        "asr":             kwargs.get("asr_model"),
        "align":           kwargs.get("align_model"),
        "mt":              kwargs.get("mt_model"),
        "target_language": kwargs.get("mt_target_language"),
        "translit":        kwargs.get("translit_model"),
        "duration_adjust": kwargs.get("duration_adjust"),
        "voice_engine":    kwargs.get("voice_engine"),
        "tts":             kwargs.get("tts_model"),
        "vc":              kwargs.get("vc_model"),
        "face_detector":   kwargs.get("face_detector"),
        "asd":             kwargs.get("asd_model"),
        "lipsync":         kwargs.get("lipsync_model"),
        "restore":         kwargs.get("restore_model"),
        "mix":             kwargs.get("mix"),
        "skip_stems":      kwargs.get("skip_stems"),
        "timing_producer": kwargs.get("timing_producer"),
        # `plan_stages` keys off these; without them the newly-enabled stages
        # run but are not planned, so the bar counts "12 of 14" and then
        # appends four unplanned nodes as they arrive.
        "resegment_turns": kwargs.get("resegment_turns"),
        "refine_speakers": kwargs.get("refine_speaker_labels"),
        "tts_style":       kwargs.get("tts_style"),
        "emotion":         kwargs.get("emotion_model"),
        "tts_force_align": kwargs.get("tts_force_align"),
        "visemes":         kwargs.get("attach_visemes_stage"),
    }


# ---------------------------------------------------------------------------
# estimates
# ---------------------------------------------------------------------------

def estimate_runtime(duration_s: float, *, lipsync_model: Optional[str] = None,
                     restore_model: Optional[str] = None) -> Dict[str, Any]:
    """`{low_s, high_s, text, exact}` for a clip of `duration_s`.

    `exact=False` always: the base band is not measured the way the lip-sync
    factors are, and lip-sync itself spans 8-21x on the same model. The text
    is written to be read as a range, not a promise.
    """
    d = max(0.0, float(duration_s or 0.0))
    low = max(BASE_FLOOR_S, d * BASE_FACTOR[0])
    high = max(BASE_FLOOR_S * 2, d * BASE_FACTOR[1])

    parts = ["the dub itself"]
    ls = (lipsync_model or "").lower()
    if ls in LIPSYNC_FACTORS:
        lo, hi = LIPSYNC_FACTORS[ls]
        low += d * lo
        high += d * hi
        parts.append(f"{ls} at {lo:g}-{hi:g}x")
    rs = (restore_model or "").lower()
    if rs in RESTORE_FACTORS:
        lo, hi = RESTORE_FACTORS[rs]
        low += d * lo
        high += d * hi
        parts.append(f"{rs} at a further ~13x")

    return {
        "low_s": int(round(low)),
        "high_s": int(round(high)),
        "text": f"about {human_duration(low)} to {human_duration(high)}",
        "basis": "Measured on this host: " + ", ".join(parts) + ".",
        "exact": False,
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
