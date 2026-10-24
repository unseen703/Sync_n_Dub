"""Three ways to run the pipeline, and what each one costs.

`backend.pipeline._run_pipeline` takes about sixty keyword arguments. A form
that exposed them would be a worse CLI; a form that exposed none of them
could not produce a lip-synced dub. So the UI offers three named points on
that surface and sets **only** the arguments it means — everything else keeps
the pipeline's own default, which is where the knowledge about it lives.

    draft     no lip-sync. Audio dub only, stems skipped. Minutes, not hours;
              this is the "is the translation any good" run.
    standard  lip-sync with the auto-picked model.
    best      MuseTalk, regardless of face size.

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
latentsync ~83x. The non-lip-sync part of the
pipeline is NOT measured that way — the range below is a rough band and is
labelled as such, because quoting a precise number nobody measured is how a
UI teaches people to distrust it.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from backend import speed as _speed
from . import lipsync_models as _ls_models

#: Measured wall time as a multiple of clip length, on this host's 2x16 GiB
#: Blackwell cards. From `.claude/skills/clip-run`.
#:
#: Now DERIVED from `ui/lipsync_models.py` rather than written out again.
#: This table and three others each described the same five models, and
#: `wav2lip_hd` reached none of them — see that module's docstring. Kept as a
#: name because `routes_create` and the tests read it.
LIPSYNC_FACTORS: Dict[str, Tuple[float, float]] = {
    k: v["rate"] for k, v in _ls_models.FACTS.items() if v.get("rate")
}

#: Everything before lip-sync — download, stems, diarization, ASR, MT, TTS,
#: mix. Not measured as a factor the way the GPU stages were; this is a band
#: wide enough to be honest and is reported as approximate.
BASE_FACTOR: Tuple[float, float] = (1.5, 5.0)
BASE_FLOOR_S: float = 60.0          # model load + download dominate short clips

#: ONE flow, as of 2026-10-20. Every run dubs the audio and stops; lip-sync
#: is a separate, explicit second step against a model the operator names.
#:
#: The three presets this replaces differed in exactly two things — whether
#: stems were separated, and what happened at lip-sync — and the second of
#: those is no longer a submit-time decision at all:
#:
#:   draft     lipsync=None,      skip_stems=True
#:   standard  lipsync="auto",    skip_stems=False
#:   best      lipsync="musetalk" skip_stems=False
#:
#: `"auto"` is gone with them. It resolved the model from a face-size probe,
#: which meant the most expensive stage in the pipeline was chosen by a
#: measurement the operator never saw. The measurement is still taken and is
#: now SHOWN (`GET /api/jobs/{id}/lipsync/readiness`); the choice is the
#: operator's, and `POST .../lipsync` refuses without one rather than
#: guessing.
PRESET_ID = "dub"

#: Old ids keep working: a browser holding the previous bundle still posts
#: `preset: "standard"`, and every job already on disk names one of these.
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
        # Stems are separated now where `draft` skipped them. Draft skipped
        # them for speed on a throwaway pass; this is the only pass there is,
        # and its mix is what the review judges.
        "skip_stems": False,
    },
}


def resolve_preset(name: Optional[str]) -> str:
    """Map any accepted preset id — including the retired ones — onto the
    single flow. Unknown ids still raise, so a typo is not silently dubbed."""
    key = (name or PRESET_ID).strip().lower()
    key = PRESET_ALIASES.get(key, key)
    if key not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; expected {PRESET_ID}")
    return key

def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip()


def default_source_language() -> Optional[str]:
    """What the UI assumes is SPOKEN in the source, unless told otherwise.

    English, because the direction this project dubs in is English -> Indic.
    `DUBBING_UI_SOURCE_LANGUAGE=auto` restores per-turn detection, which is
    what this used to do implicitly and which produced one lecture
    transcribed into four different Indic scripts (see
    `models/asr/bodhan_adapter.py`).
    """
    v = _env("DUBBING_UI_SOURCE_LANGUAGE", "en").lower()
    return None if v in ("auto", "detect", "none", "") else v


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
    # 2. NO `render_mode`. The hosted indic-transcribe endpoint REFUSES it:
    #    `HTTP 422 unexpected field(s): ['render_mode']`, which fails the
    #    whole job at stage 7. It was set to "mixed" here on the reasoning
    #    that this transcript is MT input and `native` would re-script
    #    "weight" to "वेट" before Bodhan MT ever saw it — sound reasoning
    #    for a knob the API does not have. Whatever the server defaults to
    #    is what every UI run has ever used, because the field was never
    #    accepted. Re-add it only after one rated call returns 200.
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
#: What a UI submission adds to the pipeline kwargs to pace the dub.
#:
#: This was a hand-kept COPY of what `backend.cli.resolve_speed_setup` builds
#: for `--speed-mode segment`, with `tests/test_speed_mode.py` checking the
#: two still agreed. Both sides call `backend.speed` now, so the UI and a
#: bare CLI run are the same values by construction. Taking no arguments is
#: the point: the UI does not expose these bounds, so it gets the same
#: defaults `--speed-cap` / `--speed-floor` carry.
UI_TIMING_DEFAULTS: Dict[str, Any] = _speed.segment_timing_kwargs()


def build_kwargs(preset: str, *, target_language: str,
                 tts: Optional[str] = None,
                 lipsync_model: Optional[str] = None,
                 num_speakers: Optional[int] = None,
                 skip_stems: Optional[bool] = None,
                 style: Optional[str] = None,
                 source_language: Optional[str] = None) -> Dict[str, Any]:
    """The `run_pipeline` kwargs for one submission.

    `lipsync_model` is the *resolved* model: `"auto"` is not a pipeline value,
    so `ui/runner.py` substitutes the face-picked model before calling. A
    `None` here means no lip-sync, whatever the preset said.
    """
    spec = PRESETS[resolve_preset(preset)]
    engine = (tts or default_tts()).lower()

    # NO lip-sync here, ever. Phase one dubs the audio and stops; the model
    # is named at `POST /api/jobs/{id}/lipsync` by the operator, against the
    # face size this run measured. `lipsync_model` is still accepted on the
    # submit schema so the older bundle does not 422, and is ignored.
    #
    # Stage 5 (faces) and 6 (ASD) run ANYWAY, which is new. They used to be
    # gated on wanting lip-sync, but their output is what the review screen
    # reads — the average size of the face ASD attributes to each speaker —
    # and `assignments.json` also feeds the gender vote (CLAUDE.md §4) and
    # the speaking-face targeting a later lip-sync uses. Gating them on a
    # decision that has not been made yet is what made the measurement
    # invisible in the first place.
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
        # 9c2, ON for every UI run since 2026-10-22 by request. The CLI
        # leaves it off; the UI does not, because a UI run has nobody
        # watching stage 9c's overrun warnings scroll past and the first
        # sign of the problem is a finished dub that drifts off the mouth.
        #
        # It only fires past `condense_speed_cap`, so an ordinary run where
        # the tempo stretch can absorb the overrun costs ZERO LLM calls --
        # the gate is evaluated from 9c's estimates before any request is
        # made. Measured on svwrTOm55y4 (50.9s of Hindi for 30.6s of slots,
        # 1.66x against a 1.5x cap): 3 of 5 segments clamped and the dub ran
        # ~8s long, which is exactly the case this catches.
        #
        # Requires `duration_adjust` above -- it reads 9c's per-segment
        # estimates -- and OPENROUTER_API_KEY. Without the key the stage
        # logs and leaves every line as it was; it never fails a dub.
        "condense_overlong_text": True,
        # 1.5, not the 1.8 the mixer is allowed to reach. See
        # `backend/condense.py`: those are two different questions.
        "condense_speed_cap": 1.5,
        # saying it
        "voice_engine": voice_engine_for(engine),
        "tts_model": engine,
        "mix": True,
        "mux_video": True,
        # Always, not only when lip-sync is wanted — see the note above.
        "face_detector": "retinaface",
        # Recorded at SUBMIT even though lip-sync happens later, because the
        # replay that renders it rebuilds its kwargs from this job's manifest
        # config. Collapsing the presets moved this inside a `wants_lipsync`
        # branch that no longer runs, which silently handed the adapter
        # MuseTalk's own default of 20 — measured at a 15496 MiB peak on a
        # 16311 MiB card, i.e. 815 MiB of headroom and an OOM as soon as the
        # ~460 MiB onnxruntime CUDA EP is resident after face detection.
        #
        # Batch size moves throughput and VRAM only, not a single output
        # pixel: `datagen` is a pure chunker, each frame pairs with its own
        # latent by frame index, and the loop is `@torch.no_grad()` with the
        # models in `.eval()`.
        #
        # Named parameter, NOT the shared `batch_size` adapter kwarg: stage 4
        # picks that one up too and would hand it to the diarizer.
        #
        # 8 since 2026-10-20, tracking the adapter default. This box is
        # shared and a neighbour holding ~4 GiB is enough to push a batch-10
        # render past the 16311 MiB ceiling, where MuseTalk fails silently.
        "lipsync_batch_size": 8,
    }
    # Voice conversion (9f) is deliberately absent. The adapters are here but
    # neither upstream is installed on this host -- `third_party/OpenVoice` and
    # `third_party/Amphion` are both missing, so `get_vc(...)` raises on load.
    # It is also an ALTERNATIVE to stage 9d, not an addition: VC re-voices the
    # output and overrides the stock voice just assigned. Enable it only with
    # a deliberate `vc_model`, once a repo exists to run it.
    kw.update(UI_STAGE_DEFAULTS)
    # An explicit delivery type from the form REPLACES the `"auto"` default,
    # which is stage 9b1 classifying the register from the video's own
    # metadata. Choosing one is the operator saying they already know, so
    # 9b1 is not planned and not run — `plan_stages` keys 9b1 off
    # `tts_style == "auto"`, so the progress bar follows automatically.
    if style:
        kw["tts_style"] = style
    # Stated beats the install default, which beats detection. An explicit
    # "auto" from the form reaches here as the string, and means detect.
    if (source_language or "").lower() in ("auto", "detect"):
        kw["asr_language"] = None
    else:
        kw["asr_language"] = source_language or default_source_language()
    kw.update(UI_TIMING_DEFAULTS)
    return kw


def plan_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Manifest-shaped config.

    Read by `progress.plan_stages` (which stages to draw) AND, since
    2026-10-22, by `backend.edit_agent.rerun` (how to re-pace a replay).

    The second reader is why the pacing block at the bottom exists.
    `rerun` recovers a run's timing setup by re-parsing its argv — but a UI
    job HAS no pipeline argv: `ui/runner.py` is invoked with the job id and
    nothing else, so `recover_args` gets `['<job_id>']`, the pipeline parser
    exits on it, and the replay fell through to CLI DEFAULTS. Those defaults
    are not this submission's settings: `timing_params` became None (so
    `naive_slot` ran with `use_target_duration=False`), `mix_strategy` became
    `global_tempo` and `tts_apply_stretch` became True.

    `use_target_duration=False` is the one that bites, because
    `backend/edits.py::set_speed` works ENTIRELY by writing
    `target_duration_s` — `backend/timing.py` says so in its own docstring.
    With it off, the shortened slot was ignored, every clip was re-fitted to
    its original `end - start`, and a speed edit applied, re-rendered and
    came back at exactly the original speed.
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
        "vc":              kwargs.get("vc_model"),
        "face_detector":   kwargs.get("face_detector"),
        "asd":             kwargs.get("asd_model"),
        "lipsync":         kwargs.get("lipsync_model"),
        # 9i is never part of a submission -- there is nothing to restore
        # until a render exists, so the UI reaches it through
        # `POST .../enhance` on a finished job. Recorded as None so the key
        # exists and `plan_stages` can read it either way.
        "restore":         kwargs.get("restore_model"),
        "mix":             kwargs.get("mix"),
        "skip_stems":      kwargs.get("skip_stems"),
        "timing_producer": kwargs.get("timing_producer"),
        # `plan_stages` keys off these; without them the newly-enabled stages
        # run but are not planned, so the bar counts "12 of 14" and then
        # appends four unplanned nodes as they arrive.
        "resegment_turns": kwargs.get("resegment_turns"),
        "refine_speakers": kwargs.get("refine_speaker_labels"),
        # 9c2. `plan_stages` reads this name together with `duration_adjust`;
        # without it the stage runs and arrives as an UNPLANNED node, which
        # is the "12 of 14 then four extra" shape the comment above describes.
        "condense_overlong_text": kwargs.get("condense_overlong_text"),
        "tts_style":       kwargs.get("tts_style"),
        "emotion":         kwargs.get("emotion_model"),
        "tts_force_align": kwargs.get("tts_force_align"),
        "visemes":         kwargs.get("attach_visemes_stage"),
        # -- how this run was PACED, for the replay --------------------------
        # Not read by `plan_stages`; read by `edit_agent._setup_from_config`
        # when there is no argv to re-parse. Every key here is one the replay
        # would otherwise take from a CLI default that this run never used.
        # `tests/test_ui_backend_sync.py` asserts the two sides agree.
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

    `exact=False` always: the base band is not measured the way the lip-sync
    factors are, and lip-sync itself spans 8-21x on the same model. The text
    is written to be read as a range, not a promise.

    **`lipsync_measured` is the load-bearing field.** A model with no band
    used to fall through silently and return the dub-only estimate as an
    ordinary "about X to Y" — so an un-timed model advertised as the FASTEST
    option on the list, which is how `wav2lip_hd` came to look cheaper than
    plain Wav2Lip. The caller is now told, and `text` says so itself, because
    the readiness panel renders `text` and nothing else.
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
        # Never a bare number for a model nobody timed. `ui/cost.py` already
        # warns by name in this case; this is the same honesty one layer up.
        text = (f"{text} for the dub, plus the {ls or lipsync_model} render "
                f"— not timed on this host")

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
