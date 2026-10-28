# Stage internals and conventions (long form)

Moved out of CLAUDE.md on 2026-10-24. CLAUDE.md keeps the one-page summary; this
file holds the long form.

---


**One run, one job folder.** Everything a run produces goes under
`data/jobs/<job_id>/` with a `manifest.json` inventory. Nothing is written
outside it. Layout detail: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

**Every version of the merged JSON is kept, and none may look like the live
one.** Added 2026-10-22. The merged ASR+MT JSON is rewritten IN PLACE by
9b, 9b2, 9c, 9c2, 9e and every applied edit, so the original translation
stopped existing the moment the second stage ran — the per-segment
`edit_backup` / `condense_backup` keys are a diff, not a document.
`backend/mt_versions.py::snapshot` now copies it to
`mt/versions/<stem>.v<NNN>.<producer>.json` at each writer.

**The naming is the delicate half, and it is a trap in waiting rather than
a bug today.** `edit_agent._find_merged_json` globs `mt/*.mt.json` and,
when several match, takes **the newest by mtime** with only a log line. A
snapshot is newer than the live file by construction — so a version left in
`mt/`, or named to end in `.mt.json`, would quietly become the document
every later edit is applied to. Hence: its own subdirectory (`glob` is not
recursive), a `.<producer>.json` suffix, and `mt` as a *reserved* producer
name. Numbers are zero-padded to three digits because `v10` sorts before
`v9` unpadded. A snapshot never overwrites; an applied edit's producer
carries the same `NNN` as its `edits/edit_NNN.json` entry.

`tests/test_mt_versions.py` greps every stage that writes the document and
fails if one of them stops snapshotting — a stage added later that writes
and does not record loses a version in silence.

**A download keeps the uploader's metadata, not just the media.** Added
2026-10-17: `backend/downloader.py::video_metadata()` lifts 18 fields off
yt-dlp's info dict — `categories`, `description`, `tags`, `chapters`,
`channel`, `language` — and both download paths persist them.
`download_youtube` writes `<stem>.meta.json` beside the video (suppress with
`write_metadata=False`, or collect in memory via `metadata_out={}`);
`scripts/fetch_corpus.py` nests the same dict under `meta.json`'s
`source_metadata`, importing the one helper so the two cannot drift.

The probe always returned this and both paths used to discard it. Re-fetching
later is another API call, another chance at a 429, and no guarantee the video
is still public or the description unedited — the IIT Madras GAN lecture is
already `availability: unlisted`.

`categories` is the useful one: YouTube's own 15-way bucket, which maps onto a
TTS register (`Education` → `educational lecture`) without a model. **But
`description`, `title` and `tags` are uploader-authored.** Matching keywords
against them is safe; pasting one into an LLM prompt is a live injection, so
they are passed as delimited data with the answer constrained to a known enum.

**Adding a model is one adapter plus one registry line** in
`models/<stage>/__init__.py`. The CLI picks up the new choice automatically.
`models/lipsync/` is the clearest worked example of the pattern: four
adapters behind one `base.py`, three of them shelling into their own conda
env via `<MODEL>_PYTHON`.

### Canonical implementations — find these, do not write them again

De-duplicated 2026-10-23. Each row is the ONE place a thing is written; the
bracketed count is how many copies it replaced. A new caller **imports the
canonical one**. A new adapter, registry or route **inherits** rather than
pastes. If one of these does not fit a new case, widen it — a second copy is
how every row here started.

| Thing | Lives in | Was |
|---|---|---|
| adapter registry: lazy load, lister, kwarg filter, `get_*` | `models/_registry.py::make_registry` | 12x |
| Bodhan auth, env resolution, pacing floor, `Retry-After` backoff | `models/_bodhan.py::BodhanClientMixin` | 4x |
| ffprobe → `(fps, duration)` | `backend/media_io.py::probe_video` | 4x |
| media duration (ffprobe, then ffmpeg's banner) | `backend/media_io.py::probe_duration` | 2x |
| `--speed-mode segment` knobs and their bounds | `backend/speed.py` | 2x |
| best-effort JSON read of a job file | `ui/_io.py::read_json` | 4x |
| the lip-sync model ids the edit agent advertises | `models.lipsync.available_models()`, via `edits._lipsync_model_ids` | hand-typed |

Four things about this that are load-bearing rather than tidy:

- **`_REGISTRY_SPEC` stays a dict LITERAL at module level in every stage.**
  `scripts/repo_checkup.py::_registry` reads it with `ast.parse` and never
  imports the module, so the five-hourly README regeneration can run in an
  env where half the adapters will not import. Build that dict from parts,
  or move it behind a helper, and the structure table silently reports zero
  adapters for that stage.
- **`models/_registry.py` reproduces the error strings character for
  character, nouns included** ("Unknown aligner …", but "Aligner 'mms_fa' is
  not installed …"). They are what an operator reads when a `--lipsync-model`
  typo reaches the CLI. `scripts/ab_refactor.py` pins every one of them.
- **The Bodhan retry LOOPS are deliberately NOT shared.** ASR posts
  multipart and diagnoses a `422 + render_mode`; TTS posts JSON, returns
  bytes and diagnoses a `422 + rate field`; three of the four fail fast on a
  repeated 5xx body and transliteration does not. Folding those together
  means dropping an asymmetry or encoding it as a flag, and **no test in
  this repo exercises the Bodhan HTTP path at all** — so the loops stay
  where their special cases are readable. The constructor-time half is what
  moved.
- **`backend/speed.py` is dependency-free on purpose.** `ui/presets.py` is
  imported when the web server starts and pulls in nothing heavier than `os`
  and `typing`; routing it through `backend.cli` would drag argparse and all
  seven model registries into that import.

What was explicitly left alone, and why, so it is not re-proposed as an
oversight: `edit_agent._SYSTEM` is still hand-written against `OPS` (its
prompt is measured, and the grep test already catches an omission);
`_find_merged_json` still has three copies (CLAUDE.md §2 calls its naming a
trap in waiting, and merging them means picking one glob); the 13
`scripts/run_*.sh` still repeat their preamble (it doubles as the record of
what that run did).

**There is no clip-window flag, with one exception.** Running a *stage* on a
time range means cutting video and audio with ffmpeg first — see the `clip-run`
skill for the recipe. The exception, added 2026-10-16:
`run_lipsync(..., window=(start, end))` renders only that slice and splices it
back into the full timeline itself (measured overshoot: one frame, and it does
not grow with the number of joins). It is reached from
`backend/pipeline.py::run_post_mt_stages(lipsync_window=…)` and from the edit
agent below. Nothing else takes a window.

**The dub is re-paced one of two ways, and `segment` — the timing plan — is
the default as of 2026-10-22.** `--speed-mode segment` fits each clip to its
own slot in both directions: a short line is stretched out to fill its slot
rather than leaving a pause, a long one is compressed. `--speed-mode global`
(the previous default) places every clip at natural pace and applies ONE
`atempo` to the whole file.

**The bounds are ASYMMETRIC, which is new and deliberate.** `--speed-cap`
(default **1.8**, raised from 1.5) is the speed-UP bound; `--speed-floor`
(default **0.75**) is the slow-DOWN bound. They used to be one knob, with
`min_tempo = 1/cap` — at 1.8 that would slow a short line to 0.56x, nearly
double length, which is audible as drawl. A floor that cannot absorb the gap
leaves a real pause, reported as `unfilled_slot_s` in the timing plan; `0.0`
is the checkable form of "no pauses left". `--speed-slot-margin` still
defaults to 0.0 because any margin is itself a pause.

**The shift-forward cursor used to put speech on top of the music, two ways.**
Both fixed 2026-10-22 in `plan_naive_slot`, both on by default under
`--speed-mode segment` and in the UI.

1. **A cue nested in another cue's slot is SIMULTANEOUS SPEECH**, and a
   monotone cursor cannot say so — it pushes the nested cue past the turn it
   overlaps. On job `20261022-044302` a 0.35 s backchannel (26.22–26.58,
   inside a 23.93–30.73 turn) landed at ~32.2 s, in the middle of 7.2 s of
   music. `overlap_policy="keep"` places it at its own slot start and does
   **not** advance the cursor. The mix already renders this — `_ffmpeg_mix_plan`
   is one `adelay` per clip into `amix` and has never required non-overlap —
   so this is a planner-only change. It also makes the plan honest:
   `overlap_segments` read `0` when the overlap had been *displaced*, not
   when there was none.
2. **A long source gap is a wall** (`gap_barrier_s`, `--speed-gap-barrier`,
   default 1.0 s). A cue that drifted late has its target shortened to what
   is left before its slot ends, so it speeds up within `max_tempo` instead
   of running on over the gap. Clamping it resets the cursor for free, so the
   next cue starts at its own slot rather than inheriting the drift. What the
   cap still cannot absorb is reported as `qc["overrun_into_gap_s"]`; the clip
   is never truncated, because losing a line is worse than a short bleed.

The threshold has a flag and the policy does not, deliberately: what counts
as "a gap" is material, but serialising simultaneous speech was never a
choice anyone made.

**The cap alone cannot make a cue fit, and that limit is now visible rather
than hidden.** On that job three cues needed 1.97/2.09/2.17x — past anything
intelligible in Hindi. They stay pinned at 1.8 and report the overrun. The
real lever is shortening the text (`backend/condense.py`,
`--condense-overlong`), on for every UI run since 2026-10-22.

**Why the default moved**, since it changes the output of every CLI run:
`global_tempo` met a +4.86 s TTS overrun on job `20261021-114009` by
applying `global_tempo: 1.0` — no compression at all — so 35.50 s of dub was
laid over a 37 s video and every line landed progressively later than the
mouth saying it (dub 87% voiced against the source's 66%). The timing plan
was even BUILT on that run and then ignored by the mixer, which is exactly
what `pipeline.py:954` warns about. 1.5 was no headroom either: Hindi TTS
runs 1.21–1.65x longer than the English it covers, and 4 of that job's 5
segments overran at 1.22 / 1.17 / 1.31.

**`--speed-mode` parses to `None`, not to a literal**, and the distinction
is load-bearing. `resolve_speed_setup` refuses `segment` together with
`--timing-method` because they set the same four knobs — so a literal
`"segment"` default would have killed every research-arm invocation at
startup on a collision with a default nobody typed. None lets the resolver
tell "the operator asked" from "nobody said"; only the first collides, and
`--timing-method` implies `global`. The two guards in `main` stay keyed on
the explicit flag for the same reason: refusing a transcribe-only run over
an unasked-for default would be a regression.

`ui/presets.py::UI_TIMING_DEFAULTS` was a deliberate COPY of what the
resolver builds (the CLI resolver takes an argparse namespace the UI path
does not have), merged unconditionally at `build_kwargs`. Since 2026-10-23
it is `backend.speed.segment_timing_kwargs()` — both sides call the one
function, so they agree by construction rather than by a test. The UI has
been on the timing plan since before this change; only its bounds moved.
`tests/test_speed_mode.py` still asserts the two sides match, now as the
guard against someone reintroducing a second copy or giving the UI its own
bounds without routing them through `backend/speed.py`.

Both resolve through `backend/cli.py::resolve_speed_setup`, which is also what
`backend/edit_agent.py` calls to replay a run's pacing. The raw argparse
namespace does not describe how a run was paced — `--speed-mode` and
`--timing-method` both override `args.mix_strategy` / `args.no_tts_stretch`
after parsing, so `resolve_speed_setup` is the accurate source.

**There is an operator UI, and it is a second entry point into the pipeline.**
`ui/server.py` (FastAPI, loopback-only, port 7861) serves a built Vite app from
`ui/web-dist/` (API-only, with a warning, when no build exists). Jobs submitted from it are launched DETACHED through
`scripts/ui_run_job.sh`, which is the only launcher that exports the
`libstdc++` preload — starting `ui.runner` with a bare `Popen` walks stage 9h
into the cv2/insightface import-order trap in section 1.

**Speaker preview before casting.** The voices stage writes
`voices/<stem>.refs/*.wav` only after the cast is chosen, so a job waiting on
the cast has no reference clips. `ui.speaker_media.speaker_preview_paths` cuts
each speaker's longest solo stretch (max 8 s) from the vocals stem into
`voices/previews/` so the cast row can play it. It is kept out of `*.refs`
on purpose; the pipeline must never mistake a preview for its own reference.

Three things about it are easy to get wrong:

- **`include_router` is called ABOVE `app.mount("/", StaticFiles(...))`.**
  Starlette matches in registration order, so a router added after the static
  mount is shadowed completely: every GET answers a bare 404 and every POST a
  405, served by the static handler. Introspecting `app.routes` does not
  reveal this — included routers appear as one opaque `_IncludedRouter` entry
  whether or not they work. The honest check is to call them, which
  `tests/test_ui_server.py` does.
- **The progress bar is EIGHT pills, always, on every run — seven named
  stages plus `Finished`.** Changed 2026-10-22 by request. `PHASE_GROUPS`
  in `ui/progress.py` folds the 24 markers into Download video · Speaker
  annotation · Transcribe · Translate · Speech generation · Mix · Lip sync,
  and `_group_phases` appends `Finished`. A stage marker added to
  `pipeline.py` therefore also belongs in a group (or in `HIDDEN_KEYS`, for
  a QA pass that produces no part of the dub), or it surfaces as a nameless
  extra segment. The tests fail loudly on an omission — which is how stages
  `9b1` and `9b2` were caught the day they were added.

  **The length no longer depends on the config, and that was the bug.**
  `_group_phases` used to `continue` past a group with no planned
  substages, and `plan_stages` is config-driven — so a run that enabled few
  optional stages drew a bar with one or two pills on it, and the same
  video drew a different-length bar on two runs. An operator cannot tell
  "this pipeline has no transcribe step" from "the bar is broken". A group
  with nothing planned now reads **`skipped`**, not absent and not
  `pending`: the pipeline has that step, this run is not taking it, and
  `pending` would leave a finished run looking unfinished.

  **`Finished` is synthetic and is deliberately NOT in `PHASE_GROUPS`** —
  that table maps real `=== stage N ===` markers and
  `tests/test_ui_progress.py` greps `pipeline.py` for every key in it, so a
  group with no marker would fail that grep for doing its job. Two
  properties of the pill are load-bearing: its `wall_s` is **None**, because
  the ribbon sums `wall_s` across phases for elapsed time and a total here
  would report double the runtime; and it stays `pending` while any
  substage is `deferred`, so a draft run's un-run 9h cannot light it. The
  ribbon also excludes it from the "`<label>` not started" heading, which
  would otherwise read "Finished not started".

  **`plan_stages` appends `9h` unconditionally**, which is the one place it
  deliberately does not mirror `_run_pipeline`. A submission sets
  `lipsync: None` and stops at the mix (lip-sync is its own explicit act), so
  keying the node off the config drew a SIX-node bar reading "Finished · all
  6 stages" — a dub *waiting* for its most expensive stage, rendered
  identically to one that will never have it. The row carries
  `deferred: True`, and that flag is load-bearing in three places:
  `ProgressWriter.finish()` and `_from_manifest` both resolve a leftover
  `pending` to `skipped` on a successful run and must skip it, and
  `StageRibbon.isDone` counts `skipped` AS DONE — so "skipped" would have
  completed the bar for a dub nobody had lip-synced. The later
  `POST .../lipsync` opens that same row rather than appending an eighth,
  because `_open` reuses an existing row for a key.

  `StageRibbon` therefore derives "finished" from the NODES, not from
  `progress.state`. The run genuinely finished; the pipeline did not, and
  "Finished" over an unlit Lip sync node is the screen contradicting itself.
  A terminal job with a pending node heads `<label> not started`.
- **Progress is scraped from the pipeline's own `=== stage N ===` log lines**
  by a `logging.Handler` that `ui/runner.py` installs on the ROOT logger.
  `backend/` has no idea the UI exists, which is the point — but renaming a
  marker silently freezes the progress bar. `tests/test_ui_progress.py` greps
  `pipeline.py` for every marker literal and fails if one stops matching.
  The handler needs the root logger's LEVEL raised too: a handler never sees
  a record its logger filtered first, and root defaults to WARNING.
- **The UI never shows the runner log.** `ui/errors.py` turns a traceback
  into `{title, detail, action}` — "Missing Hugging Face credentials · Add
  HF_TOKEN to .env", "Ran out of GPU memory · Use a shorter section" — and
  `read_progress` attaches it as `progress.failure`. An unrecognised failure
  is **marked** as unrecognised rather than dressed up, and `raw` is capped
  at one line so the UI cannot drift back into streaming a log. The log stays
  at `GET /api/jobs/{id}/log` for whoever is debugging.
- **Tone options are the engine's own list, not a plausible one.**
  `ui/speakers.py::tone_vocabulary()` reads
  `models.tts.bodhan_adapter.INDIC_SPEAK_STYLES` — 14 values where case and
  spacing are significant and an unknown value is a **422, not a silent
  fallback**. This module first shipped an invented list ("formal",
  "casual", "polite"); every tone change would have failed at the API with
  nothing on screen to explain it. `value` is sent verbatim and `label` is
  what is displayed.

  **`models.tts.NEUTRAL_STYLE` ("neutral") leads both dropdowns and is the
  one exception**, added 2026-10-21. It is not an invented engine style: it
  never reaches the engine, because `models.tts.resolve_style` maps it to
  None and `bodhan_adapter._post_one` omits a None style from the payload.
  It exists because there was no way to take a dub OFF a register — every
  option in the picker was another register, and the new-job form's other
  non-engine value, `auto`, means *classify one from the metadata*, which is
  the opposite. It cannot be expressed by clearing `tts_style` either: None
  is what an unedited segment looks like, so the `segment or document`
  precedence in `backend/tts.py` would fall straight back to the register
  being dropped. Hence a sentinel that travels as itself and is resolved at
  the last hop. `--tts-style neutral` is the same thing on the CLI, and
  `_h_set_tone` skips the LLM rewrite for it — there is no register to
  rewrite a line *into*, and a paraphrase is a different length and so a
  different timing fit.
- **Gender is persisted now, so the UI never recomputes it.** Stage 9d
  predicts it with ECAPA and writes it into
  `voices/<stem>.speaker_voice_map.json` beside the voice. Re-deriving it in
  a request would mean loading a speaker-embedding model inside a GET.
  Voices carry `recommended_for` (speaker ids of matching gender) — a
  recommendation rather than a filter, because the prediction is a
  prediction.
- **Renaming a speaker writes to `<job>/speaker_labels.json`, not the merged
  JSON.** It is a display preference; putting it in the merged JSON would
  dirty a pipeline stage and cost a re-render for changing a caption.
- **Three job-folder shapes exist on disk** (`progress.json`, the clip
  runner's `STATUS.json`, a bare `manifest.json`) plus 16 folders with no
  status file at all. `ui/progress.py::read_progress` normalises all four,
  so nothing above it branches on which runner produced a job.

**Two things the UI refuses on purpose.** (Moved from README 2026-10-26.)

- **Lip-sync is capped at 20 minutes of video.** It runs at 3–21× clip
  length on two 16 GiB cards, so 20 minutes is already one to seven hours.
  An audio-only dub can be longer; the cap is on the lip-synced portion. The
  form shows an estimate before you start.
- **A voice change doesn't re-render until you say so.** It dirties voice
  assignment through TTS, mix and lip-sync, and 9e has no per-speaker
  incremental mode, so one speaker's new voice is a full re-synthesis. The
  dialog quotes what re-runs and what stays (download, diarization,
  transcription and translation are never at risk). The path never asks
  for a retranslate, so the quoted cost is the applied cost.

**The lip-sync model is chosen from the video.** The UI samples ~40
frames, takes the median largest-face height as a fraction of frame
height, and picks **wav2lip** below 25% (wide shots; its 96×96 mouth crop is
proportionate) or **MuseTalk** above (close-ups need 256×256). Measured: a
wide 9-face lecture scores 0.105, two close-ups 0.419 and 0.401. It says
what it picked and why, and the operator can override. The threshold is
calibrated on three clips from one source video, a sane default rather than
a validated classifier; `DUBBING_UI_FACE_THRESHOLD` moves it. A 4K source
is the case to override by hand: a 96×96 crop on a ~520 px face is upscaled
roughly 5×.

**Every hosted stage of a default UI run is Bodhan.** As of 2026-10-17
`ui/presets.py` defaults ASR, MT and TTS all to `bodhan` (ASR moved off
`whisperx` by request). That removes the one stage that used to run locally,
so an outage — which these endpoints answer as `internal_error` 5xx, not as
rate limiting — takes the whole run rather than most of it. Since 2026-10-28
there is no fallback at all, manual or automatic: the local adapters were
deleted, so Bodhan is the only name in the ASR, MT, translit and TTS
registries. ASR needs its own `BODHAN_ASR_KEY`, separate from the MT and TTS
keys.

**Code-mixing is the MT call's job, and stage 9b is not in a UI run.**
The UI asks for `mt_target_script="codemix"`, which is
`models/mt/bodhan_adapter.py`'s own default and chosen there on measured
grounds. It used to ask for `native` and then run the transliterator (9b) to
put the English terms back — an extra Bodhan call per retained span, editing
the translation it had just asked to be made differently. `translit_model`
is therefore ABSENT from `build_kwargs`, not None: `plan_stages` keys off
presence, so a None would still draw the stage on the progress bar.

Bodhan's `target_script` takes three values; `codemix` is the default in
the adapter (`DEFAULT_TARGET_SCRIPT`), the CLI (`--mt-target-script`) and
`scripts/clip_dub_audio.py` (`--target-script`):

| `target_script` | What comes back |
|---|---|
| `codemix` *(default)* | English technical terms stay in Latin inside the target-script sentence — `weight सही है` |
| `native` | Everything re-scripted into Devanagari/Tamil/… |
| `roman` | The whole sentence in Latin script |

**Why not `native`: it re-scripts inconsistently.** On one 30 s Hindi clip
of the NAG lecture, the same term came back three ways:

```
cue 0  →  जहां हमारे पास डेल्टा डब्ल्यूटी था       ("WT" transliterated)
cue 5  →  W के संबंध में लॉस फंक्शन                ("W" left in Latin)
cue 7  →  ब्रैकेट हमारे पास WT था                  ("WT" left in Latin)
```

An Indic TTS voice doesn't reliably pronounce a stray Latin remnant, so the
listener hears a dropped or mangled term. `codemix` keeps the Latin
deliberately and uniformly, which is also how Indian technical lectures are
actually spoken. On the same clip it gave clean Hinglish on every cue, and
`indic-speak` synthesised all 8 cues across the window.

For fully native script, pass `--target-script native` and clean the
remnants with `backend/codemix.py` (`--translit`, `--termbase`), exposed in
`scripts/clip_dub_audio.py` as `--normalize-script`. That's the opposite of
`codemix`; combining the two cancels out.

**One speaker keeps one voice, and the weak link is the LABEL, not the map.**
Diarization runs once over the whole audio, so pyannote's labels are already
global — `SPK_00` at 0:10 is the same cluster as `SPK_00` at 4:30. Stage 9d
builds `speaker_voice_map` keyed by speaker id and `backend/tts.py` resolves
every segment through `seg["speaker"]`, so one voice per speaker for the whole
video holds by construction. There is no code path that gives one speaker two
voices; only a per-segment `tts_voice` from `backend/edits.py` does that, by
design.

What is NOT guaranteed is that a label holds exactly one PERSON. On noisy
material pyannote under-clusters, several characters land in one label, that
label gets one voice, and the dub reads them all in it — with every artifact
looking correct. **Stage 4c** (`backend/speaker_id.py`, added 2026-10-17,
`--refine-speaker-labels`, on by default in the UI) splits a label whose turns
do not cohere and merges two labels that do. Audio only, and before face
detection, so the stage order stays acyclic — ASD's face tracks are a second
identity signal but only exist at stage 6, after this.

**Its thresholds are set against the control rather than against intuition.**
Short windows do not give the ~0.9 ECAPA cosine a full-clip comparison does.
Measured on one known speaker, clean lecture audio, 1.5 s windows: **0.45–0.55,
min 0.35** — that *is* the single-speaker ceiling at this length, and
`PURITY_FLOOR = 0.35` sits under it. Different speakers scored 0.007–0.141
between centroids. On a 279 s Dark Knight mob scene (6+ characters in 4 labels)
the pass took 4 labels to 11, and labels below the floor went 3-of-4 to 1-of-11.

Two behaviours found while building it:

1. **Absorbing a fragment into its "nearest sibling" always finds someone.**
   A sub-cluster too short to cast gets folded back, but ungated that undid
   the split it had just made — SPK_00 returned to 0.331 and SPK_02 to 0.274,
   below the floor they had just been cleaned past. `ABSORB_MIN_COSINE` now
   gates it: a bit-part resembling nobody present is a real speaker with thin
   evidence and keeps its own label (recorded in `extra.speaker_id.stranded`).
2. **A synthetic test fixture has to match the measured numbers.**
   At `dim=32` two random "speakers" sit at cosine 0.31 — nothing like the
   0.007–0.141 real ones scored — and the tests silently stop testing the
   split. `tests/test_speaker_id.py` uses ECAPA's real width (192) with
   orthogonalised identities and a jitter tuned to land a synthetic speaker at
   ~0.50, so the fixture tracks the host rather than the threshold tracking
   the fixture.

`extra.speaker_id` records purity per input label, the splits, the absorbed and
stranded fragments and the merges, so a finished run can be audited without
re-running the embeddings. `<job>/diarization/<stem>.<model>.refined.rttm` is
the relabelled turn list.

**There is a second page now: Talking Head (`#/talking-head`).** Added
2026-10-20. A still portrait plus a chosen line, rather than a video plus its
own soundtrack — `ui/talking_head.py`, `ui/routes_talkinghead.py`, and
`ui/web/src/components/talkinghead/`. Three gates in order: a photo WITH a
face, a language, a line.

**The lines are DATA, not code** — `data/talking_head/dialogues.json`, five
categories (slogan, inspiring, funny, catchy, famous) across 14 languages. An
entry without an `audio_url` is **offered and marked** rather than hidden, so
adding the link is the only step that makes it playable. No code change, no
deploy. (Every entry shipped that way; the 50 clips are in
`data/talking_head/audio/` since 2026-10-21 and no null is left. The rule
stays, for lines added later.) The language dropdown is
`ui/langs.options("bodhan")`, i.e. the voice catalog — not a list written on
the page, which is how the tone picker once shipped values the engine 422s
on.

Two measured constraints are encoded rather than guessed:

1. **A 4K still kills the render.** Measured 2026-10-19: a 3840x2160 portrait
   made MuseTalk's S3FD return `[None]`, and with zero detections
   `average_range_minus` is empty so a *print statement* at
   `preprocessing.py:135` divides by zero — after the model has loaded.
   `MAX_EDGE = 1280` downscales first and the probe says so, because the
   output resolution is then not what was uploaded. 1280x720 is the width
   verified end to end (7 s of audio → 7.0 s at 25 fps, 83 s wall).
2. **Only the mouth moves.** MuseTalk inpaints the occluded lower face and
   copies the rest through — measured 0.87% of the frame changing between
   output frames. No head motion, no blinking, no gaze. `ANIMATION_NOTE` is
   on the page itself: a user expecting SadTalker's head movement would read
   a correct render as a broken one.

The face check runs at UPLOAD, not at render: learning that a picture has no
face after a GPU job is queued is the expensive way to find out.

**The render actually renders, since 2026-10-21.** `POST .../render` was a
deliberate 501 while the catalogue had no clips, and the Generate button was
hard-coded `disabled`; both outlived their reason the day the 50 wavs
landed, and the page then looked broken rather than unfinished. It is a
background THREAD plus `GET .../render/{id}` to poll — the same shape
`ui/bbox_video.py` uses, and for the same reason: 8-21x the clip length is
minutes, long enough that an inline request dies to a browser timeout while
the GPU work carries on. Four things about it:

- **It still refuses two cases, and they are different.** A line whose
  `audio_url` is null (MuseTalk drives the mouth from a waveform; without one
  the output is the photo unchanged, and a job that returns a still is worse
  than a refusal naming the entry), and a line whose link points at a file
  that is not on disk. `playable` is computed from the URL being non-null, so
  those two are indistinguishable from the catalogue alone.
- **`render_id` is REJECTED, not sanitised**, when it is not
  `[A-Za-z0-9][A-Za-z0-9_-]{0,127}` — it arrives from the client on the poll
  and `Path / ".."` is not a string operation. A cleaned-up id would read a
  different render's status and show someone else's video as the answer.
- **The file is the arbiter**, in `render()` and again in `render_status()`.
  `rc=0` with no output is the Wav2Lip shape (§4), and a killed process
  leaves a stale `running` that would otherwise be polled forever.
- **`_preflight_gpu()` runs at the POST**, not at upload: that is the moment
  the VRAM is committed, same as `routes_edits.run_lipsync_now`.

**"insightface not found" on this page is a WRONG INTERPRETER, not a broken
install.** Seen 2026-10-21: `python -m ui.server` had been started from
`.venv-diarization` (a uv venv carrying torch and cv2 but not insightface)
instead of from `envs/dubbing`. The UI server is started by hand and has no
launcher wrapping it, so nothing pins the env the way `scripts/ui_run_job.sh`
does for jobs. `_detector` now separates `ImportError` from a detector that
loads and misbehaves, and names `sys.executable` plus the python to use —
`type(e).__name__` was the one fact that does not help, since
`ModuleNotFoundError` is equally true of a typo, a broken install and the
wrong env. Start it with
`/data1/dipak_data/miniconda3/envs/dubbing/bin/python -m ui.server`.

**A dub can be reviewed BEFORE the expensive stage, in three steps.**
Added 2026-10-19. The `draft` preset already dubbed without lip-syncing
(`lipsync: None`); what was missing was a way to inspect the result and then
finish the job. The flow is: review the cast (`GET .../speakers`), review the
cues (`GET .../transcript`), correct either in plain language (`POST
.../agent`, then `POST .../edits/apply`), and finally `POST .../lipsync`.

**All of it is on the JOB PAGE now, under the player.** Changed 2026-10-21.
The agent had its own route at `#/job/<id>/edit` reached from a top-bar
button, so reviewing a dub meant bouncing between two screens — the video on
one, the controls that change it on the other — and judging a voice meant
judging it from memory. `components/agent/EditPanel.tsx` is that page's
contents in one column under `VideoStage`: cast · chat box · script ·
lip-sync · prompts · actions. `useHashRoute` REDIRECTS the old URL rather
than 404-ing it, and `store.editing` is gone.

Two things went with the move:

- **The "What this can change" summary is deleted.** It listed the clip,
  language, lip-sync model and every speaker's voice and tone — a read-only
  second copy of what the cast table and the lip-sync panel already show,
  kept in step by hand. A staged change appears ON the thing it changes now:
  `components/ui/StagedValue.tsx` renders old-struck-through → new-in-blue,
  on the cast rows for gender/voice/tone and under the cue for a rewritten
  line.
- **`blue-ink` is its own token, not `brand-ink`.** Brand means
  "active/selected" everywhere else in this app, and a staged edit is the
  opposite claim — not yet true until Apply spends the GPU. `npm run
  contrast` checks it in both themes like every other ink.

**The cast table reads `/agent/params` as well as `/speakers`.** Two staging
paths reach the same row and only one used to show: the pickers write a
local store that never left the browser, while the chat box writes the job's
`pending.json`. Gender can ONLY arrive the second way — there is no picker
for it — so "make SPEAKER_00 female" changed nothing on screen until it was
applied and the dub came back in a different voice. `agent_params` now
returns `gender` as `{current, staged}` like voice and tone, read off the
staged `set_gender` op rather than resolved (the panel polls that endpoint,
so it must not load a catalog or a model).

Three pieces were added and each exists because of a specific silent failure:

- **`set_speaker`** (`backend/edits.py`, the 9th op) is the only op that
  touches `seg["speaker"]`. `backend/tts.py` resolves every segment's voice
  through that key, so a cue diarization mis-attributed is read in the wrong
  person's voice while every artifact looks correct — and no other op could
  fix it. It `dirty_from="voices"` rather than `"tts"` because moving the
  last cue off a speaker empties it and 9d re-derives the cast; that case is
  reported as a warning rather than letting the speaker list change in
  silence. It also CLEARS any per-segment `tts_voice`, which would otherwise
  keep reading the moved line in the old speaker's voice, since `tts.py`
  reads that key ahead of the map. An unknown target id is refused: a made-up
  speaker has no reference clip, no gender vote and no catalog voice, so TTS
  would fall back to the engine default.
- **`GET .../transcript`** (`ui/transcript.py`) is a READ of the merged JSON,
  paged. It carries source text AND dub text — a translation cannot be judged
  from one side — plus `tts_error` per cue, because a Bodhan outage leaves
  the stage green with every segment failed and that has to be visible
  somewhere.
- **`POST .../lipsync`** finishes a draft run. It does NOT add a second
  launch path: it stages a whole-video `lipsync_window` with an **explicit**
  model and goes through the same replay the edit agent uses, because
  `edit_agent.available_stages` adds back a stage the edit introduces. The
  model has to be explicit — `_h_lipsync_window` falls back to
  `ctx.lipsync_model`, which on a draft job is None, and a None model reaches
  the render as "no lip-sync", so the stage would appear to run and produce
  nothing. It refuses while edits are staged, since rendering around a
  pending edit would use the audio from before the review.

**The regex fallback parser could not name any voice that exists.** Its
`set_voice` pattern matched only `xx-XX-NameNeural`, the Azure/Edge shape
removed from the registry on 2026-10-17; `indic-speak` voices are bare names
like `Amit`. So "use Amit for SPEAKER_00" parsed to nothing whenever no
`OPENROUTER_API_KEY` was set — which is exactly when the fallback is the only
parser. Widened, with a stopword list so "use faster speech" stays a speed
edit rather than pinning a voice called "Faster".

**Stage 7 uses the uploader's transcript when there is a human-authored
one.** Added 2026-10-19, on by default. `backend/caption_asr.py` takes the
caption track `resolve_input` already fetches at stage 1 and reshapes it into
the exact `ASRResult` turn-sliced ASR produces, so nothing downstream can
tell which path ran. That skips a hosted Bodhan call per turn.

**Manual tracks only, and the distinction is the entire point.** yt-dlp
reports `manual` vs `automatic`, and an automatic track IS YouTube's own
ASR — taking it swaps the engine chosen here for an unconfigurable one on
Indic-accented English. `--captions-allow-auto` exists; nothing defaults to
it. `--no-captions-transcript` forces ASR.

Both gates are load-bearing, not theoretical. Measured on the W4L11 lecture
(`_IBfVkrvqAI`): `info.json`'s `subtitles` is EMPTY — every track is
auto-generated — so the manual gate alone sends it to ASR. Forced past that
with `allow_auto`, the quality gate catches **66% rolling duplication**,
YouTube's karaoke style where each cue restates the tail of the last; that
would have duplicated phrases through the whole dub. A thin-coverage floor
(60% of turns) catches forced captions on a few signs.

Three things worth knowing:

- **Cues are mapped onto diarization turns, one cue to its best-overlapping
  turn.** A cue breaks on line length, not on who is speaking, and carries
  no speaker. Everything downstream resolves through `seg["speaker"]`, so
  using cues as segments would read the whole dub in one voice with every
  artifact looking correct. Assigning a cue to *every* turn it touches
  would instead repeat the line once per speaker.
- **Stage 8 re-times the text**, which is why loose cue boundaries are
  survivable — forced alignment re-derives word timings from the audio.
- **Artifacts are written on the shared path.** `asr.json`,
  `transcript.txt` and `result.asr_*` are produced identically either way
  (tagged `captions` instead of the model name); they used to sit inside
  the ASR branch, which would have left a caption run with no transcript on
  disk while stage 8 aligned one.
- Language keys match by BASE code: manual English appears under `en-GB`,
  and in this corpus under the channel-suffixed `en-LUU0EuDKgKo`, where
  plain `en` is a machine translation of another track.

**A finished dub is edited in plain language, not re-run from scratch.**
`backend/edit_agent.py` turns a sentence into typed ops from
`backend/edits.py`, applies them to the run's merged ASR+MT JSON, and replays
only the stages the change invalidated — stages 1..9a are never re-run, because
that JSON is already their output. It is dry-run by default: `--apply` writes,
`--run` re-renders.

    $PY -m backend.edit_agent -j <job_id> "speed up 0:30 to 0:45 by 1.2x"
    $PY -m backend.edit_agent -j <job_id> --list-ops

Edits are reversible (`--revert`): every overwritten key is kept in the
segment's `edit_backup`, and each request is journalled into `extra.edits` and
`<job>/edits/edit_NNN.json`. The ops are text replacement, retranslation, tone,
code-mix, speaker gender, voice, speed and the lip-sync window; each one
declares the earliest stage it dirties, and the rerun set is that stage onward
intersected with the stages the job actually configured.

---

**LLM calls are traced to Langfuse, and tracing is off until two keys are
set.** Added 2026-10-19. `backend/tracing.py` wraps the one module that
talks to an LLM (`backend/llm.py`, on behalf of `edit_agent` and
`classify`), so every call becomes a Langfuse *generation* carrying the
prompt, the raw reply, token counts, dollar cost and latency. It closes a
real gap: the parsed ops survive in `<job>/edits/edit_NNN.json` and
`extra.edits.history`, but what the model ACTUALLY said was never stored,
so a hallucinated voice id or a degenerate scope left no evidence.

Five things to know:

- **Both keys or nothing.** Langfuse is HTTP Basic as
  `public_key:secret_key`, for the SDK and the REST API alike, so a lone
  `LANGFUSE_SECRET_KEY` sends and reads nothing. That half-configured state
  WARNS rather than failing silently, because `.env` looks switched on.
  The public key is **not** a secret — Project Settings → API Keys shows
  the `pk-lf-…` any time; only the `sk-lf-…` is one-time.
  `--check-langfuse` verifies the pair and the host and prints no secrets.
- **It must never break a dub**, so every function swallows and logs:
  no keys, an unreachable host, a broken SDK all degrade to "no tracing".
  Unconfigured is silent — a warning per call would be noise on every run
  that never opted in.
- **Short-lived processes must flush.** The SDK batches on a background
  thread, so a CLI invocation can exit with a full queue and lose the trace
  it just made — which reads as "tracing is broken". `flush()` is on
  `atexit` and `make_plan()` calls it outright.
- **`usage: {"include": true}` is now in every OpenRouter payload**, and
  that is the route a PLAIN api key has to token counts and cost: the
  account-level Analytics API needs a *management* key and 403s otherwise.
  Verified live — `usage.cost` came back as `1.24e-06` on an ordinary key.
  Langfuse's own `cost_details` would otherwise be a model-table estimate.
- **The prompts carry the cue table**, i.e. the dub's actual dialogue. On
  Langfuse Cloud that text leaves this host.
  `DUBBING_LANGFUSE_CAPTURE_CONTENT=0` keeps timings, tokens and cost and
  drops prompt/completion text; `LANGFUSE_HOST=http://localhost:3000`
  avoids the question.

One request is ONE trace — `edit_agent.request` wrapping `edit_agent.parse`
and N `edit_agent.rewrite` generations (plus `classify.register` elsewhere),
nested by OTel context rather than by threading ids. A repair round is its
own generation on purpose: a reply that needed a second ask is the
interesting case.

`scripts/langfuse_usage.py` reads it back over the API — per-operation
calls, errors, tokens, cost and p50/p95 latency — because the numbers worth
watching here are per-operation, not per-model.

**Installing it needs `protobuf` pinned.** `pip install langfuse` alone
resolves protobuf 6.32.0 → 7.36.2, a major bump in a package onnxruntime,
NeMo and TensorFlow all link in this env. `pip install langfuse
"protobuf==6.32.0"` installs three packages and touches nothing else
(verified: torch, onnxruntime CUDA EP, insightface and transformers all
still import afterwards).

---

