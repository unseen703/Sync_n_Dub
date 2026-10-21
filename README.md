# Dubbing Studio

Indic video dubbing pipeline. Takes a video — a YouTube URL or a local file —
transcribes it, translates it, synthesizes a dubbed track in an Indian language,
mixes it against the original music and effects, and optionally lip-syncs the
speaker's mouth to the new audio.

Design intent and the original staging plan: [plan.md](plan.md).
Host setup, environments and working conventions: [CLAUDE.md](CLAUDE.md).

## Quickstart

```bash
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python   # no `python` on PATH

bash setup.sh                                 # doctor, model fetch, 60s dry run

$PY -m backend.cli --input <url-or-file> \
    --model pyannote --asr whisperx --align mms_fa \
    --mt bodhan --mt-target-language hi \
    --tts bodhan --mix --lipsync wav2lip

$PY -m backend.jobs list                      # every run, newest first
$PY -m pytest tests/ -q
```

One input per run. Each run becomes its own job folder under `data/jobs/`.

## How a run is organised

One run, one folder. Nothing a run produces is written outside it:

```
data/jobs/<job_id>/
  manifest.json      status, config, argv, inventory of every file below
  videos/ audio/ stems/ diarization/ faces/ asd/
  asr/ align/ mt/ voices/ tts/ vc/ mix/ lipsync/ eval/ exp/
```

`<job_id>` is `<timestamp>__<input-slug>__<diar>.<asr>.<align>.<mt>.<lang>`, so
runs sort chronologically and re-running the same config never overwrites an
earlier one. `manifest.json` carries a `status` of `running`, then `ok` or
`failed` — a crashed run closes out its own folder rather than leaving it
indistinguishable from one still in flight.

Three things live deliberately *outside* job folders because they are inputs
rather than outputs: `models/duration/<lang>.<voice>.json` (fitting one costs a
full TTS calibration pass), `experiments/suites/*.json` (sweep definitions), and
`data/voice_refs/<engine>/<voice>.wav` — one clip per catalog voice, shared by
every run that assigns voices.

Those clips are what makes voice assignment compare *timbre* rather than only
gender: `assign_voices` ranks each same-gender candidate by ECAPA cosine against
the speaker's own reference clip, and `SpeakerVoice.score` is that cosine. The
directory was empty until 2026-10-17, so the term scored nothing, every
candidate tied at 0.0, and the sort fell back to catalog order — a dead branch
that logged like a working feature. All 45 `indic-speak` clips are now rendered;
regenerate with `$PY scripts/build_voice_refs.py` (idempotent, re-runs fill only
gaps, expect 429s). They are gitignored because that one command rebuilds them.

**They are rendered by the engine rather than downloaded.** The cosine is only
meaningful against what the candidate voice actually sounds like, so a clip of a
different human filed under `Amit.wav` scores a confident match on the wrong
timbre — strictly worse than having none. A zero-shot engine whose voices *are*
reference clips (IndicF5 takes `ref_audio_path` + `ref_text`) needs a separate
catalog with its own provenance, not extra files here.

Full description, including the retired per-stage layout now in
`data/_archive/`: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

### Dubbing one window of one video

`backend/cli.py` has no clip flag — it runs whole files — so a time-window run
means cutting first and running on the cut. `scripts/clip_asd_musetalk_run.py`
packages that for the common case: one window, driven by an already-downloaded
YouTube dub track (so MT and TTS are skipped outright), lip-synced with
MuseTalk, with every visual stage dumping what it detected — face boxes per
track, ASD score per track beside the speaker's own crop, and the exact crop
MuseTalk will use, rendered *before* inference rather than after.

It exists mainly because of one thing the lip-sync stage does not do: nothing
anywhere gates lip-sync on who is speaking. `run_lipsync` hands the whole video
to the adapter, and the adapters animate one face per frame with no identity
tracking, so on a multi-face shot an arbitrary face gets the dubbed mouth. The
script cuts on shot boundaries, asks ASD which shots have a single speaking
face, lip-syncs only those, passes the rest through untouched, and splices the
result back to the original length. Shots are the unit because a face box only
means anything within one continuous shot.

Launch it detached via `scripts/run_clip_asd_musetalk.sh` (tmux `dubsync`); it
writes `STATUS.json` per phase so a later session can answer "where is it" from
disk instead of from the transcript.

### The face comes back soft, and nothing sharpens it

Lip-sync models rebuild the mouth at a small fixed working resolution —
Wav2Lip generates a **96x96** crop and pastes it back, MuseTalk 256x256 — so
the region they touched comes back softer than the face around it. The higher
the source resolution, the more visible it is: on a 4K frame with the face at
24% of frame height (~520 px), a 96x96 crop is upscaled roughly 5x.

There used to be a face-restoration stage for this (`models/restore/`,
CodeFormer, `--restore codeformer`). **It was removed on 2026-10-17** along
with its CLI flags, its comparison script and the UI plumbing. Nothing in the
pipeline re-sharpens the animated region now.

What is left is model choice. `--lipsync musetalk` works at 256x256 rather
than 96x96, which is ~2.7x the linear mouth resolution, at 8-21x clip length
instead of 3-5x. The face probe picks between them automatically from face
size (the threshold is 25% of frame height); `--lipsync musetalk` overrides it
when the shot is unusual — a 4K source is exactly that case.

For reference, the removed pass measured 386 s on 30 s / 747 frames of 720p
(~13x realtime, ~770 MiB VRAM), roughly doubling a Wav2Lip run, and raised
detail in a 139x131 box around the face by +36%. The vendored
`third_party/CodeFormer/` checkout is still on disk; the adapter that drove it
is not.

## Translation script: code-mix is the default

Bodhan's `indic-translate` endpoint takes a **`target_script`** argument, and
this repo now sends **`codemix`** by default — in the adapter
([models/mt/bodhan_adapter.py](models/mt/bodhan_adapter.py),
`DEFAULT_TARGET_SCRIPT`), on the CLI (`--mt-target-script`, default `codemix`)
and in the clip driver ([scripts/clip_dub_audio.py](scripts/clip_dub_audio.py),
`--target-script`). The three values:

| `target_script` | What comes back |
|---|---|
| `codemix` *(default)* | English technical terms stay in Latin inside the target-script sentence — `weight सही है`, `derivative की गणना करें` |
| `native` | Everything re-scripted into Devanagari/Tamil/… |
| `roman` | The whole sentence in Latin script |

**Why `codemix` rather than `native`.** `native` re-scripts *inconsistently*.
On one 30 s Hindi clip of the NAG lecture, Bodhan returned the same term three
different ways inside half a minute:

```
cue 0  →  जहां हमारे पास डेल्टा डब्ल्यूटी था       ("WT" transliterated)
cue 5  →  W के संबंध में लॉस फंक्शन                ("W" left in Latin)
cue 7  →  ब्रैकेट हमारे पास WT था                  ("WT" left in Latin)
```

That is not cosmetic. An Indic TTS voice fed a bare Latin remnant does not
reliably pronounce it, so the inconsistency reaches the listener as a dropped
or mangled term. `codemix` asks for the Latin deliberately and applies it
uniformly, which is also closer to how the source material is actually
spoken — Indian technical lectures are genuinely code-mixed.

Measured on the same clip, `codemix` produced clean Hinglish for every cue with
no term left ambiguous, and Bodhan `indic-speak` synthesised all 8 cues with
full speech occupancy across the 30 s window.

**If you do want fully native script**, pass `--target-script native`. Bodhan's
Latin remnants are then worth cleaning up, which is what the separate
code-mix/script-normalisation stage in [backend/codemix.py](backend/codemix.py)
is for (`--translit`, `--termbase`): it detects retained spans, transliterates
them in one batch, and substitutes them back. `scripts/clip_dub_audio.py`
exposes that as `--normalize-script` — off by default, and the *opposite* of
`--target-script codemix`; the two cancel each other out when combined.

## Pacing the dub: two speed modes

The dubbed line is almost never the length of the slot it has to fill —
measured on this corpus, Hindi TTS runs **1.21–1.65× longer** than the English
it covers, and sometimes it comes back shorter. `--speed-mode` is the choice of
what to do about that.

| `--speed-mode` | What happens | Dub length |
|---|---|---|
| **`global`** *(default)* | Every clip plays at its natural pace, placed shift-forward so nothing overlaps, then **one** `atempo` over the whole file | Exactly the source duration |
| `segment` | **Each clip is fitted to its own slot** — compressed when the line is longer, **stretched out when it is shorter**, so a short line does not leave a pause inside its own slot | Each slot filled; overruns cascade forward |

```bash
# default — nothing to pass
$PY -m backend.cli -i <video> --asr bodhan --mt bodhan --mt-target-language hi     --tts bodhan --mix

# per-segment fit, no pauses from short lines
$PY -m backend.cli -i <video> … --tts bodhan --mix --speed-mode segment
```

`--speed-cap` (default **1.5**) bounds `segment` mode symmetrically: at most
1.5× faster, at most 1/1.5 = 0.67× slower. `--speed-slot-margin` (default
**0.0**) is slot time left deliberately empty at the end of each segment —
any margin *is* a pause, which is the thing this mode removes, so it starts at
zero.

**When the bound is not enough, it is reported, not hidden.** A clip too short
to reach its slot even at the floor leaves silence, and the timing plan records
it per segment as `qc.unfilled_s` and in the summary as `unfilled_slot_s` /
`unfilled_slot_segments`. `unfilled_slot_s: 0.0` is the machine-checkable
version of "no pauses were left".

**`segment` turns the TTS-stage stretch off** (`tts_apply_stretch=False`) and
mixes from `tts_raw_audio`. Both layers fit a clip to the same slot, so leaving
both on re-times every clip twice — the double-stretch this repo has already
been bitten by. Stage 9c (`--duration-adjust`) and
`backend.edits.set_speed` still apply: `segment` mode reads
`target_duration_s` as the slot when it is present.

### Which one to pick

`global` is the default because it measured best on slot overrun: **+0.6%**
against the source duration, versus +2.3% for per-segment fitting at the same
1.5× cap and +16–24% for every arm capped at 1.15×
([comparison-results.md §13.2](comparison-results.md)). Its weakness is that
one global factor cannot fix an individual cue — it makes the *whole* dub
uniformly slightly fast.

Reach for `segment` when individual cues are the problem: lines that end early
and leave the speaker silent on screen, or a few long cues dragging while the
rest fit. It is also the mode to use when lip-sync follows, because a clip that
fills its own slot stays aligned to the mouth that is moving in it.

The research arms (`--timing-method m2|m4|m7`) are the same machinery with
measured parameter sets; `m2` is `segment` with a 0.10 s slot margin.
`--speed-mode` and `--timing-method` set the same knobs and refuse to be
combined.

## The operator UI

Everything above is a CLI. `ui/` is the same pipeline with a screen on it:
paste a YouTube link, pick a language and a part of the video, watch it run,
then listen to the result and recast the voices without touching a terminal.

```bash
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
$PY -m ui.server                       # http://127.0.0.1:7861
```

It binds **loopback only**. Reaching it from a laptop is an SSH tunnel
(`ssh -L 7861:localhost:7861 <host>`); binding a public address is refused
unless `DUBBING_UI_USER` / `DUBBING_UI_PASS` are set, because this UI runs
yt-dlp against whatever URL it is given and starts GPU jobs.

### What it does

| | |
|---|---|
| **New job** | URL, target language, whole video or a time range. The server pre-cuts the clip, so the job *is* the window. |
| **Live progress** | A seven-node step bar — Download video · Speaker annotation · Transcribe · Translate · Speech generation · Mix · Lip sync — with the line between nodes painted as each stage finishes. The pipeline's 24 internal markers fold into those seven; the detail stays in the log. |
| **Video** | Appears as soon as the clip exists, with an Original ⇄ Lip-synced toggle |
| **Audio** | Play and download the original and the dub, under the video |
| **Speakers** | Every speaker with their assigned voice (and tone where the engine supports it), changeable |
| **Jobs** | A rail of job cards titled by video title; failures read "Failed" in red |

### Two things it refuses to do, on purpose

**Lip-sync is capped at 20 minutes of video.** It runs at 3–21× clip length on
this host's two 16 GiB cards, so 20 minutes is already between one and seven
hours. A longer *audio-only* dub is fine — the cap is on the lip-synced
portion. The form shows an estimate for every video before you start.

**A voice change does not re-render until you say so.** Changing a voice
dirties the chain from voice assignment through TTS, mix and lip-sync, and
stage 9e has no per-speaker incremental mode, so one speaker's new voice costs
a full re-synthesis. The dialog quotes what will re-run, marks the GPU line,
and — as importantly — lists what stays: download, diarization, transcription
and translation are not at risk. Because the path never asks for a
retranslate, the quoted cost is the applied cost.

### The lip-sync model is chosen for you

From the video, not from a default. The UI samples ~40 frames, measures the
median largest-face height as a fraction of frame height, and picks **wav2lip**
below 25% (wide shots — its 96×96 mouth crop is proportionate) or **MuseTalk**
above (close-ups need its 256×256 working resolution). Measured on the clips
in this repo: a wide 9-face lecture scores 0.105, two close-ups score 0.419 and
0.401. It tells you what it picked and why, and you can override it.

That threshold is calibrated on three clips from one source video. It is a
defensible default, not a validated classifier; `DUBBING_UI_FACE_THRESHOLD`
moves it.

### Building the frontend

The UI is a Vite app under `ui/web2/`, built to `ui/web-dist/`. Node is a
**build-time** dependency only and lives in its own conda env:

```bash
conda create -n dubui nodejs=20
conda run -n dubui npm --prefix ui/web2 install
conda run -n dubui npm --prefix ui/web2 run build
```

The `conda run` wrapper is what makes this work — the system node is v12 and
bare `npm` fails with `Cannot find module 'node:path'`, which looks like a
corrupt install and is not. See CLAUDE.md §1.

If `ui/web-dist/` is missing or stale the server falls back to the old
read-only inspector in `ui/web/`, which stays reachable at `/legacy`. A broken
frontend build never takes the UI down.

## Editing a finished dub in plain language

A dub is rarely right the first time, and the thing that is wrong is usually
small: one mistranslated line, a male voice on a female speaker, a stretch of
narration that drags. Re-running the pipeline to fix that re-downloads,
re-diarizes, re-detects faces and re-transcribes — hours of work to change one
sentence.

[backend/edit_agent.py](backend/edit_agent.py) takes the change as a sentence
instead:

```bash
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python

$PY -m backend.edit_agent -j 20261016T1200__nag__hi \
    "the speaker at 1:20 should sound female, and speed up 0:30 to 0:45 by 1.2x"
```

It prints a plan — which cues each change touches, what the new text or voice
is, which stages that invalidates and roughly what they cost — and stops.
Nothing is written without `--apply`, and nothing is re-rendered without
`--run`. `--revert` undoes every edit on the job.

**What it can change**

| Op | Example |
|---|---|
| `set_text` | `change the line at 1:20 to "…"` |
| `retranslate` | `retranslate 0:30–0:45, it is too literal` |
| `set_tone` | `make the tone more formal after 1:00` |
| `code_mix` | `code-mix the translation, keep gradient and weight in English` |
| `set_gender` | `the speaker at 1:20 should sound female` |
| `set_voice` | `use Amit for SPEAKER_00` |
| `set_speed` | `speed up the audio between 0:30 and 0:45 by 1.2x` |
| `lipsync_window` | `lip-sync only from 1:00 to 1:30 with musetalk` |

`--list-ops` prints the same table with every parameter.

**How it works.** Stages 9b..9h all read and write one merged ASR+MT JSON in
place, so that file is the edit surface: the agent changes fields in it and
replays only the chain from the earliest stage the change broke. A text edit
starts at TTS. A voice or gender change starts at voice assignment. A lip-sync
window re-renders nothing but the lip-sync, against the mix already on disk.
Stages 1..9a never run again — the merged JSON is already their output.

Each edit is reversible and auditable: every overwritten key is kept in the
segment's `edit_backup`, and each request is journalled into `extra.edits` and
`<job>/edits/edit_NNN.json`.

**The LLM is used for two things only** — reading the request, and rewriting
text for `retranslate` / `set_tone` / `code_mix mode=llm`. Resolving a
timestamp to a cue, picking a voice of a given gender, computing a stretch
factor and deciding which stages are dirty are all ordinary code, which keeps
those parts deterministic. It goes through
[backend/llm.py](backend/llm.py) to OpenRouter, default
`google/gemma-3-12b-it`; `--check-llm` verifies the key and slug in one round
trip. With no key the agent still runs on a regex parser, and only the three
text-rewriting ops refuse.

**Two things worth knowing before you use it.** A speed edit re-fits each clip
into a shorter slot (see *Pacing the dub* above) rather than resampling the
finished mix, so the timeline and the video stay aligned — a sped-up cue ends early and leaves silence,
rather than dragging everything after it out of sync. And `set_tone`'s engine
style hint only reaches the Bodhan adapter, while a synthesis rate only reaches
engines with `supports_rate` — today Bodhan alone, since Edge was removed.
IndicF5 reads neither, so there the tone change comes from the rewritten
wording alone, and the plan says so rather than letting it pass.

## One voice engine, and how a speaker gets cast

**Bodhan indic-speak is the only TTS engine, and the UI does not offer a
choice.** Changed 2026-10-17. The new-job form states the engine instead of
picking it, `GET /api/options` returns a one-row `tts_engines` list, and
`POST /api/jobs` 400s on an engine the registry does not have — it used to
allocate, queue and launch the job and then die at stage 9e with `Unknown
TTS model`, after the operator had already seen a job card. The CLI can
still name `--tts indicf5` for zero-shot cloning; `DUBBING_UI_TTS_MODEL`
moves the UI's default.

**`--tts bodhan` now implies `--voice-engine indic-speak`.** These were
separate flags and the gap between them was a silent wrong output: without
`--voice-engine`, stage 9d skipped casting entirely, every speaker got
`voice=None`, and the adapter's own `"voice": voice or "Amit"` read the
whole dub — every speaker, every line — in one Hindi male voice. The
predicted genders were computed in that same stage and thrown away. Nothing
errored. `pipeline._default_voice_engine` closes it; passing
`--voice-engine` explicitly still wins.

**Casting is timbre first, then target language and gender.**
`assign_voices` filters the catalog to the speaker's predicted gender — a
hard filter, never a weight — and ranks what is left by ECAPA cosine against
the speaker's own clip. That needs three things that can each be absent: a
clip of the speaker, a clip of the catalog voice, and `speechbrain`. Any one
missing and every candidate ties at 0.0.

That case used to resolve by accident, through a stable sort over a catalog
that happens to be ordered native-language-first — a thing that works until
someone reorders a list. It is now explicit and recorded: **a voice recorded
in the target language, of the speaker's gender**, with `rationale.basis`
set to `"embedding"` or `"language+gender"` so a finished run says which
path ran. A `score: 0.0` carrying no `basis` is a pre-10-17 file.

Same-language is a preference, not a filter — indic-speak voices read any of
the 22 languages, so when the target language's voice of that gender is
taken, a cross-language voice of the right gender beats the right language
in the wrong gender. English always lands here: it is a valid `lang` for the
engine but no voice is *recorded* in it.

The catalog in `backend/voices.py::DEFAULT_INDIC_SPEAK` was re-verified
against [Bodhan's own API docs](https://console.bodhan.ai/api-docs/) on
2026-10-17 — all 45 voices, languages and genders match, so the table itself
needs no maintenance. Only the reference clips do.

**The speaking rate goes to the engine, and the field name is a guess that
fails loudly.** `BodhanTTS.supports_rate = True`, so
`--tts-speed-control oneshot|closed_loop` re-synthesises at a new pace
rather than resampling a finished clip — the distinction
[models/tts/base.py](models/tts/base.py) draws between method 1 and method
2. But the published docs list only `model`, `input`, `voice` and
`instructions`, with no speed parameter, so the adapter sends `speed`, taken
from the OpenAI `/v1/audio/speech` shape this endpoint otherwise matches
exactly. Two things keep that honest: the field is sent **only** when the
rate differs from 1.0, so an ordinary run's payload is byte-identical to
what it always was; and a 422 on a rated request raises an error naming
`BODHAN_TTS_RATE_FIELD`, which retargets it without a code change (`rate`,
or `instructions.speed` to nest it). Treat rated output as unvalidated until
one such call has returned 200 on this host.

## Adding a model

Every stage resolves its implementation through a `_REGISTRY_SPEC` dict in that
stage's `models/<stage>/__init__.py`. Adding a model means writing one adapter
against the stage's `base.py` and adding one line to the registry — the CLI
picks up the new choice automatically. The generated table below is read
straight from those registries, so it is the authoritative list.

## Removed on 2026-10-17

A cleanup pass, recorded here because the generated block below shows what the
repo *is* and not what it stopped being. Every item was verified unreferenced
before deletion, and the suite (368 tests) passes unchanged either side of it.

**The Edge TTS adapter.** `models/tts/edge_adapter.py` is deleted and `edge`
is out of the registry; the suite is 434 tests and green. It was removed for
coverage, not tidiness — it spoke 11 of the 23 languages offered here and had
exactly one voice per gender per language, so it could neither cover the
catalog nor tell two same-gender speakers apart, and stage 9d carried a whole
second code path for it that dealt voices out round-robin by sorted speaker
id without consulting the predicted gender at all.

Verifying it before deleting it turned up two faults worth recording, because
both were silent:

- **`pa-IN-OjasNeural` does not exist.** Both voice tables listed it for
  Punjabi — as the female voice *and* the male one. `edge_tts.list_voices()`
  returns 322 voices and not one of them is `pa-*`, nor is any voice anywhere
  named Ojas.
- **An unsupported language was answered with Hindi.** `synthesize` resolved
  its default as `DEFAULT_VOICES.get(primary, DEFAULT_VOICES["hi"])`, so a
  request to dub into any of the 12 languages Edge could not speak produced a
  full set of fluent *Hindi* clips at exit code 0 — a wrong output no
  downstream check could detect. Combined with the line above, every Punjabi
  Edge run ever done here was Hindi.

`ui/spec.py` and `ui/server.py` keep `edge` as a **display label** on purpose:
jobs recorded before 10-17 still name it on disk, and those have to render.
`scripts/calibrate_tts_duration.py` was ported to the registry at the same
time — it fits the per-(language, voice) model stage 9c plans durations with,
so calibrating it against a different engine than the one that speaks would
produce a confident number for the wrong speaking rate.

One fix found while verifying the above: **`attach_voice_references` cried
wolf.** It logged *"no voice reference clips found; the ECAPA similarity term
is INACTIVE"* whenever it attached zero clips — including when all 45 were
already attached, since the built-in catalog is a module-level list of shared
`Voice` objects and only the first `load_voice_catalog` call does any work.
Anyone acting on that line would have rebuilt clips that already existed. It
now reports how many *have* a clip, not how many this call attached.

**`backend/jobs.py` lost its v1 half** — 812 lines to 387. Layout v1 was a
manifest that only *referenced* artifacts in shared `data/<stage>/` folders;
those folders were archived on 2026-10-11, and v1's filename parsing
(`parse_mt_filename`), id synthesis (`job_id_for`), artifact collection,
`build_manifest`/`write_manifest`/`write_manifest_for_mt` and `backfill` had
had no caller outside the module's own `main()` ever since. The module
docstring was the real cost: it still described the referencing layout and
claimed `write_manifest` was "called at the end of `run_pipeline`", which was
the first thing a reader of the job layout hit and was false on both counts.
`python -m backend.jobs` keeps `list` and `show`; `backfill` is gone with the
layout it rebuilt.

**The `pause_aware` stand-in in `backend/timing_advanced.py`** —
`_reference_pause_aware` was written as a placeholder "until the real one
lands", guarded by `if not hasattr(timing, "plan_pause_aware")`. It landed, so
the guard stopped firing and the 67-line stand-in, plus the dead branch in
`_resolve_fallback` that preferred it, were unreachable. The module docstring
explained the whole file as a way to avoid merge conflicts "while other agents
are editing" `timing.py` — a development-process artifact, not architecture —
and now describes what the split actually buys: `plan("pause_aware", …)` does
not pay to import the DP.

**Dead definitions**, each with zero references anywhere in the repo:
`align.asr_result_from_json`, `codemix.CodemixSegment`, `llm.get_client`,
`facepick.probe_face_size`, `timing_advanced._percentile` and
`._split_groups_at`, and three unused dataclass helpers in
`models/faces/base.py`.

**Six shell scripts.** `retry_wav2lip.sh`, `retry_latentsync.sh` and
`run_musetalk.sh` were one script three times over — same GPU-wait, same log
setup, differing only in model, card and timeout — and all three, plus
`persistent_lipsync_runner.sh`, pinned the stem `2vxQaT0Fy-c_1min`, whose media
the archive deleted. `scripts/run_lipsync.sh <model> <stem>` replaces them with
the differences as arguments. `persistent_lipsync_runner.sh` also carried a
completed one-time env repair (waiting on an mmcv wheel, `sed`-patching version
ceilings in site-packages). `restore_nag_only.sh` was a finished one-off whose
output already exists and which `run_lipsync_only.py --restore codeformer` does
generically; `fetch_nag.sh` was two lines wrapping `fetch_corpus.py`.

Two fixes found while verifying the above, neither of them deletions:

- **`requirements.txt` pinned `onnxruntime`, the CPU build.** Installing it
  alongside `onnxruntime-gpu` means the CPU build wins, and the failure is
  silent — `get_available_providers()` still lists `CUDAExecutionProvider`.
  A fresh install would have reintroduced the fault fixed on 2026-10-12, at a
  measured 55.9 ms/infer versus 6.0 ms. Now `onnxruntime-gpu`.
- **`models/asd/ts_talknet_adapter.py` did not import.** The 2026-10-11 TalkNet
  fix split `_crop_track` into `_crop_all_tracks`/`_crop_one` and never updated
  its sibling, so `get_asd("ts_talknet")` raised the registry's "clone the
  upstream repo" error and misattributed the cause entirely. The same file had
  also missed that commit's *other* half — the int16 PCM rescale — so its
  scores would have been pinned near zero for exactly the reason recorded in
  CLAUDE.md section 5. Both are fixed, and the crop now sweeps the video once
  for all tracks instead of once per track. Note the adapter is still untested
  end-to-end here: `third_party/TS-TalkNet` is not cloned on this host.

## Structure

The block below is regenerated by `scripts/repo_checkup.py`, which runs on a
schedule and rewrites it whenever the repo's shape changes. Treat it as current.
Prose outside the block is written by hand and is not touched by the script.

<!-- BEGIN AUTOGEN:structure -->

*Regenerated by `scripts/repo_checkup.py` — 2026-10-20T00:09:22. Edits inside this block are overwritten; write prose outside it.*

### Layout

```
CLAUDE.md
LatentSync/
README.md
audio_browser.ipynb
backend/
comparison-results.md
data/
docs/
experiments/
gfpgan/
lipsync-history.md
logs/
models/
next_implementation_steps.py
notebooks/
plan.md
reference.py
requirements.txt
resourses_for_suggested_edits./
run_w1l2_cli.sh
scripts/
setup.sh
tests/
third_party/
ui/
```

### `backend/` modules

`align`  `asd`  `asr`  `assign_faces`  `caption_asr`  `classify`  `cli`  `codemix`  `diarize`  `downloader`  `duration`  `edit_agent`  `edits`  `emotion`  `experiment`  `faces`  `gender`  `gpu`  `jobs`  `lipsync`  `lipsync_targets`  `llm`  `media_io`  `mix`  `pipeline`  `sanitize`  `segment`  `speaker_id`  `stem_separation`  `stretch`  `timing`  `timing_advanced`  `tracing`  `transcripts`  `translate`  `tts`  `voice_convert`  `voices`

### Adapter registries

Every stage resolves its implementation through a `_REGISTRY_SPEC`, so adding a model means writing one adapter and registering it.

| Stage | Registered |
|---|---|
| `align` | `mms_fa`, `whisperx` |
| `asd` | `talknet`, `ts_talknet` |
| `asr` | `bodhan`, `indicconformer`, `indictranscribe_core`, `indictranscribe_flex`, `whisperx` |
| `diarization` | `diarizen`, `nemotron`, `pyannote` |
| `emotion` | `emotion2vec` |
| `faces.detect` | `mediapipe`, `retinaface` |
| `faces.track` | `bytetrack` |
| `lipsync` | `latentsync`, `lipgan`, `musetalk`, `wav2lip`, `wav2lip_hd` |
| `mt` | `bodhan`, `nllb` |
| `translit` | `bodhan`, `indicxlit` |
| `tts` | `bodhan`, `indicf5` |
| `vc` | `openvoice`, `vevo` |

### Pipeline stages (26)

In execution order, as logged by `backend/pipeline.py`:

```
1     download / resolve
2     extract audio
3     stem separation (HTDemucs)
4     diarization
4b    logical resegmentation
4c    speaker label refinement
5     face detect+track
6     ASD + Hungarian assignment
7     transcript from captions
7     ASR
8     forced alignment
9     machine translation
9b    code-mix / script-norm
9b1   content register = %r
9b2   source emotion
9c    duration planning
9d    speaker voice assignment
9e    TTS synthesis
9f    voice conversion
9f1   post-TTS forced alignment
9f1b  viseme extraction
9f2   timing plan
9g    audio mix (ffmpeg)
9h    lip sync
10a   ASR vs transcript
10b   MT vs dubbed transcript
```

### `scripts/`

`archive_data.sh`  `benchmark_gender.py`  `build_clip_cues.py`  `build_voice_refs.py`  `calibrate_tts_duration.py`  `clip_asd_musetalk_run.py`  `clip_dub_audio.py`  `extract_corpus_audio.sh`  `fetch_captions_runner.sh`  `fetch_corpus.py`  `fetch_indic_audio.py`  `fetch_indic_captions.py`  `gpu_watcher.py`  `hi_cue_floor.py`  `langfuse_usage.py`  `regroup_lipsync_into_jobs.py`  `repo_checkup.py`  `rerun_iit_musetalk_revoiced.sh`  `revoice_lipsynced.py`  `run_clip_asd_musetalk.sh`  `run_gt_hindi_dub.py`  `run_iit_musetalk_indic15.sh`  `run_lipsync.sh`  `run_lipsync_only.py`  `run_nag_wav2lip_indic15.sh`  `run_sweep.sh`  `run_wav2liphd_clips.sh`  `select_clips.py`  `setup.py`  `setup_wav2liphd.py`  `ui_run_job.sh`  `wav2liphd_enhance.py`

### `tests/`

`conftest`  `test_caption_asr`  `test_coord_cache`  `test_edit_agent`  `test_experiment_isolation`  `test_face_import_order`  `test_gender`  `test_gpu_pick`  `test_lipsync_env_python`  `test_lipsync_failure_honesty`  `test_lipsync_targets`  `test_musetalk_frame_parity`  `test_register_to_tts`  `test_speaker_face_tracks`  `test_speaker_id`  `test_speed_mode`  `test_staged_review`  `test_stretch`  `test_talking_head`  `test_timing_plan`  `test_tracing`  `test_ui_backend_sync`  `test_ui_create`  `test_ui_edits`  `test_ui_errors`  `test_ui_media`  `test_ui_progress`  `test_ui_queue`  `test_ui_register_row`  `test_ui_server`  `test_ui_speakers`

### `third_party/` clones

`LatentSync`  `LipGAN`  `MuseTalk`  `TalkNet-ASD`  `Wav2Lip`  `Wav2Lip-HD`  `langfuse`

### Project skills (`.claude/skills/`)

`clip-run`  `long-run`  `youtube-media-fetch`

### Host

GPUs: NVIDIA RTX PRO 2000 Blackwell, 16311 MiB; NVIDIA RTX PRO 2000 Blackwell, 16311 MiB

Environments: `chemformer`, `chemformer-gpu`, `dubbing`, `dubui`, `latentsync`, `lipgan`, `musetalk`, `vllm`, `wav2liphd`, `.venv-diarization`

CLI surface: 79 flags on `backend/cli.py` (`python -m backend.cli --help`).

<!-- END AUTOGEN:structure -->

## Keeping this file honest

`scripts/repo_checkup.py` snapshots the structure, diffs it against
`.repo-state.json`, rewrites the generated block above, and records what moved
in [CHANGES-PENDING.md](CHANGES-PENDING.md). That buffer holds the *narrative*
follow-up — what changed and why it matters — until someone folds it into the
prose here and clears it.

```bash
$PY scripts/repo_checkup.py --dry-run   # report drift, write nothing
$PY scripts/repo_checkup.py             # update README + pending buffer
```
