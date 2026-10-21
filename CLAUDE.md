# CLAUDE.md — Dubbing Studio


Dont run anything ask for the permission before running.
and provide the sh command that I will run.
Indic video dubbing pipeline: take a video (YouTube or local), transcribe it,
translate it, synthesize a dubbed track in an Indian language, mix it, and
optionally lip-sync the speaker's mouth to the new audio.

Repo: `git@github.com:unseen703/Sync_n_Dub.git` · branch `main`.

**Structure lives in [README.md](README.md), not here.** Its generated block —
layout, `backend/` modules, adapter registries, pipeline stages, scripts, tests,
skills, host summary — is rewritten by `scripts/repo_checkup.py` every 5 hours,
so it is current by construction.

This file holds what a snapshot cannot capture: how the host behaves, which
conventions are in use, and the behaviour worth knowing before you hit it.

---

## 1. The host

### Interpreters — there is no `python` on PATH

A bare `python ...` fails with "command not found". The interpreter is named
explicitly:

```
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
```

`dubbing` is the main env — everything in `backend/` and `models/` runs there.
The `musetalk`, `latentsync` and `lipgan` envs exist only because those upstream
repos have incompatible pins; adapters shell into them via `<MODEL>_PYTHON`.
The `lipgan` env (Python 3.7 / TF 1.15) is vestigial — the adapter now uses an
in-env PyTorch port.

### Node is for the UI build only, and bare `npm` is broken

`ui/web2/` (the operator UI) is a Vite app. Its toolchain lives in a separate
conda env so it never reaches the Python server's PATH:

```
conda create -n dubui nodejs=20     # 20.20.2 from pkgs/main
conda run -n dubui npm --prefix ui/web2 install
conda run -n dubui npm --prefix ui/web2 run build     # -> ui/web-dist/
```

The route in is `conda run -n dubui`. The system has node **v12.22.9** on
PATH, and `envs/dubui/bin/npm` is a shim whose shebang is `/usr/bin/env node` —
so calling it directly runs npm 10 under node 12 and dies with
`Error: Cannot find module 'node:path'`, which looks like a corrupt install and
is not. `conda run` sets PATH first, so npm finds its own node. The equivalent
escape hatch is naming the interpreter, same as everywhere else here:
`envs/dubui/bin/node envs/dubui/lib/node_modules/npm/bin/npm-cli.js`.

Node is a **build-time** dependency only — the server serves static files from
`ui/web-dist/` and never shells into node, which is why it is not in
`requirements.txt`.

**The conda `libstdc++` is preloaded, because imports otherwise break by
order.** `cv2`'s bundled `libtbb.so.12` needs `CXXABI_1.3.15`, which the
*system* `/lib/x86_64-linux-gnu/libstdc++.so.6` does not have (the conda one
does). So `import insightface` first fails outright, while `import torch;
import insightface` succeeds — whichever library loads first decides which
`libstdc++` the process gets. The failure looks intermittent and
package-related; it is neither. Scripts touching faces/ASD set:

```
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
```

### GPUs

**2 × NVIDIA RTX PRO 2000 Blackwell, 16311 MiB each** (sm_120).

The 16 GiB ceiling is a real constraint, not a formality: it rules out
LatentSync at 512×512 (~18 GiB) and leaves MuseTalk under 100 MiB of headroom at
`batch_size=20`. Measured per-model costs are in the `clip-run` skill.

Devices are pinned with `CUDA_VISIBLE_DEVICES`, or `DUBBING_GPUS=0,1`. Something
running on CPU here is usually an environment fault rather than a deliberate
choice.

**onnxruntime needs its CUDA libraries preloaded, and fails quietly if not.**
Fixed 2026-10-12; the shape of it recurs on any env rebuild. Two faults were
stacked:

1. Both `onnxruntime` and `onnxruntime-gpu` were installed into the same
   `site-packages/onnxruntime`, and the CPU build won. Only the GPU one is
   installed now — `pip check` will then claim `faster-whisper` / `insightface`
   are missing `onnxruntime`, which is cosmetic: onnxruntime-gpu provides that
   module, pip just cannot see the distribution name.
2. Even after that, `libonnxruntime_providers_cuda.so` could not dlopen
   `libcublasLt.so.12` — the CUDA libs live inside torch's `nvidia-*` pip
   packages, off the loader path. **This failure is silent in the way that
   matters**: `get_available_providers()` still lists `CUDAExecutionProvider`,
   and only `InferenceSession.get_providers()` afterwards shows it fell back
   to CPU. The former is not evidence of GPU use.

`backend.gpu.ensure_onnxruntime_cuda()` calls `ort.preload_dlls()` and verifies
the EP; `retinaface_adapter` and `indicconformer_adapter` both call it before
building sessions. It is code rather than a conda `activate.d` hook, because
the scripts here invoke the env's python directly without activating it.

Measured on `buffalo_l/det_10g.onnx` at 640x640: 55.9 ms/infer on CPU versus
6.0 ms on CUDA. In the pipeline, face detection over 500 frames at 1080p went
33.9 s -> 10.8 s (the phase is part decode and tracking, which stayed on CPU).
The CUDA EP then holds ~460 MiB in the process, which counts against a later
stage's batch on the same GPU.

### ffmpeg is filter-specific

There is no system ffmpeg, and the two available builds differ in filters:

- `envs/dubbing/bin/ffmpeg` (v7.1) — has `atempo`, **not** `rubberband`
- bundled `imageio_ffmpeg` static build (v7.0.2) — has both

`media_io._ffmpeg_bin()` prefers `shutil.which`, so it returns the one *without*
rubberband. `backend/stretch.py::available_backends()` probes per **filter**
rather than per binary. yt-dlp calls take `--ffmpeg-location`, or they report
success while silently leaving unmerged `.fNNN.` part files.

**Nothing is on PATH, and the upstream repos all assume something is.** Scripts
here name the interpreter outright (`$PY script.py`) instead of activating the
env, so a subprocess inherits a PATH with no ffmpeg and no ffprobe on it, while
`envs/dubbing/bin` has both. Every vendored repo calls the bare name somewhere
and none of them check:

- `third_party/Wav2Lip/inference.py:289` muxes `temp/result.avi` onto
  `--outfile` with `subprocess.call(...)` and discards the return code. With no
  ffmpeg it loses the render *after* all the GPU work and **exits 0 with no
  output file** — three minutes of inference, rc=0, nothing on disk.
- `third_party/CodeFormer` did the same, answering it with `sys.exit(0)`.
  That checkout was deleted with the restoration stage on 2026-10-17; the
  pattern is recorded because the next vendored repo will have its own.

Fixed 2026-10-12 in two places, and both are load-bearing:
`media_io._ffmpeg_bin()` / `_ffprobe_bin()` now also look in the *running
interpreter's own directory* (`sys.executable`'s parent), which is the one
reliable pointer back to the env actually executing; and
`BaseLipSync._subprocess_env()` prepends that directory to the subprocess
PATH, so every shelled-out adapter gets it without each runner script
repeating it.

The `_ffprobe_bin()` half mattered more than it looks. It used to return `None`
here, and `None` is a *silent* answer in every caller: probes report
`fps=0.0 dur=0.0`, and `backend/lipsync.py::_needs_remux` returns `False` for
every video — which is how Wav2Lip's 16 kHz mono AAC shipped past the remux
that exists to fix it. An adapter reporting `fps=0.0` on a file that plays is
this bug, not a broken render.

### Secrets

`.env` (gitignored) holds `HF_TOKEN` and the Bodhan credentials
(`BODHAN_{ASR,MT,TTS,TRANSLIT}_{URL,KEY,MODEL}`), plus repo and checkpoint paths
for TalkNet, OpenVoice, Vevo, LipGAN and Wav2Lip. Full key list:
[.env.example](.env.example), which is also where a newly added variable is
documented. The live `.env` is read only by the code that needs it, not
inspected directly, so its values stay out of transcripts and logs.

`OPENROUTER_{API_KEY,URL,MODEL}` was added 2026-10-16 for LLM-side work. It is
now read by `backend/llm.py`, the one module here that talks to an LLM, on
behalf of `backend/edit_agent.py` (§2). The plan's "LLM with termbase" code-mix
path and LLM-judge evaluation still do not use it.

**An empty value still breaks nothing**, and that is deliberate: the edit agent
falls back to a regex parser without a key, and only the three ops that rewrite
text refuse. `OPENROUTER_MODEL` defaults to `google/gemma-3-12b-it` (raised from
the 4B on 2026-10-19) — there is no `gemma-3b` slug on OpenRouter; the family
is `google/gemma-3-{1b,4b,12b,27b}-it`.
A 4B model is a weak structured-output generator, which is why every op it
returns is schema-validated and re-resolved against real segment ids before
anything runs. `--check-llm` makes one round trip and reports whether the key
and slug work, printing neither.

---

## 2. Conventions

**One run, one job folder.** Everything a run produces goes under
`data/jobs/<job_id>/` with a `manifest.json` inventory. Nothing is written
outside it. Layout detail: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

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

**There is no clip-window flag, with one exception.** Running a *stage* on a
time range means cutting video and audio with ffmpeg first — see the `clip-run`
skill for the recipe. The exception, added 2026-10-16:
`run_lipsync(..., window=(start, end))` renders only that slice and splices it
back into the full timeline itself (measured overshoot: one frame, and it does
not grow with the number of joins). It is reached from
`backend/pipeline.py::run_post_mt_stages(lipsync_window=…)` and from the edit
agent below. Nothing else takes a window.

**The dub is re-paced one of two ways, and `global` is the default.**
`--speed-mode global` (unchanged behaviour) places every clip at natural pace
and applies ONE `atempo` to the whole file. `--speed-mode segment` fits each
clip to its own slot in both directions — a short line is stretched out to fill
its slot rather than leaving a pause — bounded by `--speed-cap` (default 1.5,
symmetric) with `--speed-slot-margin` defaulting to 0.0 because any margin is
itself a pause. Whatever the slowdown floor could not absorb is reported as
`unfilled_slot_s` in the timing plan; `0.0` is the checkable form of "no
pauses left".

Both resolve through `backend/cli.py::resolve_speed_setup`, which is also what
`backend/edit_agent.py` calls to replay a run's pacing. The raw argparse
namespace does not describe how a run was paced — `--speed-mode` and
`--timing-method` both override `args.mix_strategy` / `args.no_tts_stretch`
after parsing, so `resolve_speed_setup` is the accurate source.

**There is an operator UI, and it is a second entry point into the pipeline.**
`ui/server.py` (FastAPI, loopback-only, port 7861) serves a built Vite app from
`ui/web-dist/`, falling back to the old read-only inspector in `ui/web/` when
no build exists. Jobs submitted from it are launched DETACHED through
`scripts/ui_run_job.sh`, which is the only launcher that exports the
`libstdc++` preload — starting `ui.runner` with a bare `Popen` walks stage 9h
into the cv2/insightface import-order trap in section 1.

Three things about it are easy to get wrong:

- **`include_router` is called ABOVE `app.mount("/", StaticFiles(...))`.**
  Starlette matches in registration order, so a router added after the static
  mount is shadowed completely: every GET answers a bare 404 and every POST a
  405, served by the static handler. Introspecting `app.routes` does not
  reveal this — included routers appear as one opaque `_IncludedRouter` entry
  whether or not they work. The honest check is to call them, which
  `tests/test_ui_server.py` does.
- **The progress bar shows seven stages, and only seven.** `PHASE_GROUPS` in
  `ui/progress.py` folds the 24 markers into Download video · Speaker
  annotation · Transcribe · Translate · Speech generation · Mix · Lip sync.
  A stage marker added to `pipeline.py` therefore also belongs in a group (or
  in `HIDDEN_KEYS`, for a QA pass that produces no part of the dub), or it
  surfaces as a nameless eighth segment. The tests fail loudly on an omission
  — which is how stages `9b1` and `9b2` were caught the day they were added.
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

**Every hosted stage of a default UI run is Bodhan.** As of 2026-10-17
`ui/presets.py` defaults ASR, MT and TTS all to `bodhan` (ASR moved off
`whisperx` by request). That removes the one stage that used to run locally,
so an outage — which these endpoints answer as `internal_error` 5xx, not as
rate limiting — now takes the whole run rather than most of it. There is no
automatic fallback; `DUBBING_UI_ASR_MODEL=whisperx` is the manual one, and
ASR needs its own `BODHAN_ASR_KEY`, separate from the MT and TTS keys.

**Code-mixing is the MT call's job, and stage 9b is not in a UI run.**
The UI asks for `mt_target_script="codemix"`, which is
`models/mt/bodhan_adapter.py`'s own default and chosen there on measured
grounds. It used to ask for `native` and then run the transliterator (9b) to
put the English terms back — an extra Bodhan call per retained span, editing
the translation it had just asked to be made differently. `translit_model`
is therefore ABSENT from `build_kwargs`, not None: `plan_stages` keys off
presence, so a None would still draw the stage on the progress bar.

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
`ui/web2/src/components/talkinghead/`. Three gates in order: a photo WITH a
face, a language, a line.

**The lines are DATA, not code** — `data/talking_head/dialogues.json`, five
categories (slogan, inspiring, funny, catchy, famous) across 14 languages. Every
entry ships `audio_url: null` by design: the clips are supplied later, and an
entry without one is **offered and marked** rather than hidden, so adding the
link is the only step that makes it playable. No code change, no deploy. The
language dropdown is `ui/langs.options("bodhan")`, i.e. the voice catalog —
not a list written on the page, which is how the tone picker once shipped
values the engine 422s on.

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
face after a GPU job is queued is the expensive way to find out. `POST
.../render` is a deliberate **409** while every `audio_url` is null — MuseTalk
drives the mouth from a waveform, so without a clip the output would be the
photo unchanged, and a job that runs and returns a still is worse than a
refusal that names the entry.

**A dub can be reviewed BEFORE the expensive stage, in three steps.**
Added 2026-10-19. The `draft` preset already dubbed without lip-syncing
(`lipsync: None`); what was missing was a way to inspect the result and then
finish the job. The flow is: review the cast (`GET .../speakers`), review the
cues (`GET .../transcript`), correct either in plain language (`POST
.../agent`, then `POST .../edits/apply`), and finally `POST .../lipsync`.

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

## 3. Project skills

Three in `.claude/skills/`, covering workflows that are otherwise re-derived:

- **`clip-run`** — running a stage or lip-sync model on a time window: pinning
  the spec, VRAM and ETA budgets, artifact paths, what to report back.
- **`long-run`** — launching long GPU jobs detached in tmux `dubsync`, answering
  status from disk, resuming after a session dies, stopping cleanly.
- **`youtube-media-fetch`** — yt-dlp: video, captions, per-language dub tracks,
  dub provenance, the 429 and unmerged-part-file traps.

---

## 4. Known rough edges

- **Three commits now exist.** `8c77641` ("Commit full pipeline work…")
  brought `backend/`, `models/`, `tests/` and `ui/` under version control, so
  the long-standing "only two commits; everything lives on this disk" warning
  no longer applies. `git status` is still busy — `data/` and a number of
  `scripts/` remain untracked — so check it rather than assuming either way.
- `extra_margin` is not plumbed through `models/lipsync/musetalk_adapter.py` —
  that is the knob controlling the bottom of the face crop, and `bbox_shift`
  (which is exposed) only moves the top edge.
- `data/jobs/` was emptied by the 2026-10-11 archive, which deleted the bulk
  media; `logs/` is the surviving record of older runs.
- **The stretch cap is ±0.20 (0.8x-1.2x) and widening it widened the
  DOUBLE-fit too.** Raised from 0.10 on 2026-10-20 by request. It is a
  per-clip cap, not a global speed: a clip already near natural pace is
  untouched, and only one that misses its slot by more than 20% gets pinned
  at the bound. The thing to watch is that the two layers below now each
  have twice the room, so a clip the TTS stage stretches to 1.2x can still
  meet a timing plan that fits it again.

- **Two layers can fit the same clip to the same slot, and both default on
  is a double stretch.** `backend/tts.py` stretches each clip to
  `target_duration_s` (±`--tts-max-stretch`, **default 0.20 since
  2026-10-20** — a ± FRACTION, so 0.20 clamps every clip to 0.8x-1.2x speed
  via `backend/stretch.py:378`'s `max(1-cap, min(1+cap, ratio))`; it was 0.10
  and was widened by request), and a timing plan
  fits it again in the mix. `--speed-mode segment` and `--timing-method`
  therefore both set `tts_apply_stretch=False` and `mix_clip_source=raw`.
  The default `global` path has its own version of this: `global_tempo` +
  `clip_source=auto` reads the already-stretched `tts_audio` and then applies
  a global `atempo` on top. That is long-standing behaviour, documented in
  `backend/mix.py`'s docstring, and `--mix-clip-source raw` is the way out.
- **Per-segment TTS overrides exist now, and only some engines honour them.**
  `backend/edits.py` writes `tts_voice`, `tts_style`, `tts_rate_override`,
  `tts_max_stretch` and `tts_reference_audio` onto individual segments, and
  `backend/tts.py` reads all five. But `style` reaches only the Bodhan adapter
  (`models/tts/bodhan_adapter.py` puts it in `instructions`), and `rate` only
  engines with `supports_rate = True` — since Edge was removed on 2026-10-17
  that is Bodhan alone, and IndicF5 reads neither. A tone edit on
  an IndicF5 run therefore changes the wording and nothing about the delivery.

  **Bodhan's rate field is an inference, and a checkable one.** The hosted
  docs (console.bodhan.ai/api-docs) list only `model`, `input`, `voice` and
  `instructions` — no speed parameter — so the adapter sends `speed`, from
  the OpenAI `/v1/audio/speech` shape this endpoint otherwise matches
  exactly. It is sent ONLY when the rate differs from 1.0, so an ordinary
  run's payload is unchanged and cannot be broken by a wrong guess. If the
  name is wrong the API answers 422 and the adapter says so by name,
  pointing at `BODHAN_TTS_RATE_FIELD` (alternatives: `rate`, or
  `instructions.speed` to nest it). `--tts-speed-control` output counts as
  validated once one rated call has returned 200 on this host.

  **Reachable from the CLI since 2026-10-17** — `--tts-speed-control
  {off,oneshot,closed_loop}` plus `--tts-speed-bounds LO,HI`,
  `--tts-speed-tolerance` and `--tts-speed-max-rounds`. It was
  `experiment.py`-only before, so the adapter's rate support had no pipeline
  caller at all. Default stays `off`, and at rate 1.0 the payload is
  byte-identical to a pre-rate run (verified: keys `input, instructions,
  model, voice`), so the undocumented field name cannot break a run that
  does not ask for a rate.

  Two bounds exist and they are not the same thing: `--tts-speed-bounds`
  (default 0.85–1.15) is what the *planner* is allowed to ask for, while
  `BODHAN_TTS_RATE_RANGE` (default 0.25–4.0) is the adapter's own clamp on
  whatever it is handed. A rate outside the second is clamped with a
  warning; the first never reaches it.

  `--tts-speed-control` does not disable the post-hoc WSOLA stretch, and
  that is deliberate — unlike the double-fit above, these converge on one
  target rather than compounding: the engine gets close, the stretch mops up
  the residual within `--tts-max-stretch`. `--no-tts-stretch` gives a clean
  method-1-vs-method-2 comparison, and the agent says so in its plan rather
  than letting it pass silently.
  `tests/test_edit_agent.py` asserts `backend/tts.py` still reads every key
  `backend/edits.py` writes — the cheap guard against this pair drifting
  back into the dead-code shape that put a female voice on a male lecture.
- **Wav2Lip defaults to `resize_factor=2`, and 1 is not a safe default.**
  Changed 2026-10-18. The vendored S3FD does not downscale a large frame — it
  refuses: `third_party/Wav2Lip/inference.py:81` raises `Image too big to run
  face detection on GPU. Please use the --resize_factor argument`, at stage 9h,
  after ASR, MT, TTS and the mix have been paid for. Measured on a 4K source:
  shard 1 of a 2-GPU split died exactly there while shard 2 of the same run
  succeeded, so it is frame size alone, not the model or the environment.

  **The cost is real and should be stated rather than discovered.**
  `resize_factor` halves the whole OUTPUT, so 1080p renders at 960x540.
  Wav2Lip repaints a 96x96 mouth crop either way, so the mouth loses nothing,
  but the rest of the frame is downscaled — which reads as a quality gap next
  to MuseTalk, which composites at full resolution. `resize_factor=1` is still
  reachable for a small-frame source. VRAM tracks it too: ~1.2 GiB at 2 versus
  ~4.0 GiB at 1 on 720p.

  There is no CLI flag for it; it travels as an adapter kwarg, so the adapter
  default is the only place that reaches the CLI, the UI runner, the edit
  agent and `experiment.py` at once.

- **MuseTalk and LatentSync resolve their own interpreter, and a lip-sync
  failure can no longer read as a success.** Both fixed 2026-10-17 after one
  run demonstrated all three halves at once.

  `<MODEL>_PYTHON` was absent from `.env.example` (it documented "Four
  adapters" and listed two), so the adapters fell back to `sys.executable` —
  the `dubbing` env, where `import diffusers` fails. A blank looks harmless
  because `LIPGAN_PYTHON` / `WAV2LIP_PYTHON` legitimately are: those two DO
  run in `dubbing`. `models/lipsync/base.py::resolve_env_python` now tries the
  explicit argument, then `<MODEL>_PYTHON`, then the **sibling conda env**
  (`<prefix>/envs/<name>/bin/python`, derived from `sys.executable` the same
  way `media_io._ffmpeg_bin()` finds the running env), then `sys.executable`
  with a warning. Resolving it in the adapter covers the CLI, `ui/runner.py`,
  the edit agent and `experiment.py` at once; `.env` is now optional. All four
  variables are in `.env.example`.

  The reporting was the expensive part. `run_lipsync` logged
  `lipsync[musetalk]: done → <path>` at INFO with `(err=… rc=1)` as a trailing
  parenthetical after a screen of upstream TensorFlow banner noise, and the
  pipeline recorded `result.lipsync_mp4 = ls.out_video` regardless — the path
  the adapter was ASKED to write. A 40 s run executed all 15 stages, exited 0,
  and left a manifest naming an mp4 that was never created. Now: **the file is
  the arbiter**, not the error field. `produced` (size > 0) is computed, rc=0
  with no output is promoted to an error — the Wav2Lip shape, where
  `inference.py` muxes with `subprocess.call` and discards the return code —
  a failure logs at ERROR with the verdict first, "done" appears only for a
  real render, and `out.lipsync_mp4` is set only when the file exists.

- **The adapters do not choose the SPEAKING face, and the trap is in both of
  them.** Wav2Lip takes the first S3FD detection (`d = d[0]`,
  `third_party/Wav2Lip/face_detection/api.py:73`) — and so does MuseTalk,
  which vendors the same face_detection lineage at
  `musetalk/utils/face_detection/api.py:80`. This file used to record it for
  Wav2Lip alone, which read as though MuseTalk were safe. It is not: grep the
  whole MuseTalk tree for ASD and the only hit is a GPT-2 vocabulary file.

  MuseTalk is in fact worse, because its crop is landmark-derived. The 68
  points come from DWPose `keypoints[0]` — instance 0 of a **whole-frame**
  pose estimate — and are never matched to the S3FD box they are paired with,
  so on a multi-face frame the box and the landmarks can describe different
  people.

  **`run_lipsync(..., coord_list=…)` is the fix, added 2026-10-19.**
  `backend/lipsync_targets.py::build_coord_list` turns the tracker's face
  tracks plus TalkNet's per-frame scores into one box per frame and reaches
  MuseTalk through its own `--use_saved_coord` pickle, so nothing upstream is
  forked. Measured on the Dark Knight mob clip (132–137 s, 10 face tracks):
  125 frames, 37 with a confirmed speaker across **three different tracks**,
  all 37 landmarked with no fallback, 88 left untouched.

  Four things decided that design, each checked rather than assumed:

  1. **`(0,0,0,0)` now means "write this frame unchanged".** It is MuseTalk's
     `coord_placeholder`, and the 2026-10-18 patch at
     `scripts/inference.py:253` writes such a frame through instead of
     dropping it — the earlier behaviour truncated a 40 s render at 10.64 s.
     This module is only safe against a MuseTalk carrying that fix.
  2. **InsightFace's `landmark_3d_68` is iBUG-68, verified on this host** —
     points 27→30 descend the nose bridge, 8 is the chin, 48/54 are the mouth
     corners. So MuseTalk's `face_land_mark[29]` is InsightFace index 29 and
     the box math transfers 1:1, with no DWPose dependency. `2d106det` is a
     different topology and is deliberately not used.
  3. **Gap-filling is not optional.** A two-frame ASD dip mid-sentence would
     otherwise leave un-animated frames between animated ones, which reads as
     a mouth stutter — worse than the wrong-face problem being fixed.
     `MAX_GAP_FILL` (12 frames) and `MIN_SPEAK_RUN` (5) smooth the runs.
  4. **A supplied list is never written into the coord cache**, and forces
     the single-GPU path. It is gated on one run's ASD rather than being a
     property of the file, and the shards are cut by time with their own
     frame numbering — a whole-video list applied to shard 2 animates the
     wrong frames at exit code 0.

  `bbox_shift` is a no-op on this path: it is consumed inside the detection
  function being skipped, so the shift goes into `build_coord_list`'s own
  argument instead. Wav2Lip and LatentSync ignore `coord_list` rather than
  half-honouring it. `scripts/clip_asd_musetalk_run.py` still does the older
  shot-boundary workaround, and that gating still lives in the script.
- **The manifest is the only durable record of a run, and it used to be
  incomplete.** `_run_pipeline` calls `_jobs.allocate(config=…)` at startup,
  which OVERWRITES whatever the submitter wrote — so the UI's richer config
  survived only until the pipeline began. That dict carried 15 keys and
  none of the optional-stage switches, while `ui/progress.plan_stages`
  branches on 22. A finished job therefore re-planned to 15 stages against
  a progress doc holding 25, and `ui/spec.py`, `ui/cost.py`, `_list_jobs`
  and `edit_agent._config` all described a run with no resegmentation, no
  speaker refinement, no register, no emotion, no visemes and no face
  detector — none of which was true. Fixed 2026-10-17: the allocate config
  now records every switch under the key name `plan_stages` reads, and
  `tests/test_ui_backend_sync.py` greps both sides and fails on the
  difference, so the two cannot drift apart again in silence.

- **A re-run used to duplicate a stage row rather than reuse it.**
  `ProgressWriter._open` matched only a **pending** entry, so a second
  attempt over a stage the first had passed or skipped appended a SECOND
  row with the same key. Every reader takes the first match, so the UI
  reported the previous attempt's verdict: stage 9b2 read `skipped` on a
  run whose own log shows it executing for 11.9 s. `_open` now reuses an
  existing row for the key and appends only when the key is genuinely
  absent. `--rerun` and the edit agent's replay both take this path.

- **Bodhan is the only TTS engine, and the UI no longer offers a choice.**
  Changed 2026-10-17. `ui/presets.py::default_tts()` returns `bodhan`, the
  new-job form states the engine instead of picking it, and `/api/options`
  returns a one-row `tts_engines` list. `POST /api/jobs` validates the id
  against the registry and 400s on an unknown one — it used to allocate,
  queue and launch the job, then die at stage 9e with `Unknown TTS model`.

  Two things follow that are easy to miss:

  1. **`--tts bodhan` now implies `--voice-engine indic-speak`**
     (`pipeline._default_voice_engine`). Those were separate flags, and the
     gap was a silent wrong output: without `--voice-engine`, stage 9d
     skipped casting, every speaker got `voice=None`, and
     `bodhan_adapter._post_one`'s `"voice": voice or "Amit"` read the whole
     dub — every speaker, every line — in one Hindi male voice, with the
     predicted genders computed and discarded. An explicit `--voice-engine`
     overrides it.
  2. **Edge is gone, not deprecated** — `models/tts/edge_adapter.py` is
     deleted and `edge` is out of the registry. It spoke 11 of the 23
     languages offered here and had one voice per gender per language, so it
     could neither cover the catalog nor tell two same-gender speakers
     apart. Its table also carried `pa-IN-OjasNeural`, which does not exist
     — verified absent from all 322 voices `edge_tts.list_voices()` returns
     — and the adapter answered an unknown language by silently substituting
     Hindi, so every Punjabi Edge run produced fluent Hindi at exit code 0.
     `ui/spec.py` and `ui/server.py` still carry `edge` as a *display label*,
     deliberately: jobs recorded before 10-17 name it on disk.

- **The TTS register is chosen at stage 9b1, and a UI run used to miss it
  entirely.** `--tts-style auto` (the UI default) runs
  `backend/classify.py::classify_register` over the uploader's metadata and
  picks one of Bodhan's 8 document-level styles, which rides to the engine
  in `instructions.style`. Four tiers, cheapest first: YouTube `categories`
  → channel map → description patterns → an LLM over the transcript. It is
  resolved at `pipeline.py:1676`, before `run_post_mt_stages`, so it reaches
  TTS; a None answer means the voice's neutral reading, which is the correct
  answer for a local file with no metadata.

  **The trap, fixed 2026-10-17: `source_metadata` is only filled by the
  pipeline's OWN download.** The UI pre-cuts the clip and hands the pipeline
  a local path, so `download_youtube` is never reached there and the dict
  stayed empty — on exactly the runs that have the richest metadata. Tiers
  1–3 all abstained, the LLM tier read the transcript and answered
  "unknown", and the run recorded `register = None` while
  `categories: ["Education"]` sat unused in `<job>/source/<stem>.meta.json`.
  Nothing errored; the dub was merely read neutral. `_recover_source_metadata`
  now picks that file up when `source_metadata` is empty, which took the
  W3L9 lecture from `None (via none)` to `educational lecture (via
  categories)`.

  All 8 `REGISTER_STYLES` are members of the engine's 14
  `INDIC_SPEAK_STYLES` — they are declared in different modules and an
  unknown style is a 422 on every segment, so `tests/test_register_to_tts.py`
  asserts the subset along with the whole five-hop chain from metadata to
  payload. **`--tts-style` defaults to None on the CLI**, and 9b1 fires only
  on the literal `"auto"`, so CLI runs do not classify unless asked.

- **Gender prediction is a THREE-WAY MAJORITY VOTE as of 2026-10-18.**
  `backend/gender.py` runs all three and takes the majority:

  1. **wav2vec2-XLSR-53** (`alefiury/wav2vec2-large-xlsr-53-gender-
     recognition-librispeech`) over the speaker's reference clip.
  2. **F0 median pitch** (librosa.pyin, `F0_THRESHOLD_HZ = 165`), restored
     after a spell deleted. It now ABSTAINS on an unvoiced clip instead of
     answering "male" — with a vote at stake, a guess behind no evidence can
     carry a majority.
  3. **insightface `genderage.onnx`** over the face track ASD attributed to
     this speaker. **CPU only** (`FACE_GENDER_DEVICE`), by request and
     because the CUDA EP holds ~460 MiB for the life of the process (§1)
     upstream of a lip-sync stage with ~100 MiB of headroom.

  The ECAPA logistic head stays deleted (44.4%, 0% female recall, no class
  separation). `DUBBING_GENDER_METHOD` and `DUBBING_ECAPA_THRESHOLD` are
  gone; `DUBBING_GENDER_MODEL` swaps the wav2vec2 checkpoint.

  **The vote helps and does not fix the hard case, which is worth knowing
  before trusting it.** Measured end to end on 25 clips from five videos with
  known truth (`data/gender_bench/eval_5s_*_2026-10-18/`): wav2vec2 alone
  **16/25 (64%)**, the three-way majority **18/25 (72%)**. The 7 survivors
  are all one speaker — the W3L9/W4L10 GAN lecturer — and they survive
  because **the two audio methods fail together on him**: his median F0
  really is 150–207 Hz, above the line, and wav2vec2 moves with it. Two
  correlated voters outvote the one correct face 2-1. The face was right on
  **23 of the 23 clips where it answered**, every one of those seven
  included. A face veto would score 25/25 here; that is a departure from
  majority rule rather than an oversight, and it is not what runs.

  Three more things that are easy to get wrong:

  1. **The wav2vec2 label map is `{0: female, 1: male}`** — the opposite of
     the alphabetical guess — and it is read off `model.config.id2label`,
     never assumed. Hard-coding it inverts the catalog filter for every
     speaker at once with no error anywhere. A checkpoint whose labels are
     not male/female is REFUSED at load.
  2. **Abstention is not disagreement.** A 2-0 with the face silent logs at
     INFO; only voters contradicting each other log at WARNING. Conflating
     them fires on every audio-only run and trains the reader to skip the
     line that matters. `unanimous` additionally requires that all three
     voted, so the artifact cannot claim three-way support it lacks.
  3. **Ties go to the face** (`TIEBREAK_ORDER`), reachable only when one
     voter abstains and the other two split. Justified by the 23/23 above.

  **No votes at all → `FALLBACK_GENDER` ("male") at WARNING, and then a
  TARGET-LANGUAGE male voice.** `pipeline.py` marks that speaker
  `gender_basis="fallback"` and `assign_voices` switches its sort to
  language-first, recording `rationale.basis = "language+gender (no gender
  evidence)"`. Ranking a placeholder gender by ECAPA cosine would dress a
  guess up as a timbre match.

  **Stage 6's `assignments.json` finally has a reader.** It recorded which
  face belongs to which speaker and nothing consumed it, which is why the
  face signal had no caller. `backend/assign_faces.py::speaker_face_tracks`
  reads it back: the winning track is the one with the most ON-SCREEN
  SECONDS (not the most turns — six reaction shots must not outvote one
  close-up), off-screen turns are skipped, and a speaker under
  `min_on_screen_s` is omitted rather than guessed at. Without stage 5 + 6
  the face abstains and the vote is between the two audio methods.

  `GenderVote.to_jsonable()` is written per speaker into the merged JSON
  (`speakers[].gender_vote`) and into `voices/<stem>.speaker_voice_map.json`,
  so a finished run says whether its label came from a 3-0 sweep, a 1-1
  tiebreak, an edit override or a no-evidence fallback without re-running
  three models.

- **The edit agent runs Gemma 3 12B, its catalogue lives in the SYSTEM
  prompt, and OpenRouter JSON mode is OFF because it corrupted output.**
  All three changed 2026-10-19.

  `OPENROUTER_MODEL` now defaults to `google/gemma-3-12b-it` (verified live:
  131k context, $0.05/M in, $0.15/M out). The operation catalogue moved out
  of the per-request user turn into `_SYSTEM`, cutting the user message from
  ~3400 to ~1100 characters.

  **That did NOT make requests cheaper, and it is worth being exact about:**
  measured on Langfuse either side, input tokens per parse went from 1653 to
  **2156, up 30%** — this prompt is 1772 tokens against the old
  `op_catalogue()`'s 635, so the system half grew by more than the user half
  fell. The win is correctness, plus a fixed prefix a cache could reuse.

  The cost is drift risk, since `OPS` is generated and the prompt is
  hand-written, so `tests/test_edit_agent.py` now greps every registered op
  out of `OPS` and fails if the prompt omits it, and fails the other way if
  the prompt advertises an op that does not exist.

  That guard immediately paid: `set_speaker` has been registered in
  `backend/edits.py` all along, but `op_catalogue()` advertised only eight
  of the nine ops, so the model was never told it could reattribute a line.

  **The JSON-mode finding is the one worth remembering.**
  `response_format={"type":"json_object"}` — the flag whose entire purpose
  is to guarantee parseable JSON — is what was breaking it. On this model
  and provider it answered:

      {"ops":[{}]}
      {".ops":[{".op":".lipsync_window",".scope":{"start_s":"0:20.00", ...

  an empty op, and every key prefixed with a full stop. Both are **valid
  JSON**, so `extract_json` parses them without complaint and the caller
  silently receives `ops=None` or a nameless op; the symptom at the CLI was
  `unknown op ''`. The same requests with json mode OFF came back correct
  inside ```json fences, which `extract_json` already strips. So
  `OPENROUTER_JSON_MODE` defaults to `auto`, which means off;
  `OPENROUTER_JSON_MODE=on` restores it for a model that needs it.

  Measured after the change, 8 representative requests: 7 resolve correctly
  (including `set_speaker`, `code_mix` term extraction and a correct refusal
  on an unknown speaker). The eighth, "make it a little faster", returns
  `set_speed` with no `factor` where the prompt asks it to return `{"ops":
  []}` — `backend/edits.py` rejects that with "set_speed needs a positive
  `factor`", so it fails loudly rather than guessing a speed.

- **Voice assignment falls back to target language + gender, and says so.**
  `assign_voices` ranks the same-gender pool by ECAPA cosine, but that needs
  a clip of the speaker, a clip of the catalog voice, and `speechbrain`. Any
  one missing and every candidate ties at 0.0. That used to resolve by
  accident — a stable sort over a catalog that happens to be ordered
  native-language-first. It is now explicit: a voice **recorded in the target
  language, of the speaker's gender**, with `rationale.basis` set to
  `"embedding"` or `"language+gender"` so a finished run says which ran. A
  `score: 0.0` with no `basis` is a pre-10-17 file.

  Gender stays the hard filter: when the target language's voice of that
  gender is taken, a cross-language voice of the right gender wins over the
  right language in the wrong gender. English always lands here — it is a
  valid `lang` for the engine but no voice is *recorded* in it.

  The catalog itself was re-verified against Bodhan's own API docs on
  2026-10-17: all 45 names, languages and genders in `DEFAULT_INDIC_SPEAK`
  match, so the table needs no maintenance, only the clips do.

- **Voice reference clips are RENDERED BY THE ENGINE, never downloaded.**
  `backend/voices.py::assign_voices` ranks candidate voices by ECAPA cosine
  between the real speaker's clip and the catalog voice's own clip, found by
  convention at `data/voice_refs/<engine>/<voice>.wav`. All 45 `indic-speak`
  clips were rendered on 2026-10-17 and the term is now live — verified by
  feeding a catalog voice's own clip back in as a speaker reference and
  getting that same voice at cosine 1.0, with the next-nearest male voice at
  0.75 and the pool spanning 0.12–1.00. Before that the directory was empty,
  `attach_voice_references` attached 0, and the branch scored nothing: every
  candidate tied at 0.0 and the stable sort silently fell back to catalog
  order. Two behaviours worth knowing, in order of cost:
  1. **A clip of some other human filed under `Amit.wav` is worse than no
     clip at all.** The cosine then compares the speaker against the wrong
     timbre and returns a confident similarity, so the log looks like the
     feature is working. `scripts/build_voice_refs.py` re-renders them; the
     clips are not sourced from the internet. (A zero-shot engine whose
     "voices" ARE reference clips — IndicF5 takes `ref_audio_path` +
     `ref_text` — is a different catalog with a different provenance story,
     not more files in this directory. `load_voice_catalog("indicf5")`
     returns 0 voices today.)
  2. `scripts/build_voice_refs.py` had no `load_dotenv()` call, so run
     directly it saw no `BODHAN_TTS_KEY` and every render failed on auth —
     which reads like an upstream outage rather than a missing line. Fixed
     2026-10-17. `backend/cli.py`, `edit_agent.py` and `experiment.py` each
     call it for themselves, and a new entry point under `scripts/` needs
     the same.
  Expect 429s throughout a full 45-voice run; the adapter backs off and the
  script is idempotent, so a re-run fills only the gaps.

- **Bodhan endpoints return `internal_error` 5xx during upstream outages** —
  this is not rate limiting, and since Edge was removed there is no second
  engine to fall back to. `bodhan_adapter.synthesize` catches per request, so
  an outage does not raise: every segment comes back with `tts_error` set and
  the stage returns green with `extra.tts.errors == len(segments)`. That count
  is the signal, not the exit code. A missing `BODHAN_TTS_KEY` is the louder
  case — `_ensure_loaded` raises before the first request.
- **ASD scores produced before 2026-10-11 are worthless.** `talknet_adapter.py`
  fed `soundfile` floats in [-1, 1] into an MFCC front-end that upstream trains
  and infers on int16 PCM, which shifted every log-energy cepstral term and
  pinned all scores near zero. Measured on one 3 s single-speaker close-up:
  mean 0.044 / max 0.129 / 0 of 75 frames above 0.5 before the fix, versus
  mean 0.411 / max 0.798 / 31 of 75 after. The stage never errored — it just
  answered "nobody is speaking" every time, so any archived `assignments.json`
  or on-screen count from an earlier run is stale rather than reusable.

---

## 5. Keeping the docs honest

Six files, six different owners. Knowing which is which prevents the usual
failure, where someone hand-edits a generated table and their work is erased on
the next run.

| File | Owned by | Covers |
|---|---|---|
| **README.md** — generated block | `scripts/repo_checkup.py` | structure: layout, modules, registries, stages, scripts, skills |
| **README.md** — prose outside it | you | what the project is, how a run is organised, why |
| **CHANGES-PENDING.md** | script writes, you clear | changes awaiting a narrative |
| **CLAUDE.md** (this file) | you | host behaviour, conventions, traps |
| **docs/EDIT_AGENT.md** | you | the edit agent's ten ops, scopes, and every refusal |
| **docs/PROMPTS.md** | you | every LLM prompt verbatim, where it fires, what reads the answer |
| **docs/UI_RUN.md** | you | what a UI submission sets per stage, and the stage plan |

### The scheduled checkup

A cron job runs every 5 hours:

```
17 */5 * * *  →  scripts/repo_checkup.py  →  logs/repo_checkup.log
```

It snapshots the structure, diffs it against `.repo-state.json`, and when
something moved it rewrites README's generated block and appends what changed to
`CHANGES-PENDING.md`. When nothing moved it exits silently, so the common case
costs nothing. It also runs by hand:

```bash
$PY scripts/repo_checkup.py --dry-run   # report drift, write nothing
$PY scripts/repo_checkup.py             # update README + pending buffer
$PY scripts/repo_checkup.py --clear     # empty the buffer once folded in
```

### When CHANGES-PENDING.md is non-empty

The generated block is already correct — the structure table does not need
help. What the buffer is asking for is the part a diff cannot produce: *why* it
changed, and what it means for someone using the repo.

The entries get read, anything that deserves explanation is folded into README's
prose or into this file, and then `--clear` empties the buffer. An entry that
needs no narrative — a renamed helper, a deleted scratch file — can be cleared
unexplained. The buffer is a prompt, not a backlog to discharge in full.

### When this file changes

The checkup cannot see any of what §1–§4 describe, so those sections stay true
only by being maintained. The occasions that call for an edit here:

- adding, rebuilding or moving a conda env or interpreter
- changing hardware, drivers, CUDA, or an ffmpeg build
- adding an env var or secret to `.env.example`
- changing a convention — job layout, where artifacts go, how models are
  registered
- adding or materially changing a skill in `.claude/skills/`
- fixing something in §4, or hitting new behaviour worth recording

Edits go in the affected section rather than appended at the end, and a fact
checked against the live system — the registry, `nvidia-smi`, the log — before
being written down. A contradicted line is worse than a missing one, so a line
that stops being true gets deleted.
