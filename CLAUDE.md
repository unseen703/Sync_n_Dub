# CLAUDE.md — Dubbing Studio

Indic video dubbing pipeline: take a video (YouTube or local), transcribe it,
translate it, synthesize a dubbed track in an Indian language, mix it, and
optionally lip-sync the speaker's mouth to the new audio.

Repo: `git@github.com:unseen703/Sync_n_Dub.git` · branch `main`.

**Structure lives in [README.md](README.md), not here.** Its generated block —
layout, `backend/` modules, adapter registries, pipeline stages, scripts, tests,
skills, host summary — is rewritten by `scripts/repo_checkup.py` every 5 hours,
so it is current by construction. Read it there rather than trusting any list
written by hand, including one in this file.

This file holds what a snapshot cannot capture: how the host behaves, which
conventions matter, and the traps worth knowing before you hit them.

---

## 1. The host

### Interpreters — there is no `python` on PATH

A bare `python ...` fails with "command not found". Always name the interpreter:

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

**Always go through `conda run -n dubui`.** The system has node **v12.22.9** on
PATH, and `envs/dubui/bin/npm` is a shim whose shebang is `/usr/bin/env node` —
so calling it directly runs npm 10 under node 12 and dies with
`Error: Cannot find module 'node:path'`, which looks like a corrupt install and
is not. `conda run` sets PATH first, so npm finds its own node. The equivalent
escape hatch is naming the interpreter, same as everywhere else here:
`envs/dubui/bin/node envs/dubui/lib/node_modules/npm/bin/npm-cli.js`.

Node is a **build-time** dependency only — the server serves static files from
`ui/web-dist/` and never shells into node, which is why it is not in
`requirements.txt`.

**Preload the conda `libstdc++` or imports break by order.** `cv2`'s bundled
`libtbb.so.12` needs `CXXABI_1.3.15`, which the *system* `/lib/x86_64-linux-gnu/
libstdc++.so.6` does not have (the conda one does). So `import insightface`
first fails outright, while `import torch; import insightface` succeeds —
whichever library loads first decides which `libstdc++` the process gets. The
failure looks intermittent and package-related; it is neither. Any script
touching faces/ASD should set:

```
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
```

### GPUs

**2 × NVIDIA RTX PRO 2000 Blackwell, 16311 MiB each** (sm_120).

The 16 GiB ceiling is a real constraint, not a formality: it rules out
LatentSync at 512×512 (~18 GiB) and leaves MuseTalk under 100 MiB of headroom at
`batch_size=20`. Measured per-model costs are in the `clip-run` skill.

Pin devices with `CUDA_VISIBLE_DEVICES`, or `DUBBING_GPUS=0,1`. If something is
running on CPU, that is almost always an environment fault rather than a
deliberate choice — check before assuming it was intended.

**onnxruntime needs its CUDA libraries preloaded, and fails quietly if not.**
Fixed 2026-10-12; the shape of it is worth keeping because it recurs on any env
rebuild. Two faults were stacked:

1. Both `onnxruntime` and `onnxruntime-gpu` were installed into the same
   `site-packages/onnxruntime`, and the CPU build won. Only install the GPU
   one — `pip check` will then claim `faster-whisper` / `insightface` are
   missing `onnxruntime`, which is cosmetic: onnxruntime-gpu provides that
   module, pip just cannot see the distribution name.
2. Even after that, `libonnxruntime_providers_cuda.so` could not dlopen
   `libcublasLt.so.12` — the CUDA libs live inside torch's `nvidia-*` pip
   packages, off the loader path. **This failure is silent in the way that
   matters**: `get_available_providers()` still lists `CUDAExecutionProvider`,
   and only `InferenceSession.get_providers()` afterwards shows it fell back
   to CPU. Never trust the former as proof of GPU use.

`backend.gpu.ensure_onnxruntime_cuda()` calls `ort.preload_dlls()` and verifies
the EP; `retinaface_adapter` and `indicconformer_adapter` both call it before
building sessions. It has to be code rather than a conda `activate.d` hook,
because the scripts here invoke the env's python directly without activating it.

Measured on `buffalo_l/det_10g.onnx` at 640x640: 55.9 ms/infer on CPU versus
6.0 ms on CUDA. In the pipeline, face detection over 500 frames at 1080p went
33.9 s -> 10.8 s (the phase is part decode and tracking, which stayed on CPU).
The CUDA EP then holds ~460 MiB in the process, so budget for it when sizing
a later stage's batch on the same GPU.

### ffmpeg is filter-specific

There is no system ffmpeg, and the two available builds differ in filters:

- `envs/dubbing/bin/ffmpeg` (v7.1) — has `atempo`, **not** `rubberband`
- bundled `imageio_ffmpeg` static build (v7.0.2) — has both

`media_io._ffmpeg_bin()` prefers `shutil.which`, so it returns the one *without*
rubberband. Probe per **filter** via `backend/stretch.py::available_backends()`,
never per binary. yt-dlp calls need `--ffmpeg-location`, or they report success
while silently leaving unmerged `.fNNN.` part files.

**Nothing is on PATH, and the upstream repos all assume something is.** Scripts
here name the interpreter outright (`$PY script.py`) instead of activating the
env, so a subprocess inherits a PATH with no ffmpeg and no ffprobe on it, while
`envs/dubbing/bin` has both. Every vendored repo calls the bare name somewhere
and none of them check:

- `third_party/Wav2Lip/inference.py:289` muxes `temp/result.avi` onto
  `--outfile` with `subprocess.call(...)` and discards the return code. With no
  ffmpeg it loses the render *after* all the GPU work and **exits 0 with no
  output file** — three minutes of inference, rc=0, nothing on disk.
- `third_party/CodeFormer/basicsr/utils/video_util.py` answers the same
  `FileNotFoundError` with `sys.exit(0)`.

Fixed 2026-10-12 in two places, and both are load-bearing:
`media_io._ffmpeg_bin()` / `_ffprobe_bin()` now also look in the *running
interpreter's own directory* (`sys.executable`'s parent), which is the one
reliable pointer back to the env actually executing; and
`BaseLipSync._subprocess_env()` / `BaseRestore._subprocess_env()` prepend that
directory to the subprocess PATH, so every shelled-out adapter gets it without
each runner script remembering to.

The `_ffprobe_bin()` half mattered more than it looks. It used to return `None`
here, and `None` is a *silent* answer in every caller: probes report
`fps=0.0 dur=0.0`, and `backend/lipsync.py::_needs_remux` returns `False` for
every video — which is precisely how Wav2Lip's 16 kHz mono AAC shipped past the
remux that exists to fix it. An adapter reporting `fps=0.0` on a file that
plays is this bug, not a broken render.

### Secrets

`.env` (gitignored) holds `HF_TOKEN` and the Bodhan credentials
(`BODHAN_{ASR,MT,TTS,TRANSLIT}_{URL,KEY,MODEL}`), plus repo and checkpoint paths
for TalkNet, OpenVoice, Vevo, LipGAN and Wav2Lip. Full key list:
[.env.example](.env.example).

`OPENROUTER_{API_KEY,URL,MODEL}` was added 2026-10-16 for LLM-side work. It is
now read by `backend/llm.py`, the one module here that talks to an LLM, on
behalf of `backend/edit_agent.py` (§2). The plan's "LLM with termbase" code-mix
path and LLM-judge evaluation still do not use it.

**An empty value still breaks nothing**, and that is deliberate: the edit agent
falls back to a regex parser without a key, and only the three ops that rewrite
text refuse. `OPENROUTER_MODEL` defaults to `google/gemma-3-4b-it` — there is no
`gemma-3b` slug on OpenRouter; the family is `google/gemma-3-{1b,4b,12b,27b}-it`.
A 4B model is a weak structured-output generator, which is why every op it
returns is schema-validated and re-resolved against real segment ids before
anything runs. `--check-llm` makes one round trip and reports whether the key
and slug work, printing neither.
Never explicitly load or read .env file via any direct or indirect method to preseve screts.
Get guideline from the .env.example and modify it for any addition to new env varioble if required.

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
against them is fine; pasting one into an LLM prompt is a live injection, so
pass them as delimited data and constrain the answer to a known enum.

**Adding a model is one adapter plus one registry line** in
`models/<stage>/__init__.py`. The CLI picks up the new choice automatically.
`models/restore/` (face restoration, post-lip-sync) is the newest stage and the
cleanest worked example of the pattern — GFPGAN or RestoreFormer would drop in
beside `codeformer` with one line each.

**There is no clip-window flag, with one exception.** Running a *stage* on a
time range means cutting video and audio with ffmpeg first — see the `clip-run`
skill for the recipe. This is the single biggest source of wasted turns in this
repo. The exception, added 2026-10-16: `run_lipsync(..., window=(start, end))`
renders only that slice and splices it back into the full timeline itself
(measured overshoot: one frame, and it does not grow with the number of joins).
It is reached from `backend/pipeline.py::run_post_mt_stages(lipsync_window=…)`
and from the edit agent below. Nothing else takes a window.

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
`backend/edit_agent.py` calls to replay a run's pacing. Do not read
`args.mix_strategy` / `args.no_tts_stretch` directly to decide how a run was
paced — `--speed-mode` and `--timing-method` both override them after argparse,
so the raw namespace lies.

**There is an operator UI, and it is a second entry point into the pipeline.**
`ui/server.py` (FastAPI, loopback-only, port 7861) serves a built Vite app from
`ui/web-dist/`, falling back to the old read-only inspector in `ui/web/` when
no build exists. Jobs submitted from it are launched DETACHED through
`scripts/ui_run_job.sh`, which is the only launcher that exports the
`libstdc++` preload — never start `ui.runner` with a bare `Popen`, or stage 9h
walks into the cv2/insightface import-order trap in section 1.

Three things about it are easy to get wrong:

- **`include_router` must be called ABOVE `app.mount("/", StaticFiles(...))`.**
  Starlette matches in registration order, so a router added after the static
  mount is shadowed completely: every GET answers a bare 404 and every POST a
  405, served by the static handler. Introspecting `app.routes` does NOT
  reveal this — included routers appear as one opaque `_IncludedRouter` entry
  whether or not they work. The only honest check is to call them, which
  `tests/test_ui_server.py` does.
- **The progress bar shows seven stages, and only seven.** `PHASE_GROUPS` in
  `ui/progress.py` folds the 24 markers into Download video · Speaker
  annotation · Transcribe · Translate · Speech generation · Mix · Lip sync.
  Adding a stage marker to `pipeline.py` therefore means adding it to a group
  (or to `HIDDEN_KEYS`, for a QA pass that produces no part of the dub), or it
  surfaces as a nameless eighth segment. The tests fail loudly if you forget —
  which is how stages `9b1` and `9b2` were caught the day they were added.
- **Progress is scraped from the pipeline's own `=== stage N ===` log lines**
  by a `logging.Handler` that `ui/runner.py` installs on the ROOT logger.
  `backend/` has no idea the UI exists, which is the point — but renaming a
  marker silently freezes the progress bar. `tests/test_ui_progress.py` greps
  `pipeline.py` for every marker literal and fails if one stops matching.
  Note the handler needs the root logger's LEVEL raised too: a handler never
  sees a record its logger filtered first, and root defaults to WARNING.
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
  nothing on screen to explain it. Send `value` verbatim, show `label`.
- **Gender is persisted now, so the UI never recomputes it.** Stage 9d
  predicts it with ECAPA and writes it into
  `voices/<stem>.speaker_voice_map.json` beside the voice. Re-deriving it in
  a request would mean loading a speaker-embedding model inside a GET.
  Voices carry `recommended_for` (speaker ids of matching gender) — a
  recommendation, never a filter, because the prediction is a prediction.
- **Renaming a speaker writes to `<job>/speaker_labels.json`, not the merged
  JSON.** It is a display preference; putting it in the merged JSON would
  dirty a pipeline stage and cost a re-render for changing a caption.
- **Three job-folder shapes exist on disk** (`progress.json`, the clip
  runner's `STATUS.json`, a bare `manifest.json`) plus 16 folders with no
  status file at all. `ui/progress.py::read_progress` normalises all four;
  nothing above it may branch on which runner produced a job.

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

**Read its thresholds against the control, never against intuition.** Short
windows do not give the ~0.9 ECAPA cosine a full-clip comparison does. Measured
on one known speaker, clean lecture audio, 1.5 s windows: **0.45–0.55, min
0.35** — that *is* the single-speaker ceiling at this length, and `PURITY_FLOOR
= 0.35` sits under it. Different speakers scored 0.007–0.141 between centroids.
On a 279 s Dark Knight mob scene (6+ characters in 4 labels) the pass took 4
labels to 11, and labels below the floor went 3-of-4 to 1-of-11.

Two traps paid for while building it:

1. **Absorbing a fragment into its "nearest sibling" always finds someone.**
   A sub-cluster too short to cast gets folded back, but ungated that undid
   the split it had just made — SPK_00 returned to 0.331 and SPK_02 to 0.274,
   below the floor they had just been cleaned past. `ABSORB_MIN_COSINE` now
   gates it: a bit-part resembling nobody present is a real speaker with thin
   evidence and keeps its own label (recorded in `extra.speaker_id.stranded`).
2. **A synthetic test fixture must be calibrated to the measured numbers.**
   At `dim=32` two random "speakers" sit at cosine 0.31 — nothing like the
   0.007–0.141 real ones scored — and the tests silently stop testing the
   split. `tests/test_speaker_id.py` uses ECAPA's real width (192) with
   orthogonalised identities and a jitter tuned to land a synthetic speaker at
   ~0.50. Calibrate the fixture to the host, never the threshold to the
   fixture.

`extra.speaker_id` records purity per input label, the splits, the absorbed and
stranded fragments and the merges, so a finished run can be audited without
re-running the embeddings. `<job>/diarization/<stem>.<model>.refined.rttm` is
the relabelled turn list.

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

## 3. Guardrails

These exist because each one was paid for. They are not style preferences — every
line below traces to a run that had to be interrupted, redone, or thrown away
because the agent reported on what it *meant* to do rather than on what it
actually produced.

### Verify a mechanism before building work on it

If the value of a run depends on "knob X does Y", read the code path and quote
the line before launching. Do not infer a knob's behaviour from its name.

A three-run MuseTalk sweep was once designed around the claim that `bbox_shift`
moves the face box's *lower* edge downward. It moves the top edge. The claim was
checkable in half a minute of reading `musetalk_adapter.py`; instead it cost four
hours, three runs and three rounds of "there is no difference at all" before
anyone read the code. When the code was finally read, the answer was immediate.

### Announce a scope reduction before doing the work

Any substitution is a decision point, not a footnote: a shorter clip than asked
for, a different source stem, CPU instead of GPU, a smaller model, a skipped
stage. Say it before starting and let it be accepted or rejected.

Asked for five minutes across two videos, a past run quietly delivered a
ten-second test clip, and the substitution surfaced only when the file durations
were questioned. Separately, a LipGAN run was launched CPU-only — a defensible
call, since no TF 1.x GPU wheel exists for Blackwell — but it was first disclosed
**1h38m in**, and only in reply to "current status". The decision was sound; the
silence was not.

### Verify semantically, not by exit code

A stage returning 0 is not a stage working. For anything with perceptible output,
check that the artifact actually differs in the direction claimed before
reporting success.

The voice-assignment stage ran green for several dubs while its gender matching
was dead code — `predicted_gender` was never populated, so the condition never
fired and a female voice was assigned to all-male lectures. Nothing errored. It
was caught by a human listening to the output.

### When an observation contradicts a probe, the probe is incomplete

If someone says the video has no sound and ffprobe says the audio is present,
both are usually true at different layers — and the missing layer is the answer.
Treat the conflict as a gap in the measurement rather than a dispute to win.

That exact case took three rounds, including "the issue is on the viewer side,
not the file", before the real cause landed: Wav2Lip emits 16 kHz mono AAC, which
several players silently refuse. The probe was right about the bytes and wrong
about the question.

### Reuse before building

Check for an existing implementation before writing one. Hand-rolled BLEU, ROUGE
and METEOR were once drafted as "pure Python, no new deps" when established
libraries were available and expected.

---

## 4. Project skills

Three in `.claude/skills/`. Prefer them over re-deriving these workflows:

- **`clip-run`** — running a stage or lip-sync model on a time window: pinning
  the spec, VRAM and ETA budgets, artifact paths, what to report back.
- **`long-run`** — launching long GPU jobs detached in tmux `dubsync`, answering
  status from disk, resuming after a session dies, stopping cleanly.
- **`youtube-media-fetch`** — yt-dlp: video, captions, per-language dub tracks,
  dub provenance, the 429 and unmerged-part-file traps.

---

## 5. Known rough edges

- **The working tree is largely uncommitted** — only two commits exist; `tests/`
  and `ui/` are entirely untracked and `scripts/` is 1-of-16. Everything built
  since 10-01 exists only on this disk.
- **CodeFormer's own video mode reports success and writes nothing here.**
  `--input_path foo.mp4` routes through `basicsr/utils/video_util.py`, which
  shells out to the bare name `ffmpeg` — which is not on this host's PATH — and
  answers the resulting `FileNotFoundError` with `sys.exit(0)`. Exit code 0, no
  output file, no error text. `models/restore/codeformer_adapter.py` therefore
  never uses it: it extracts frames with `media_io._ffmpeg_bin()`, feeds
  CodeFormer the *folder*, and reassembles itself. Keep it that way.
- `extra_margin` is not plumbed through `models/lipsync/musetalk_adapter.py` —
  that is the knob controlling the bottom of the face crop, and `bbox_shift`
  (which is exposed) only moves the top edge.
- `data/jobs/` was emptied by the 2026-10-11 archive, which deleted the bulk
  media; `logs/` is the surviving record of older runs.
- **Face restoration restores every face it finds**, same as lip-sync animates
  whichever face it finds. `--restore-only-center-face` narrows it to one, but
  there is still no link between the face lip-sync animated and the face
  restoration rebuilt.
- **Do not measure a restoration pass by diffing the two rendered videos.**
  They come out of different H.264 encoders, so ~98% of pixels differ before
  the model has done anything, and on slide-heavy content plenty of that
  clears any sane noise threshold. Measured video-to-video, the 2026-10-12 NAG
  run reported the edit covering 84% of the frame and the face getting *less*
  sharp; the same pair measured PNG-to-PNG showed it confined to a 139x131 box
  (2.0% of frame) with +36% detail. Run with `--restore-keep-frames` and use
  `scripts/compare_restore.py --frames <work_dir>`, which has no encoder in
  the loop. The video-to-video path prints a warning saying so.
- **Two layers can fit the same clip to the same slot, and both default on
  is a double stretch.** `backend/tts.py` stretches each clip to
  `target_duration_s` (±`--tts-max-stretch`, default 0.10), and a timing plan
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
  pointing at `BODHAN_TTS_RATE_FIELD` (try `rate`, or `instructions.speed`
  to nest it). Do not treat `--tts-speed-control` output as validated until
  one rated call has actually returned 200 on this host.

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

  `--tts-speed-control` does NOT disable the post-hoc WSOLA stretch, and
  that is deliberate — unlike the double-fit above, these converge on one
  target rather than compounding: the engine gets close, the stretch mops up
  the residual within `--tts-max-stretch`. Pass `--no-tts-stretch` for a
  clean method-1-vs-method-2 comparison.
  The agent says so in its plan rather than letting it pass silently.
  `tests/test_edit_agent.py` asserts `backend/tts.py` still reads every key
  `backend/edits.py` writes — the one cheap guard against this pair drifting
  back into the dead-code shape that put a female voice on a male lecture.
- **Nothing gates lip-sync on who is actually speaking.** `backend/lipsync.py::
  run_lipsync` hands the whole video to the adapter, and the adapters animate
  one face per frame with no identity tracking — Wav2Lip literally takes the
  first S3FD detection (`d = d[0]`, `third_party/Wav2Lip/face_detection/
  api.py:73`). On a multi-face shot that is an arbitrary face, and it can flip
  between frames. `scripts/clip_asd_musetalk_run.py` works around this by
  cutting on shot boundaries and lip-syncing only shots with a single speaking
  face, but the gating lives in that script, not in the stage.
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
     predicted genders computed and discarded. Pass `--voice-engine`
     explicitly to override.
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
  order. Two traps, in order of cost:
  1. **A clip of some other human filed under `Amit.wav` is worse than no
     clip at all.** The cosine then compares the speaker against the wrong
     timbre and returns a confident similarity, so the log looks like the
     feature is working. Re-render with `scripts/build_voice_refs.py`; do not
     source these from the internet. (A zero-shot engine whose "voices" ARE
     reference clips — IndicF5 takes `ref_audio_path` + `ref_text` — is a
     different catalog with a different provenance story, not more files in
     this directory. `load_voice_catalog("indicf5")` returns 0 voices today.)
  2. `scripts/build_voice_refs.py` had no `load_dotenv()` call, so run
     directly it saw no `BODHAN_TTS_KEY` and every render failed on auth —
     which reads like an upstream outage rather than a missing line. Fixed
     2026-10-17. `backend/cli.py`, `edit_agent.py` and `experiment.py` each
     call it for themselves; a new entry point under `scripts/` must too.
  Expect 429s throughout a full 45-voice run; the adapter backs off and the
  script is idempotent, so a re-run fills only the gaps.

- **Bodhan endpoints return `internal_error` 5xx during upstream outages** —
  this is not rate limiting, and since Edge was removed there is no second
  engine to fall back to. `bodhan_adapter.synthesize` catches per request, so
  an outage does not raise: every segment comes back with `tts_error` set and
  the stage returns green with `extra.tts.errors == len(segments)`. Check that
  count, not the exit code. A missing `BODHAN_TTS_KEY` is the louder case —
  `_ensure_loaded` raises before the first request.
- **ASD scores produced before 2026-10-11 are worthless.** `talknet_adapter.py`
  fed `soundfile` floats in [-1, 1] into an MFCC front-end that upstream trains
  and infers on int16 PCM, which shifted every log-energy cepstral term and
  pinned all scores near zero. Measured on one 3 s single-speaker close-up:
  mean 0.044 / max 0.129 / 0 of 75 frames above 0.5 before the fix, versus
  mean 0.411 / max 0.798 / 31 of 75 after. The stage never errored — it just
  answered "nobody is speaking" every time, so any archived `assignments.json`
  or on-screen count from an earlier run should be regenerated, not trusted.

---

## 6. Keeping the docs honest

Three files, three different owners. Knowing which is which prevents the usual
failure, where someone hand-edits a generated table and their work is erased on
the next run.

| File | Owned by | Covers |
|---|---|---|
| **README.md** — generated block | `scripts/repo_checkup.py` | structure: layout, modules, registries, stages, scripts, skills |
| **README.md** — prose outside it | you | what the project is, how a run is organised, why |
| **CHANGES-PENDING.md** | script writes, you clear | changes awaiting a narrative |
| **CLAUDE.md** (this file) | you | host behaviour, conventions, traps |

### The scheduled checkup

A cron job runs every 5 hours:

```
17 */5 * * *  →  scripts/repo_checkup.py  →  logs/repo_checkup.log
```

It snapshots the structure, diffs it against `.repo-state.json`, and when
something moved it rewrites README's generated block and appends what changed to
`CHANGES-PENDING.md`. When nothing moved it exits silently, so the common case
costs nothing. Run it by hand any time:

```bash
$PY scripts/repo_checkup.py --dry-run   # report drift, write nothing
$PY scripts/repo_checkup.py             # update README + pending buffer
$PY scripts/repo_checkup.py --clear     # empty the buffer once folded in
```

### When you find CHANGES-PENDING.md non-empty

The generated block is already correct — the structure table does not need your
help. What the buffer is asking for is the part a diff cannot produce: *why* it
changed, and what it means for someone using the repo.

Read the entries, fold anything that deserves explanation into README's prose or
into this file, then run `--clear`. If an entry needs no narrative — a renamed
helper, a deleted scratch file — clearing it unexplained is the right call. The
buffer is a prompt, not a backlog to discharge in full.

### When to edit this file

The checkup cannot see any of what §1–§5 describe, so those sections only stay
true if you maintain them. Update this file when you:

- add, rebuild or move a conda env or interpreter
- change hardware, drivers, CUDA, or an ffmpeg build
- add an env var or secret to `.env.example`
- change a convention — job layout, where artifacts go, how models are registered
- add or materially change a skill in `.claude/skills/`
- fix something in §5, or hit a new trap worth warning the next person about

Edit the affected section rather than appending notes at the end, and verify
against the live system before writing a fact down — query the registry, run
`nvidia-smi`, read the log. A contradicted line is worse than a missing one, so
when something stops being true, delete it.
