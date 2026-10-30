# Dubbing Studio

Indic video dubbing pipeline. Takes a video — a YouTube URL or a local file —
transcribes it, translates it, synthesizes a dubbed track in an Indian language,
mixes it against the original music and effects, and optionally lip-syncs the
speaker's mouth to the new audio.

- Design intent, the experiment plan and the current file structure: [plan.md](plan.md)
- Host setup and working rules: [CLAUDE.md](CLAUDE.md)
- Stage internals, the UI, speed modes, casting: [docs/STAGES.md](docs/STAGES.md)
- Known bugs and incident history: [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md)


dev start cmd conda run --no-capture-output -n dubui npm --prefix ui/web run dev
server start cmd $PY -m ui.server --port 7861
## Quickstart

```bash
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python   # no `python` on PATH

bash setup.sh                                 # doctor, model fetch, 60s dry run

$PY -m backend.cli --input <url-or-file> \
    --model pyannote --asr bodhan --align mms_fa \
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
  asr/ align/ mt/ voices/ tts/ mix/ lipsync/ eval/ exp/
```

`<job_id>` is `<timestamp>__<input-slug>__<diar>.<asr>.<align>.<mt>.<lang>`, so
runs sort chronologically and a re-run never overwrites an earlier one.
`manifest.json` carries `status`: `running`, then `ok` or `failed`.

Three inputs live outside job folders on purpose:
`models/duration/<lang>.<voice>.json` (costly TTS calibration),
`experiments/suites/*.json` (sweep definitions), and
`data/voice_refs/<engine>/<voice>.wav` (one clip per catalog voice, used to
match timbre during casting). The voice clips are gitignored; rebuild them with
`$PY scripts/build_voice_refs.py` (idempotent, expect 429s).

Full layout: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

### Dubbing one window of one video

`backend/cli.py` runs whole files, so a time-window run means cutting first
(see the `clip-run` skill). `scripts/clip_asd_musetalk_run.py` packages the
common case: one window, driven by an already-downloaded YouTube dub track (MT
and TTS skipped), cut on shot boundaries, lip-synced with MuseTalk only on
shots where ASD finds a single speaking face, and spliced back to the original
length. Every visual stage dumps what it detected so you can check it.

Launch it detached with `scripts/run_clip_asd_musetalk.sh` (tmux `dubsync`).
It writes `STATUS.json` per phase.

## Defaults worth knowing

| What | Default | Why / detail |
|---|---|---|
| MT script | `--mt-target-script codemix`: English terms stay in Latin, e.g. `weight सही है` | `native` re-scripts inconsistently, which TTS mispronounces. [STAGES.md](docs/STAGES.md) |
| Pacing | `--speed-mode segment`: each cue fitted to its own slot, `--speed-cap` 1.8, `--speed-floor` 0.75 | `global` (one `atempo` over the whole file) is still available. [STAGES.md](docs/STAGES.md) |
| Over-long lines | Stage 9c2 rewrites a line past 1.5× (`--condense-overlong`, on in the UI) | [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |
| TTS | Bodhan `indic-speak` only; the voice catalog follows from `--tts`, there is no `--voice-engine` | Edge removed 2026-10-17, IndicF5 2026-10-28. [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |
| Voice casting | Filter by predicted gender, rank by ECAPA timbre; fall back to target language + gender | [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |
| Lip-sync model | wav2lip when the face is under 25% of frame height, MuseTalk above | [STAGES.md](docs/STAGES.md) |
| Speaking face | Picked from stage 6 ASD scores via `run_lipsync(..., coord_list=…)` | [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |
| Face restore | GFPGAN, in `models/restore/` | The lip-synced mouth comes back soft. [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |

## The operator UI

The same pipeline with a screen on it: paste a YouTube link, pick a language
and a part of the video, watch it run, then listen and recast voices.

```bash
$PY -m ui.server                       # http://127.0.0.1:7861
```

It binds **loopback only** and has **no authentication**. Reach it over an SSH
tunnel (`ssh -L 7861:localhost:7861 <host>`). Don't bind a public address: it
runs yt-dlp on any URL it's given and starts GPU jobs.

| | |
|---|---|
| **New job** | URL, target language, whole video or a time range. The server pre-cuts the clip. |
| **Progress** | Eight pills: Download video · Speaker annotation · Transcribe · Translate · Speech generation · Mix · Lip sync · Finished |
| **Video / audio** | Original ⇄ lip-synced toggle; play and download both tracks |
| **Speakers** | Each speaker's voice and tone, changeable |
| **Edit panel** | Plain-language edits, under the player (see below) |
| **Jobs** | A rail of job cards titled by video title |

Two limits on purpose: lip-sync is capped at **20 minutes** of video (it runs
at 3–21× clip length here), and a voice change **doesn't re-render** until you
apply it, since it re-runs TTS, mix and lip-sync.

### Building the frontend

Vite app in `ui/web/`, built to `ui/web-dist/`. Node lives in its own conda
env, and bare `npm` is broken on this host (see [docs/HOST.md](docs/HOST.md)):

```bash
conda create -n dubui nodejs=20
conda run -n dubui npm --prefix ui/web install
conda run -n dubui npm --prefix ui/web run build
```

If `ui/web-dist/` is missing, the server runs API-only and logs a warning.

## Editing a finished dub in plain language

Fixing one line shouldn't mean re-running the whole pipeline.
[backend/edit_agent.py](backend/edit_agent.py) takes the change as a sentence:

```bash
$PY -m backend.edit_agent -j <job_id> \
    "the speaker at 1:20 should sound female, and speed up 0:30 to 0:45 by 1.2x"
```

It prints a plan and stops. `--apply` writes, `--run` re-renders, `--revert`
undoes every edit. Only the stages the change broke are replayed; stages
1..9a never re-run.

| Op | Example |
|---|---|
| `set_text` | `change the line at 1:20 to "…"` |
| `retranslate` | `retranslate 0:30–0:45, it is too literal` |
| `set_tone` | `make the tone more formal after 1:00` |
| `code_mix` | `code-mix the translation, keep gradient and weight in English` |
| `set_gender` | `the speaker at 1:20 should sound female` |
| `set_voice` | `use Amit for SPEAKER_00` |
| `set_speaker` | move a mis-attributed cue to another speaker |
| `set_speed` | `speed up the audio between 0:30 and 0:45 by 1.2x` |
| `lipsync_window` | `lip-sync only from 1:00 to 1:30 with musetalk` |

`--list-ops` prints every parameter. The LLM (Gemma 3 12B via OpenRouter)
only parses the request and rewrites text; without a key a regex parser runs
and the text-rewriting ops refuse. Full detail:
[docs/EDIT_AGENT.md](docs/EDIT_AGENT.md).

## Adding a model

Write one adapter against the stage's `base.py` and add one line to
`_REGISTRY_SPEC` in `models/<stage>/__init__.py`. The CLI picks it up
automatically. The registered adapters per stage are listed in
[plan.md §11](plan.md#11-repository-structure).

## Structure

Hand-written, not regenerated by any script. Update it when files are added,
moved or removed. The same listing is in [plan.md §11](plan.md#11-repository-structure);
keep the two in step.

*As built, 2026-10-27. Descriptions come from each file's own docstring.*
**Scope:** `backend/`, `models/`, `ui/` (Python side) and `tests/` (docstring
headers only). Not covered: `scripts/` and the web frontends. The
`backend/speed.py` and `backend/restore.py` rows come from CLAUDE.md and the
earlier README, not from reading those files.

### Top level

| Path | What it is |
| --- | --- |
| `README.md` | Quickstart, defaults, UI and edit-agent usage, and this structure listing |
| `CLAUDE.md` | Host setup (interpreters, GPUs, ffmpeg builds), conventions, working rules for Claude Code |
| `plan.md` | Design intent, experiment matrix, and the same structure listing |
| `docs/` | Long-form docs: `HOST.md`, `STAGES.md`, `ROUGH_EDGES.md`, `EDIT_AGENT.md`, `PROMPTS.md`, `UI_RUN.md` |
| `backend/` | The pipeline: one module per stage or concern (below) |
| `models/` | Model adapters, one folder per stage, each with a registry |
| `ui/` | FastAPI operator UI and its job runner |
| `tests/` | pytest suite, no models or network |
| `scripts/` | Operator and batch scripts: setup, corpus fetch, clip runs, sweeps, repo checkup |
| `data/` | Job folders (`data/jobs/<job_id>/`), voice reference clips, layout doc `DATA_LAYOUT.md` |
| `experiments/` | Sweep suite definitions (`experiments/suites/*.json`) |
| `third_party/` | Legacy clones of upstream repos (LatentSync, LipGAN, MuseTalk, TalkNet-ASD, Wav2Lip, Wav2Lip-HD, langfuse); lip-sync and ASD upstreams are moving into `models/*/vendor/` |
| `setup.sh`, `requirements.txt` | Doctor, model fetch and 60 s dry run; Python dependencies |
| `comparison-results.md`, `lipsync-history.md` | Results notes: timing/stretch comparison and lip-sync history |

### `backend/` — pipeline modules

Entry points and orchestration

| File | Features |
| --- | --- |
| `cli.py` | `python -m backend.cli`: argparse front end (79 flags) for a whole run |
| `pipeline.py` | End-to-end orchestrator, 26 logged stages from download to the 10b MT check; `run_post_mt_stages` (9b to 9h) is what edit replays re-run |
| `jobs.py` | Job-folder layout and `manifest.json` (status, config, stage inventory); `python -m backend.jobs list/show` |
| `experiment.py` | Runs one clip through several dubbing variants with the upstream prefix (download to MT) cached once and TTS reused where only placement changes |
| `gpu.py` | GPU discovery, free-card check, and ffmpeg sharding helpers for multi-GPU lip-sync |
| `tracing.py` | Langfuse tracing around the LLM client; never raises, flushes at exit |

Input and audio prep (stages 1 to 3)

| File | Features |
| --- | --- |
| `downloader.py` | yt-dlp download of video, captions and uploader metadata (`<stem>.meta.json`) |
| `media_io.py` | ffmpeg/ffprobe wrapper: demux, 16 kHz mono extraction, duration and fps probes |
| `stem_separation.py` | HTDemucs split into dialogue and music-and-effects stems |

Speakers and faces (stages 4 to 6)

| File | Features |
| --- | --- |
| `diarize.py` | Entry point over the diarization registry (pyannote community-1) |
| `segment.py` | Stage 4b: splits long turns at VAD gaps, merges short ones, caps length at energy valleys |
| `speaker_id.py` | Stage 4c: ECAPA check that one label holds one person; splits impure labels |
| `faces.py` | Face detection and tracking entry point; preloads insightface before other libs claim shared C libraries |
| `asd.py` | Active speaker detection entry point (`score_asd`) |
| `assign_faces.py` | Hungarian matching of diarized turns to face tracks; `speaker_face_tracks` picks each speaker's face |
| `lipsync_targets.py` | Per-frame speaking-face boxes from ASD scores, handed to MuseTalk as a saved-coordinate pickle |

Transcript and translation (stages 7 to 9b2)

| File | Features |
| --- | --- |
| `asr.py` | ASR entry point (Bodhan Indic-Transcribe) |
| `caption_asr.py` | Uses human-authored YouTube captions as the transcript, reshaped onto diarization turns |
| `align.py` | CTC forced alignment, fills word timestamps per segment |
| `translate.py` | MT entry point (Bodhan Indic-Translate), writes the merged ASR+MT JSON |
| `sanitize.py` | Text clean-up at the ASR to MT, MT output and TTS boundaries |
| `codemix.py` | Stage 9b: keeps English terms, transliterates them into the target script, normalises Latin leftovers |
| `classify.py` | Stage 9b1: picks the whole-video TTS register from YouTube categories, channel, description, then LLM |
| `emotion.py` | Stage 9b2: per-segment delivery style from source-speaker emotion, with confidence and run-length guards |
| `transcripts.py` | Parses ground-truth `.vtt`/`.srt` and scores ASR and MT (WER, CER, BLEU, chrF, ROUGE, METEOR) |
| `llm.py` | OpenRouter chat client, the only LLM caller; falls back to deterministic paths with no key |

Duration, voices and synthesis (stages 9c to 9e)

| File | Features |
| --- | --- |
| `duration.py` | Stage 9c: fitted duration model, candidate ranking, WSOLA stretch with the ±10% cap |
| `gender.py` | Speaker gender by majority vote of wav2vec2, F0 pitch and face genderage |
| `voices.py` | Stage 9d: per-speaker voice casting (gender filter, screen time, ECAPA timbre rank) |
| `tts.py` | Stage 9e: per-segment Bodhan synthesis with per-segment overrides, writes `tts_audio` and stretch data |

Timing, mix and lip-sync (stages 9f2 to 9h)

| File | Features |
| --- | --- |
| `timing.py` | Timing-plan contract: `Piece`, `TimingPlan`, and the `naive_slot` and `pause_aware` producers |
| `timing_advanced.py` | `word_anchored` (M4) and `dp` (M7) timing producers, registered onto `timing` |
| `stretch.py` | Named time-stretch backends (rubberband, atempo, linear control) chosen by filter availability |
| `mix.py` | Overlays dub clips on the M&E stem; strategies `global_tempo`, `no_tempo`, `timing_plan`; backs `--speed-mode` |
| `speed.py` | `--speed-mode segment` knobs (cap, floor), dependency-free |
| `vad.py` | Speech / internal-pause segmentation (pyannote, energy fallback); read by `pause_absorb.py`. Lifted out of the deleted `backend/eval/` on 2026-10-28 |
| `lipsync.py` | Stage 9h entry point: mux, shard over free GPUs, run the chosen adapter, concat; refuses to report success on a missing render |
| `restore.py` | Face restoration pass (GFPGAN) over a finished render |

Editing

| File | Features |
| --- | --- |
| `edits.py` | Typed `EditOp`s over the merged JSON and the stages each one invalidates; deterministic, no LLM |
| `edit_agent.py` | Plain-language edit CLI: parses a sentence into ops (LLM or regex), prints the plan, `--apply`, `--run`, `--revert` |

### `models/` — adapters

Each stage folder holds a `base.py`, one adapter per model, and an `__init__.py` with a `_REGISTRY_SPEC` dict.

Shared files

| File | Features |
| --- | --- |
| `_registry.py` | `make_registry`: lazy adapter import, lister, kwarg filtering, `get_*` |
| `_bodhan.py` | Shared Bodhan client parts: env resolution, auth session, pacing floor, `Retry-After` backoff (retry loops stay in each adapter) |
| `_vendor.py` | `resolve_vendor_repo`: finds a vendored upstream under `models/*/vendor/` or the legacy `third_party/` |

Stages and adapters (registered names in bold)

| Stage | Adapter file | Features |
| --- | --- | --- |
| `align/` | `mms_fa_adapter.py` (**mms_fa**) | torchaudio MMS forced aligner, char-level CTC; romanizes Indic text with `uroman`, skips non-Latin segments if it's missing |
| `asd/` | `loconet_adapter.py` (**loconet**) | Joint multi-speaker scoring over 112x112 crops plus VGGish audio -- the only ASD model here that weighs competing faces against each other. `_av_frontend.py` holds the shared crop and VRAM-windowing code |
| `asr/` | `bodhan_adapter.py` (**bodhan**) | Hosted Indic-Transcribe over REST; `render_mode` native, mixed or roman; adapts concurrency from rate-limit headers |
| `diarization/` | `pyannote_adapter.py` (**pyannote**) | pyannote community-1; needs an HF token and the gate accepted |
| `emotion/` | `emotion2vec_adapter.py` (**emotion2vec**) | emotion2vec+ large via FunASR; utterance-level emotion labels |
| `faces/` | `detect/` (**retinaface**, **mediapipe**), `track/` (**bytetrack**) | Per-frame detection, then ByteTrack identities; `base.py` holds `FaceBox`, `FaceDetections`, `FaceTracks` |
| `lipsync/` | `base.py`, `wav2lip_adapter.py` (**wav2lip**), `musetalk_adapter.py` (**musetalk**), `latentsync_adapter.py` (**latentsync15**) | Each shells into its own conda env and returns a rendered video; LatentSync 1.5 is 256 px (~8 GB) |
| `mt/` | `bodhan_adapter.py` (**bodhan**) | Hosted Indic-Translate, one call per segment; target script native, roman or codemix |
| `restore/` | `gfpgan_adapter.py` (**gfpgan**) | GFPGAN v1.4 plus Real-ESRGAN over a finished render via `backend/face_enhance.py`; stage 9i |
| `translit/` | `bodhan_adapter.py` (**bodhan**) | Hosted Indic-Transliterate; falls back to the MT endpoint |
| `tts/` | `bodhan_adapter.py` (**bodhan**) | Hosted Indic-Speak, returns WAV bytes; re-speaks at a different rate; the only stock-voice engine |
Removed registry entries: `wav2lip_hd` and `lipgan` (2026-10-23; restoration is now `--restore gfpgan`), `latentsync` (1.6, on 2026-10-28 — ~18 GB against 16 GiB cards), `openvoice` and `vevo`, and `ts_talknet`. All three retired display rows were dropped from `ui/lipsync_models.FACTS` on 2026-10-28 too, so the three `hd_*` job folders now label as plain Wav2Lip. Voice conversion went with them: `models/vc/`, `backend/voice_convert.py`, stage 9f and the `--vc` flag were deleted on 2026-10-28, since the registry had been empty for five days and no run on disk ever set it. Adapter files for `asd/` and `faces/` and the Bodhan base classes were not opened; their rows come from the registry docstrings.

### `ui/` — operator UI (Python side)

| File | Features |
| --- | --- |
| `server.py` | FastAPI app on loopback port 7861; serves `ui/web-dist/` and the job APIs |
| `runner.py` | Detached process that runs one job from `request.json`; fetch, cut, probe, then the pipeline; `--rerun` for edit replays |
| `progress.py` | Logging handler that turns the pipeline's `=== stage N ===` markers into `progress.json` for the 8-pill bar |
| `presets.py` | Three run presets (`draft`, `standard`, `best`), pipeline kwargs, runtime estimates |
| `routes_create.py` | Job creation: URL host allowlist, probe, window and 20-minute lip-sync cap checks, launch |
| `routes_edits.py` | Stage, price, apply and revert voice and tone edits |
| `routes_enhance.py`, `enhance.py` | Stage 9i "Enhance resolution" on a finished render, run on a background thread with a poll endpoint |
| `routes_talkinghead.py` | Talking Head page: photo face check, dialogue catalogue, render queue |
| `_io.py` | Best-effort `read_json`, `env_int`, `env_float`, `data_dir` |
| `media.py` | Picks the files the player gets (original and lip-synced video, original and dubbed audio) from the manifest first, then a ranked glob; returns `/files/...` URLs |
| `speakers.py` | Rebuilds the speaker table (who speaks, which voice) from voices JSON, merged JSON, per-segment overrides, or a replay of stage 9d |
| `jobqueue.py` | Serial job queue whose state is read from `progress.json` on disk, so a UI restart can't launch a second job onto a busy GPU (`DUBBING_UI_MAX_CONCURRENT`, default 1) |
| `ytmeta.py` | Probes a URL before the job starts: title, duration, live-stream refusal, host re-check; caches the thumbnail |
| `clip.py` | Cuts the chosen window with ffmpeg (re-encode, `-ss` before `-i`) so the pipeline gets a local clip |
| `uploads.py` | Local video upload: extension and size limits, server-chosen filename, ffprobe check |
| `langs.py` | Language list built from the adapters, so only languages the TTS engine can speak are offered |
| `talking_head.py` | Talking Head logic: face check on the photo, downscale of large stills, dialogue catalogue from `data/talking_head/dialogues.json` |
| `lipsync_models.py` | One table of per-model facts (runtime factors, cost rates, labels) the other UI modules read; retired models stay as display-only rows |
| `spec.py` | The job's settings as plain label, value and hint lines, plus the raw config for an "Advanced" view |
| `cost.py` | Turns an edit's dirty stages into what re-runs, what doesn't, and an estimated time |

`ui/auth.py` is deleted in the working tree. The frontends (`ui/web/`, `ui/web-dist/`) were not read, and there may be other `ui/` modules I haven't seen.

### `tests/`

| File | Covers |
| --- | --- |
| `conftest.py` | Shared fixtures: tiny generated WAVs, a mini merged JSON |
| `test_caption_asr.py` | Caption transcripts keep speaker labels and `SEG_NNNN` ids |
| `test_coord_cache.py` | Face-box cache keyed by content, not filename |
| `test_edit_agent.py` | Edits reach the stage that must act on them |
| `test_experiment_isolation.py` | Sweep variants never write into each other or into `data/` |
| `test_face_import_order.py` | insightface loads before pyannote's libjpeg |
| `test_gender.py` | Three-way gender vote |
| `test_gpu_pick.py` | Free-GPU selection policy |
| `test_lipsync_env_python.py` | MuseTalk and LatentSync find their own interpreter |
| `test_lipsync_failure_honesty.py` | A lip-sync that produced nothing is not reported as success |
| `test_lipsync_targets.py` | ASD-gated speaking-face boxes |
| `test_memguard.py`, `test_mt_versions.py` | Memory guard; MT version snapshots never become the live document (not opened, named from git status) |
| `test_musetalk_frame_parity.py` | Faceless frames are copied through, not dropped |
| `test_register_to_tts.py` | Stage 9b1 register reaches the Bodhan payload |
| `test_speaker_face_tracks.py` | Speaker to face track by on-screen seconds |
| `test_speaker_id.py` | Stage 4c clustering decisions |
| `test_speed_mode.py` | `segment` fills slots; `global` stays put |
| `test_staged_review.py` | Cast review, cue review, then lip-sync |
| `test_stretch.py` | Stretch backends change time, not pitch |
| `test_talking_head.py` | Talking Head page gating |
| `test_timing_plan.py` | Timing contract and the DP's degenerate cases |
| `test_tracing.py` | Tracing never breaks a run |
| `test_ui_backend_sync.py` | UI kwargs, manifest config and progress stay in step with the pipeline |
| `test_ui_create.py` | Job creation refusals and allowlist |
| `test_ui_edits.py` | Voice and tone edit pricing |
| `test_ui_errors.py` | Actionable failure messages |
| `test_ui_media.py` | Player gets the right video and audio file |
| `test_ui_progress.py` | Stage markers still match `pipeline.py` |
| `test_ui_queue.py` | One job at a time at `max_concurrent=1` |
| `test_ui_register_row.py` | Chosen register is persisted and shown |
| `test_ui_server.py` | Routing, file serving, bind guard |
| `test_ui_speakers.py` | Speaker table shows who got which voice |

### `scripts/`

Header comments read (first 15 lines of each). Several scripts hard-code the repo path and
interpreter, and the per-clip runners pin specific videos and windows (IIT panel, NAG lecture).

Setup and launch

| File | Features |
| --- | --- |
| `start_ui.sh` | Starts the operator UI with its own interpreter and libstdc++ preload; `--port` to change it; no auth |
| `ui_run_job.sh` | Launches one UI-submitted job detached from `data/jobs/<job_id>/request.json`; the only launcher that sets the libstdc++ preload |
| `setup.py` | Old doctor, model fetch and dry run; superseded by `scripts/doctor.py` and `scripts/fetch_assets.py` |
| `audit_edit_ops.py` | Runs every phrasing the edit agent advertises (op examples + quick-action fills, bare and cue-scoped) through the rule parser and prints the op and scope each produces; no network, no job folder |
| `move_lipsync_vendor.sh` | Moves the lip-sync upstreams (~21 GB) from `third_party/` to `models/lipsync/vendor/`; the code works before and after |
| `archive_data.sh` | Packs every `data/` folder except `data/jobs` into verified `data/_archive/*.tar.zst`, then deletes the originals |

Fetching and preparing media

| File | Features |
| --- | --- |
| `fetch_corpus.py` | Downloads the timing-experiment corpus: video, every language audio track, every subtitle, with a provenance `meta.json` |
| `fetch_indic_audio.py` | Downloads the Indic-language audio tracks (official dubs) of a YouTube video |
| `fetch_indic_captions.py` | Fetches YouTube auto-translated captions one language at a time, with backoff on 429 |
| `fetch_captions_runner.sh` | Runs the caption fetch for 15 Indic languages over two videos, logging to `logs/` |
| `extract_corpus_audio.sh` | Makes a 16 kHz mono WAV next to each downloaded audio track |
| `select_clips.py` | Chooses golden-set windows by speech fraction, pauses, speaker turns and rate variation |

Building a dub for one clip

| File | Features |
| --- | --- |
| `build_clip_cues.py` | Cue-timed source text for a clip window, from YouTube auto-caption word timestamps or from ASR |
| `clip_dub_audio.py` | Translates a clip's cues and synthesises a dub track with each cue placed at its own start |
| `run_gt_hindi_dub.py` | Hindi dub audio (ASR, MT, code-mix, TTS) for two 30 s clips of each ground-truth video |
| `calibrate_tts_duration.py` | Synthesises Hindi calibration text and fits the per-(language, voice) duration model |
| `build_voice_refs.py` | Renders one reference clip per catalog voice so casting can compare timbre |
| `hi_cue_floor.py` | Cue-level timing check of manual Hindi caption onsets against WhisperX word onsets |

Lip-sync runs

| File | Features |
| --- | --- |
| `clip_asd_musetalk_run.py` | One window of one video: faces, ASD, shot gating, mix, MuseTalk on single-speaker shots, splice back |
| `run_clip_asd_musetalk.sh` | Detached launcher for the above; writes `STATUS.json` per phase |
| `run_lipsync.sh` | Runs one lip-sync adapter on an already-dubbed mix, waiting for a free GPU |
| `run_lipsync_only.py` | Runs a single lip-sync adapter on an existing mix output by stem |
| `lipsync_from_dub.py` | Stage 9h only, against an existing dub video; no ASR, MT or TTS |
| `run_iit_musetalk_indic15.sh` | IIT panel clip, 15 Indic languages, ASD-gated MuseTalk |
| `rerun_iit_musetalk_revoiced.sh` | Re-runs that lip-sync against the existing female-voice TTS audio |
| `run_nag_wav2lip_indic15.sh` | NAG lecture clip, 15 Indic languages, Wav2Lip |
| `revoice_lipsynced.py` | Re-synthesises a dub in the measured speaker gender and swaps it onto a finished render (audio-only repair) |
| `regroup_lipsync_into_jobs.py` | Moves flat `data/lipsync/` and `data/mix/` renders into per-language job folders |
| `face_enhance.py` | GFPGAN (+ optional Real-ESRGAN) over a rendered video; what `models/restore/` shells into |
| `wav2liphd_enhance.py` | Older Real-ESRGAN/GFPGAN frame pass ported from Wav2Lip-HD; `face_enhance.py` is its renamed successor |

Sweeps, benchmarks and ops

| File | Features |
| --- | --- |
| `run_sweep.sh` | Runs the eight-methods timing suite over the golden set, one experiment per clip |
| `benchmark_gender.py` | Benchmarks the voice-gender classifier against the face-gender label on a 90 s clip |
| `gpu_watcher.py` | Holds a command until both GPUs have been free for N minutes |
| `langfuse_usage.py` | LLM usage, cost and latency per operation from the Langfuse REST API |

Not found on disk: `repo_checkup.py`, `setup_wav2liphd.py` and `run_wav2liphd_clips.sh`. The docstrings
also name `scripts/doctor.py`, `scripts/install.sh`, `scripts/fetch_assets.py` and `scripts/ab_refactor.py`;
I didn't open them, so they aren't listed.
