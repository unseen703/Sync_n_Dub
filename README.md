# Dubbing Studio

Indic video dubbing pipeline. Takes a video — a YouTube URL or a local file —
transcribes it, translates it, synthesizes a dubbed track in an Indian language,
mixes it against the original music and effects, and optionally lip-syncs the
speaker's mouth to the new audio.

- Design intent, the experiment plan and the current file structure: [plan.md](plan.md)
- Host setup and working rules: [CLAUDE.md](CLAUDE.md)
- Stage internals, the UI, speed modes, casting: [docs/STAGES.md](docs/STAGES.md)
- Known bugs and incident history: [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md)


dev start cmd conda run --no-capture-output -n dubui npm --prefix frontend/web run dev
server start cmd $PY -m frontend.server --port 7861
## Quickstart

```bash
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python   # no `python` on PATH

bash setup.sh                                 # doctor, model fetch, 60s dry run

$PY -m backend.cli --input <url-or-file> \
    --model pyannote --asr bodhan --align mms_fa \
    --mt bodhan --mt-target-language hi \
    --tts bodhan --mix --lipsync wav2lip

$PY -m backend.jobs list                      # every run, newest first
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
match timbre during casting). The voice clips are gitignored.

Full layout: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

### Dubbing one window of one video

`backend/pipeline/cli.py` runs whole files, so a time-window run means cutting first
(see the `clip-run` skill). The common case: one window, driven by an
already-downloaded YouTube dub track (MT and TTS skipped), cut on shot
boundaries, lip-synced with MuseTalk only on shots where ASD finds a single
speaking face, and spliced back to the original length. Every visual stage
dumps what it detected so you can check it.

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
| Face restore | GFPGAN, in `backend/restore/` | The lip-synced mouth comes back soft. [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |

## The operator UI

The same pipeline with a screen on it: paste a YouTube link, pick a language
and a part of the video, watch it run, then listen and recast voices.

```bash
$PY -m frontend.server                       # http://127.0.0.1:7861
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

Vite app in `frontend/web/`, built to `frontend/web-dist/`. Node lives in its own conda
env, and bare `npm` is broken on this host (see [docs/HOST.md](docs/HOST.md)):

```bash
conda create -n dubui nodejs=20
conda run -n dubui npm --prefix frontend/web install
conda run -n dubui npm --prefix frontend/web run build
```

If `frontend/web-dist/` is missing, the server runs API-only and logs a warning.

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
**Scope:** `backend/` and `frontend/` (Python side). Not covered: the web
frontends. The `backend/media/speed.py` and `backend/restore/restore.py` rows
come from CLAUDE.md and the earlier README, not from reading those files.

### Top level

| Path | What it is |
| --- | --- |
| `README.md` | Quickstart, defaults, UI and edit-agent usage, and this structure listing |
| `CLAUDE.md` | Host setup (interpreters, GPUs, ffmpeg builds), conventions, working rules for Claude Code |
| `plan.md` | Design intent, experiment matrix, and the same structure listing |
| `docs/` | Long-form docs: `HOST.md`, `STAGES.md`, `ROUGH_EDGES.md`, `EDIT_AGENT.md`, `PROMPTS.md`, `UI_RUN.md` |
| `backend/` | The pipeline: one package per stage or concern (below), each with its adapters and `_REGISTRY_SPEC` |
| `frontend/` | FastAPI operator UI and its job runner |
| `data/` | Job folders (`data/jobs/<job_id>/`), voice reference clips, layout doc `DATA_LAYOUT.md` |
| `experiments/` | Sweep suite definitions (`experiments/suites/*.json`) |
| `third_party/` | Legacy clones of upstream repos (LatentSync, LipGAN, MuseTalk, TalkNet-ASD, Wav2Lip, Wav2Lip-HD, langfuse); lip-sync and ASD upstreams are moving into `backend/*/vendor/` |
| `setup.sh`, `requirements.txt` | Doctor, model fetch and 60 s dry run; Python dependencies |
| `comparison-results.md`, `lipsync-history.md` | Results notes: timing/stretch comparison and lip-sync history |

### `backend/` — pipeline modules

Entry points and orchestration

| File | Features |
| --- | --- |
| `pipeline/cli.py` | `python -m backend.pipeline.cli`: argparse front end (79 flags) for a whole run |
| `pipeline/pipeline.py` | End-to-end orchestrator, 26 logged stages from download to the 10b MT check; `run_post_mt_stages` (9b to 9h) is what edit replays re-run |
| `pipeline/jobs.py` | Job-folder layout and `manifest.json` (status, config, stage inventory); `python -m backend.pipeline.jobs list/show` |
| `experiment.py` | Runs one clip through several dubbing variants with the upstream prefix (download to MT) cached once and TTS reused where only placement changes |
| `common/gpu.py` | GPU discovery, free-card check, and ffmpeg sharding helpers for multi-GPU lip-sync |
| `common/tracing.py` | Langfuse tracing around the LLM client; never raises, flushes at exit |

Input and audio prep (stages 1 to 3)

| File | Features |
| --- | --- |
| `media/downloader.py` | yt-dlp download of video, captions and uploader metadata (`<stem>.meta.json`) |
| `common/media_io.py` | ffmpeg/ffprobe wrapper: demux, 16 kHz mono extraction, duration and fps probes |
| `media/stem_separation.py` | HTDemucs split into dialogue and music-and-effects stems |

Speakers and faces (stages 4 to 6)

| File | Features |
| --- | --- |
| `diarize/diarize.py` | Entry point over the diarization registry (pyannote community-1) |
| `media/segment.py` | Stage 4b: splits long turns at VAD gaps, merges short ones, caps length at energy valleys |
| `speaker_id.py` | Stage 4c: ECAPA check that one label holds one person; splits impure labels |
| `faces/faces.py` | Face detection and tracking entry point; preloads insightface before other libs claim shared C libraries |
| `asd/asd.py` | Active speaker detection entry point (`score_asd`) |
| `assign_faces.py` | Hungarian matching of diarized turns to face tracks; `speaker_face_tracks` picks each speaker's face |
| `asd/lipsync_targets.py` | Per-frame speaking-face boxes from ASD scores, handed to MuseTalk as a saved-coordinate pickle |

Transcript and translation (stages 7 to 9b2)

| File | Features |
| --- | --- |
| `asr/asr.py` | ASR entry point (Bodhan Indic-Transcribe) |
| `caption_asr.py` | Uses human-authored YouTube captions as the transcript, reshaped onto diarization turns |
| `media/align/align.py` | CTC forced alignment, fills word timestamps per segment |
| `translate/translate.py` | MT entry point (Bodhan Indic-Translate), writes the merged ASR+MT JSON |
| `translate/sanitize.py` | Text clean-up at the ASR to MT, MT output and TTS boundaries |
| `codemix.py` | Stage 9b: keeps English terms, transliterates them into the target script, normalises Latin leftovers |
| `classify.py` | Stage 9b1: picks the whole-video TTS register from YouTube categories, channel, description, then LLM |
| `emotion/emotion.py` | Stage 9b2: per-segment delivery style from source-speaker emotion, with confidence and run-length guards |
| `asr/transcripts.py` | Parses ground-truth `.vtt`/`.srt` and scores ASR and MT (WER, CER, BLEU, chrF, ROUGE, METEOR) |
| `common/llm.py` | OpenRouter chat client, the only LLM caller; falls back to deterministic paths with no key |

Duration, voices and synthesis (stages 9c to 9e)

| File | Features |
| --- | --- |
| `media/duration.py` | Stage 9c: fitted duration model, candidate ranking, WSOLA stretch with the ±10% cap |
| `asd/gender.py` | Speaker gender by majority vote of wav2vec2, F0 pitch and face genderage |
| `tts/voices.py` | Stage 9d: per-speaker voice casting (gender filter, screen time, ECAPA timbre rank) |
| `tts/tts.py` | Stage 9e: per-segment Bodhan synthesis with per-segment overrides, writes `tts_audio` and stretch data |

Timing, mix and lip-sync (stages 9f2 to 9h)

| File | Features |
| --- | --- |
| `media/timing.py` | Timing-plan contract: `Piece`, `TimingPlan`, and the `naive_slot` and `pause_aware` producers |
| `media/timing_advanced.py` | `word_anchored` (M4) and `dp` (M7) timing producers, registered onto `timing` |
| `media/stretch.py` | Named time-stretch backends (rubberband, atempo, linear control) chosen by filter availability |
| `media/mix.py` | Overlays dub clips on the M&E stem; strategies `global_tempo`, `no_tempo`, `timing_plan`; backs `--speed-mode` |
| `media/speed.py` | `--speed-mode segment` knobs (cap, floor), dependency-free |
| `media/vad.py` | Speech / internal-pause segmentation (pyannote, energy fallback); read by `pause_absorb.py`. Lifted out of the deleted `backend/eval/` on 2026-10-28 |
| `lipsync/lipsync.py` | Stage 9h entry point: mux, shard over free GPUs, run the chosen adapter, concat; refuses to report success on a missing render |
| `restore/restore.py` | Face restoration pass (GFPGAN) over a finished render |

Editing

| File | Features |
| --- | --- |
| `pipeline/edits.py` | Typed `EditOp`s over the merged JSON and the stages each one invalidates; deterministic, no LLM |
| `pipeline/edit_agent.py` | Plain-language edit CLI (`python -m backend.edit_agent` is still the entry point via a shim); parses a sentence into ops (LLM or regex), prints the plan, `--apply`, `--run`, `--revert` |

### `backend/<stage>/` — adapters

Each stage package holds a `base.py`, one adapter per model, and the `_REGISTRY_SPEC` dict in `__init__.py`. The shared infrastructure lives in `backend/common/`: `registry.py` (`make_registry`: lazy adapter import, lister, kwarg filtering, `get_*`), `bodhan.py` (env resolution, auth session, pacing floor, `Retry-After` backoff — retry loops stay in each adapter), `vendor.py` (`resolve_vendor_repo`: finds a vendored upstream under `backend/*/vendor/` or the legacy `third_party/`).

Stages and adapters (registered names in bold)

| Stage package | Adapter file | Features |
| --- | --- | --- |
| `media/align/` | `mms_fa_adapter.py` (**mms_fa**) | torchaudio MMS forced aligner, char-level CTC; romanizes Indic text with `uroman`, skips non-Latin segments if it's missing |
| `asd/` | `loconet_adapter.py` (**loconet**) | Joint multi-speaker scoring over 112x112 crops plus VGGish audio -- the only ASD model here that weighs competing faces against each other. `_av_frontend.py` holds the shared crop and VRAM-windowing code |
| `asr/` | `bodhan_adapter.py` (**bodhan**) | Hosted Indic-Transcribe over REST; `render_mode` native, mixed or roman; adapts concurrency from rate-limit headers |
| `diarize/` | `pyannote_adapter.py` (**pyannote**) | pyannote community-1; needs an HF token and the gate accepted |
| `emotion/` | `emotion2vec_adapter.py` (**emotion2vec**) | emotion2vec+ large via FunASR; utterance-level emotion labels |
| `faces/` | `detect/` (**retinaface**, **mediapipe**), `track/` (**bytetrack**) | Per-frame detection, then ByteTrack identities; `base.py` holds `FaceBox`, `FaceDetections`, `FaceTracks` |
| `lipsync/` | `base.py`, `wav2lip_adapter.py` (**wav2lip**), `musetalk_adapter.py` (**musetalk**), `latentsync_adapter.py` (**latentsync15**) | Each shells into its own conda env and returns a rendered video; LatentSync 1.5 is 256 px (~8 GB) |
| `translate/` | `bodhan_adapter.py` (**bodhan**) + `translit/` sub-package | Hosted Indic-Translate, one call per segment; target script native, roman or codemix; the `translit/` sub-package carries the Bodhan transliteration adapter |
| `restore/` | `gfpgan_adapter.py` (**gfpgan**) | GFPGAN v1.4 plus Real-ESRGAN over a finished render via `backend/restore/face_enhance.py`; stage 9i |
| `tts/` | `bodhan_adapter.py` (**bodhan**) | Hosted Indic-Speak, returns WAV bytes; re-speaks at a different rate; the only stock-voice engine |

Removed registry entries: `wav2lip_hd` and `lipgan` (2026-10-23; restoration is now `--restore gfpgan`), `latentsync` (1.6, on 2026-10-28 — ~18 GB against 16 GiB cards), `openvoice` and `vevo`, and `ts_talknet`. All three retired display rows were dropped from `frontend/lipsync_models.FACTS` on 2026-10-28 too, so the three `hd_*` job folders now label as plain Wav2Lip. Voice conversion went with them: `backend/vc/`, `backend/voice_convert.py`, stage 9f and the `--vc` flag were deleted on 2026-10-28, since the registry had been empty for five days and no run on disk ever set it. Adapter files for `asd/` and `faces/` and the Bodhan base classes were not opened; their rows come from the registry docstrings.

### `frontend/` — operator UI (Python side)

| File | Features |
| --- | --- |
| `server.py` | FastAPI app on loopback port 7861; serves `frontend/web-dist/` and the job APIs |
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

`frontend/auth.py` is deleted in the working tree. The frontends (`frontend/web/`, `frontend/web-dist/`) were not read, and there may be other `frontend/` modules I haven't seen.
