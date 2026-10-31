<div align="center">

# 🎬 Dubbing Studio

**Indic video dubbing pipeline — from raw video to lip-synced dubbed output in one command.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-research-blue)](#)
[![GitHub](https://img.shields.io/badge/GitHub-Sync__n__Dub-181717?logo=github)](https://github.com/unseen703/Sync_n_Dub)

Takes a video — a YouTube URL or a local file — transcribes it, translates it,
synthesizes a dubbed track in an Indian language, mixes it against the original
music and effects, and optionally lip-syncs the speaker's mouth to the new audio.

**Pipeline:** Download → Diarize → Transcribe → Translate → TTS → Mix → Lip-sync → Restore

</div>

<!-- Screenshots — drop PNGs into docs/screenshots/ and uncomment:
<div align="center">
<img src="docs/screenshots/new-job.png" width="45%" alt="New job" />
<img src="docs/screenshots/progress.png" width="45%" alt="Progress" />
<br/>
<img src="docs/screenshots/edit-panel.png" width="45%" alt="Edit panel" />
<img src="docs/screenshots/speakers.png" width="45%" alt="Speaker casting" />
</div>
-->

---

## ✨ Features

- Full end-to-end dubbing: transcription, translation, speech synthesis and audio mixing
- Lip-sync with Wav2Lip, MuseTalk or LatentSync 1.5 (auto-selected by face size)
- Per-cue pacing — each line is speed-fitted to its original slot
- Active speaker detection picks the speaking face across multi-speaker scenes
- Plain-language edit agent — fix one line without re-running the whole pipeline
- Operator UI with job progress, speaker casting and voice/tone editing
- Multi-GPU support with automatic free-card detection

---

## 📋 Prerequisites

| Requirement | Detail |
|---|---|
| **OS** | Linux (tested on Ubuntu 22.04+) |
| **GPU** | NVIDIA with ≥16 GiB VRAM, CUDA 12.8+ |
| **conda** | Miniconda or Anaconda on PATH |
| **git** | For cloning vendor repos during asset fetch |
| **HuggingFace token** | Required for pyannote diarization — accept the [model gate](https://hf.co/pyannote/speaker-diarization-community-1) first |
| **Bodhan API keys** | ASR, MT, TTS and transliteration endpoints — see `.env.example` |
| **OpenRouter key** | Optional — enables the LLM edit agent and text-rewriting ops |

---

## 🚀 Quick setup (script)

```bash
git clone git@github.com:unseen703/Sync_n_Dub.git dubbingstudio
cd dubbingstudio

setup/setup.sh                    # creates 4 conda envs, fetches vendor
                                  # repos + weights, builds the UI, runs doctor

cp .env.example .env && chmod 600 .env
# Fill in HF_TOKEN, BODHAN_*_KEY, and optionally OPENROUTER_API_KEY

setup/setup.sh --check-only      # verify everything is wired up
```

`setup/setup.sh` delegates to `setup/install.sh`, which is idempotent — re-running after a
partial install only does the remaining work.

### install.sh flags

| Flag | Effect |
|---|---|
| `--skip-envs` | Skip conda env creation (assume they exist) |
| `--skip-assets` | Skip vendor repo + weight fetch (e.g. rsynced in) |
| `--skip-ui` | Skip the npm build |
| `--include-optional` | Also fetch optional weights (~6 GB extra) |

---

## 🔧 Manual setup

### 1. Create the conda environments

Four envs, each from a YAML spec in `envs/`:

```bash
conda env create -f envs/dubbing.yml -n dubbing
conda env create -f envs/musetalk.yml -n musetalk
conda env create -f envs/latentsync.yml -n latentsync
conda env create -f envs/dubui.yml -n dubui        # Node, for the UI build only
```

### 2. Install pip dependencies

Each env has a matching `.pip.txt`. The PyTorch index must target your CUDA:

```bash
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

conda activate dubbing
python -m pip install -r envs/dubbing.pip.txt --extra-index-url "$TORCH_INDEX"

conda activate musetalk
python -m pip install --no-deps -r envs/musetalk.pip.txt --extra-index-url "$TORCH_INDEX"

conda activate latentsync
python -m pip install -r envs/latentsync.pip.txt --extra-index-url "$TORCH_INDEX"
```

MuseTalk also needs mmcv built from source (~20 min):

```bash
conda activate musetalk
envs/build_mmcv_musetalk.sh
```

### 3. Fetch vendor repos and model weights

```bash
conda activate dubbing

python setup/fetch_assets.py --all                  # required weights
python setup/fetch_assets.py --all --include-optional   # +musetalk v1, syncnet (~6 GB)
```

Reads `models/assets.lock.json` — every repo commit, patch and weight checksum
is pinned.

### 4. Configure secrets

```bash
cp .env.example .env && chmod 600 .env
```

Fill in at minimum:
- `HF_TOKEN` — HuggingFace token (pyannote gate must be accepted)
- `BODHAN_ASR_KEY`, `BODHAN_MT_KEY`, `BODHAN_TTS_KEY`, `BODHAN_TRANSLIT_KEY`
- `OPENROUTER_API_KEY` — enables the LLM edit agent

Optional:

- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` — LLM tracing

### 5. Build the frontend

```bash
conda run -n dubui npm --prefix frontend/ ci
conda run -n dubui npm --prefix frontend/ run build
```

Output lands in `frontend/-dist/`. If missing, the server runs API-only.

### 6. Run the doctor

```bash
python setup/doctor.py               # full check
python setup/doctor.py --quick        # skip slow/import-heavy checks
python setup/doctor.py --json         # machine-readable output
```

---

## 🎯 Usage

Activate the dubbing environment once — all commands below assume it's active:

```bash
conda activate dubbing
```

> **Note:** There is no system-wide `python` on this host. `conda activate dubbing` puts the right interpreter on PATH.

### CLI

```bash
python -m backend.pipeline.cli --input <url-or-file> \
    --model pyannote --asr bodhan --align mms_fa \
    --mt bodhan --mt-target-language hi \
    --tts bodhan --mix --lipsync wav2lip

python -m backend.pipeline.jobs list       # every run, newest first
```

One input per run. Each run becomes its own job folder under `data/jobs/`.

### Operator UI

```bash
setup/start_ui.sh                          # http://127.0.0.1:7861
setup/start_ui.sh --port 7900              # custom port
```

Binds **loopback only** — no authentication. Reach it over an SSH tunnel:

```bash
ssh -L 7861:localhost:7861 <host>
```

| Feature | Detail |
|---|---|
| **New job** | URL, target language, whole video or a time range |
| **Progress** | Eight pills: Download → Speaker annotation → Transcribe → Translate → Speech generation → Mix → Lip sync → Finished |
| **Playback** | Original ⇄ lip-synced toggle; play and download both tracks |
| **Speakers** | Each speaker's voice and tone, changeable |
| **Edit panel** | Plain-language edits under the player |
| **Jobs** | Rail of job cards titled by video title |

Lip-sync is capped at **20 minutes** of video (3–21× real-time on 16 GiB cards).

### Edit agent

Fix a finished dub without re-running the whole pipeline:

```bash
python -m backend.edit_agent -j <job_id> \
    "the speaker at 1:20 should sound female, and speed up 0:30 to 0:45 by 1.2x"
```

Dry run by default. `--apply` writes, `--run` re-renders, `--revert` undoes.

| Op | Example |
|---|---|
| `set_text` | `change the line at 1:20 to "…"` |
| `retranslate` | `retranslate 0:30–0:45, it is too literal` |
| `set_tone` | `make the tone more formal after 1:00` |
| `code_mix` | `code-mix the translation, keep gradient and weight in English` |
| `set_gender` | `the speaker at 1:20 should sound female` |
| `set_voice` | `use Amit for SPEAKER_00` |
| `set_speaker` | move a mis-attributed cue to another speaker |
| `set_speed` | `speed up 0:30 to 0:45 by 1.2x` |
| `lipsync_window` | `lip-sync only from 1:00 to 1:30 with musetalk` |

`--list-ops` prints every parameter. The LLM (Gemini 2.5 Flash via OpenRouter)
parses the request; without a key, a regex parser runs and text-rewriting ops
refuse. Full detail: [docs/EDIT_AGENT.md](docs/EDIT_AGENT.md).

---

## 📂 How a run is organised

One run, one folder. Nothing is written outside it:

```
data/jobs/<job_id>/
  manifest.json      status, config, argv, inventory of every file below
  videos/ audio/ stems/ diarization/ faces/ asd/
  asr/ align/ mt/ voices/ tts/ mix/ lipsync/ eval/ exp/
```

`<job_id>` is `<timestamp>__<input-slug>__<diar>.<asr>.<align>.<mt>.<lang>` —
runs sort chronologically and a re-run never overwrites an earlier one.

Three inputs live outside job folders on purpose:
`models/duration/<lang>.<voice>.json` (TTS calibration),
`experiments/suites/*.json` (sweep definitions), and
`data/voice_refs/<engine>/<voice>.wav` (voice reference clips).

Full layout: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

---

## ⚙️ Defaults worth knowing

| What | Default | Detail |
|---|---|---|
| MT script | `--mt-target-script codemix` — English terms stay in Latin | [STAGES.md](docs/STAGES.md) |
| Pacing | `--speed-mode segment` — per-cue fit, cap 1.8×, floor 0.75× | [STAGES.md](docs/STAGES.md) |
| Over-long lines | Stage 9c2 rewrites past 1.5× (`--condense-overlong`) | [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |
| TTS | Bodhan `indic-speak` only; voice catalog follows from `--tts` | [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |
| Lip-sync | Wav2Lip under 25% face height, MuseTalk above | [STAGES.md](docs/STAGES.md) |
| Face restore | GFPGAN post-processing on lip-synced mouth | [ROUGH_EDGES.md](docs/ROUGH_EDGES.md) |

---

## 🧩 Adding a model

Write one adapter against the stage's `base.py` and add one line to
`_REGISTRY_SPEC` in `backend/<stage>/__init__.py`. The CLI picks it up
automatically. Worked example: `backend/lipsync/`.

---

## 🏗️ Project structure

Hand-maintained, not auto-generated. Update when files are added, moved or
removed.

*As built, 2026-10-31.*

### Top level

| Path | What it is |
|---|---|
| `CLAUDE.md` | Host setup, conventions, working rules for Claude Code |
| `setup/` | `setup.sh`, `install.sh`, `doctor.py`, `fetch_assets.py`, `start_ui.sh` |
| `backend/common/ui_run_job.sh` | Detached job launcher (sets LD_PRELOAD) |
| `backend/` | The pipeline: one package per stage, each with adapters and `_REGISTRY_SPEC` |
| `server/` | FastAPI operator UI and its job runner |
| `data/` | Job folders (`data/jobs/<job_id>/`), voice refs, `DATA_LAYOUT.md` |
| `models/` | Vendored upstream repos and model weights |
| `envs/` | Conda env specs (`.yml`) and pip deps (`.pip.txt`) for all four envs |
| `docs/` | Long-form docs (see below) |
| `experiments/` | Sweep suite definitions (`experiments/suites/*.json`) |

### `backend/` — pipeline modules

**Entry points and orchestration**

| Module | Purpose |
|---|---|
| `pipeline/cli.py` | Argparse front end (79 flags) for a whole run |
| `pipeline/pipeline.py` | End-to-end orchestrator, 26 stages from download to MT check |
| `pipeline/jobs.py` | Job-folder layout and `manifest.json` management |
| `pipeline/edit_agent.py` | Plain-language edit CLI — parses a sentence into ops (LLM or regex) |
| `pipeline/edits.py` | Typed `EditOp`s, deterministic handlers, no LLM |
| `common/gpu.py` | GPU discovery, free-card check, multi-GPU sharding |
| `common/llm.py` | OpenRouter chat client — the only LLM caller |
| `common/tracing.py` | Langfuse tracing wrapper |
| `common/registry.py` | `make_registry`: lazy adapter import, lister, kwarg filtering |
| `common/bodhan.py` | `BodhanClientMixin`: env resolution, auth, pacing, `Retry-After` backoff |
| `common/media_io.py` | ffmpeg/ffprobe wrapper: demux, extraction, duration and fps probes |
| `common/prompts/` | Centralised LLM system prompts — `PROMPTS["key"]` dict |

**Stage packages (adapters)**

| Package | Adapter(s) | What it does |
|---|---|---|
| `media/align/` | **mms_fa** | CTC forced alignment, char-level |
| `asd/` | **loconet** | Joint multi-speaker active speaker detection |
| `asr/` | **bodhan** | Hosted Indic-Transcribe |
| `diarize/` | **pyannote** | pyannote community-1 speaker diarization |
| `emotion/` | **emotion2vec** | Per-segment delivery style from source emotion |
| `faces/` | **retinaface**, **mediapipe** + **bytetrack** | Face detection and tracking |
| `lipsync/` | **wav2lip**, **musetalk**, **latentsync15** | Lip-sync (each shells into its own conda env) |
| `translate/` | **bodhan** + `translit/` | Hosted Indic-Translate + transliteration |
| `restore/` | **gfpgan** | GFPGAN face restoration on lip-synced renders |
| `tts/` | **bodhan** | Hosted Indic-Speak synthesis |

**Media processing**

| Module | Purpose |
|---|---|
| `media/downloader.py` | yt-dlp download with metadata extraction |
| `media/stem_separation.py` | HTDemucs dialogue / M&E stem split |
| `media/segment.py` | Turn splitting at VAD gaps, merging, energy-valley capping |
| `media/duration.py` | Fitted duration model, candidate ranking, WSOLA stretch |
| `media/timing.py` | Timing-plan contract: `Piece`, `TimingPlan`, producers |
| `media/speed.py` | `--speed-mode segment` knobs (cap, floor) |
| `media/mix.py` | Overlay dub clips on M&E stem |
| `media/condense.py` | Stage 9c2: LLM shortening of over-long translated lines |
| `media/vad.py` | Speech / pause segmentation (pyannote, energy fallback) |

---

## 📚 Documentation

| Doc | Covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Host setup, interpreters, GPUs, ffmpeg builds, conventions |
| [docs/HOST.md](docs/HOST.md) | Env details, GPU busy policy, onnxruntime, Node, secrets |
| [docs/STAGES.md](docs/STAGES.md) | Stage internals, UI flow, speed modes, casting, tracing |
| [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md) | Known bugs and incident history |
| [docs/EDIT_AGENT.md](docs/EDIT_AGENT.md) | Edit agent ops, scopes, refusals |
| [docs/PROMPTS.md](docs/PROMPTS.md) | LLM prompts — now in `backend/common/prompts/` |
| [docs/UI_RUN.md](docs/UI_RUN.md) | What a UI submission sets per stage |
| [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md) | Job folder layout |

---

## ⚠️ Known issues

See [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md) for the full list. The ones that bite most:

- `extra_margin` isn't plumbed through the MuseTalk adapter (only `bbox_shift`)
- Bodhan outages return `internal_error` 5xx with no automatic fallback
- ASD scores from before 2026-10-11 are worthless (float vs int16 MFCC bug)
- LatentSync 1.6 removed — ~18 GB against 16 GiB cards
