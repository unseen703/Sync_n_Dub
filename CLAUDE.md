# CLAUDE.md — Dubbing Studio


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

I'd like you to communicate in a natural, conversational, human way. Use contractions,
vary your enthusiasm, and skip the formal corporate speak.
Avoid using AI slope, invented jargons, and filler words, keep it short, simple and precise where every its possible.

First provide suggestions of  what are the changes that needs to be done, or what will be the implications of changes.
author want to review everything before its edited.

Dont run anything ask for the permission before running.
and provide the sh command that I will run.

Indic video dubbing pipeline: take a video (YouTube or local), transcribe it,
translate it, synthesize a dubbed track in an Indian language, mix it, and lip-sync the speaker's mouth to the new audio.

Repo: `git@github.com:unseen703/Sync_n_Dub.git` · branch `main`.

**Structure lives in [README.md](README.md), not here.** Its generated block —
layout, `backend/` modules, adapter registries, pipeline stages, scripts, tests,
skills, host summary — is rewritten by `scripts/repo_checkup.py` every 5 hours,
so it is current by construction.

This file holds the rules most tasks need. Deeper context lives in:

- **[docs/HOST.md](docs/HOST.md)** — host internals: GPU busy policy, onnxruntime CUDA preload, ffmpeg detection, Node/dubui env, libstdc++ preload rationale, secrets detail.
- **[docs/STAGES.md](docs/STAGES.md)** — per-stage detail: UI runner, talking-head page, draft flow, edit agent, condense, speed modes, gender vote, voice assignment, speaker refinement, captions ASR, register classifier, Langfuse tracing.
- **[docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md)** — known bugs and incident history (what was §4).
- **[docs/EDIT_AGENT.md](docs/EDIT_AGENT.md)** — the edit agent's ops, scopes, refusals.
- **[docs/PROMPTS.md](docs/PROMPTS.md)** — every LLM prompt verbatim.
- **[docs/UI_RUN.md](docs/UI_RUN.md)** — what a UI submission sets per stage.
- **[data/DATA_LAYOUT.md](data/DATA_LAYOUT.md)** — job folder layout.

---

## 1. The host

### Interpreter — there is no `python` on PATH

```
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
```

Three auxiliary envs (`musetalk`, `latentsync`, `lipgan`) exist for upstream pin conflicts; adapters shell into them via `<MODEL>_PYTHON`. All four variables are in `.env.example`. See [docs/HOST.md](docs/HOST.md) for env resolution details.

Node (`dubui` env) is for the UI build only, bare `npm` on PATH is broken — detail in [docs/HOST.md](docs/HOST.md).

### LD_PRELOAD for cv2/insightface

Scripts touching faces/ASD must set:

```
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
```

Otherwise `import insightface` fails on `CXXABI_1.3.15` depending on import order. Full rationale in [docs/HOST.md](docs/HOST.md).

### GPUs

**2 × NVIDIA RTX PRO 2000 Blackwell, 16311 MiB each** (sm_120). The 16 GiB ceiling is real — rules out LatentSync at 512, leaves MuseTalk ~100 MiB headroom at `batch_size=20`. Measured per-model costs in `clip-run` skill.

Pin devices with `CUDA_VISIBLE_DEVICES` or `DUBBING_GPUS=0,1`.

A card is "free" when IDLE, not when empty — `backend/gpu.py::free_devices` tests utilisation under `GPU_BUSY_UTIL_PCT` (10%) AND foreign memory under `GPU_BUSY_MEM_PCT` (10%). Override with `DUBBING_GPU_ALLOW_SHARED=1`. Full policy and the three traps that gate it in [docs/HOST.md](docs/HOST.md).

onnxruntime needs its CUDA libraries preloaded (silent CPU fallback otherwise) — handled by `backend.gpu.ensure_onnxruntime_cuda()`. Detail in [docs/HOST.md](docs/HOST.md).

### ffmpeg is filter-specific

Two builds, different filters:

- `envs/dubbing/bin/ffmpeg` (v7.1) — has `atempo`, NOT `rubberband`
- bundled `imageio_ffmpeg` (v7.0.2) — has both

`media_io._ffmpeg_bin()` and `_ffprobe_bin()` look in the running interpreter's own directory, and `BaseLipSync._subprocess_env()` prepends it to subprocess PATH — vendored upstreams call bare `ffmpeg` and exit 0 on silent failure otherwise. Full story in [docs/HOST.md](docs/HOST.md).

### Secrets

`.env` (gitignored) holds `HF_TOKEN`, Bodhan credentials (`BODHAN_{ASR,MT,TTS,TRANSLIT}_{URL,KEY,MODEL}`), `OPENROUTER_{API_KEY,URL,MODEL}`, Langfuse keys, and checkpoint paths. Full key list and empty-value behaviour in [docs/HOST.md](docs/HOST.md) and [.env.example](.env.example).

---

## 2. Conventions

### Job folders

**One run, one job folder.** Everything a run produces goes under `data/jobs/<job_id>/` with a `manifest.json` inventory. Nothing is written outside it. Layout: [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md).

### Merged JSON is versioned

Every version of the merged ASR+MT JSON is kept at `mt/versions/<stem>.v<NNN>.<producer>.json`. The live file is rewritten in place by 9b/9b2/9c/9c2/9e and every applied edit, so `edit_backup`/`condense_backup` is a diff, not a document. **The naming is a trap**: `edit_agent._find_merged_json` picks the newest `mt/*.mt.json` by mtime, so a snapshot named `.mt.json` or placed in `mt/` would silently become the document. Hence the subdirectory and `.<producer>.json` suffix. `tests/test_mt_versions.py` guards it. Detail in [docs/STAGES.md](docs/STAGES.md).

### Downloads keep uploader metadata

`backend/downloader.py::video_metadata()` lifts 18 fields off yt-dlp's info dict and persists them as `<stem>.meta.json` beside the video. `categories` drives the register classifier (9b1). **Uploader-authored fields (`description`, `title`, `tags`) are untrusted input** — passed as delimited data with constrained enum answers. Detail in [docs/STAGES.md](docs/STAGES.md).

### Adding a model = one adapter + one registry line

In `models/<stage>/__init__.py`. The CLI picks up the new choice automatically. `models/lipsync/` is the worked example: four adapters behind one `base.py`, three shelling into their own conda env via `<MODEL>_PYTHON`.

### Canonical implementations — find these, don't rewrite them

A new caller imports the canonical one. A new adapter inherits. A new case that doesn't fit widens the canonical.

| Thing | Lives in |
|---|---|
| adapter registry: lazy load, lister, kwarg filter, `get_*` | `models/_registry.py::make_registry` |
| Bodhan auth, env resolution, pacing floor, `Retry-After` backoff | `models/_bodhan.py::BodhanClientMixin` |
| ffprobe → `(fps, duration)` | `backend/media_io.py::probe_video` |
| media duration (ffprobe → ffmpeg banner) | `backend/media_io.py::probe_duration` |
| `--speed-mode segment` knobs | `backend/speed.py` |
| best-effort JSON read of a job file | `ui/_io.py::read_json` |
| lip-sync model ids the edit agent advertises | `models.lipsync.available_models()`, via `edits._lipsync_model_ids` |

Four load-bearing properties of this table (dict literal at module level for `scripts/repo_checkup.py`, verbatim error strings pinned by `scripts/ab_refactor.py`, Bodhan retry loops deliberately NOT shared, `backend/speed.py` dependency-free): see [docs/STAGES.md](docs/STAGES.md).

### Pipeline rules

- **No clip-window flag.** Running a stage on a time range means cutting video/audio with ffmpeg first (see `clip-run` skill). Exception: `run_lipsync(..., window=(start, end))` renders only that slice and splices it back; measured overshoot one frame.
- **Speed mode: `segment` is the default** (2026-10-22). Per-cue fit in both directions. `--speed-cap` 1.8, `--speed-floor` 0.75 (asymmetric and deliberate). Full rationale and the shift-forward cursor fixes in [docs/STAGES.md](docs/STAGES.md).
- **Stage 9c2 rewrites a line the stretch can't fit** (`--condense-overlong`, on in UI). Gate is `est > target * --condense-speed-cap` (1.5). Detail in [docs/STAGES.md](docs/STAGES.md).
- **An edit replay reads the pipeline's own `request.json`**, not argparse defaults. Fixed 2026-10-22. If you add a `SpeedSetup` field, add it to `plan_config` too — `tests/test_ui_backend_sync.py` greps both.

### UI

`ui/server.py` (FastAPI, loopback-only, port 7861) serves a built Vite app from `ui/web-dist/`. Jobs launched detached via `scripts/ui_run_job.sh` (the one launcher that exports the libstdc++ preload). The progress bar is **8 pills, always** — 7 named stage groups plus `Finished`. UI internals, progress scraping, talking-head page, draft flow, edit panel: [docs/STAGES.md](docs/STAGES.md).

**Bodhan is the only TTS engine.** `--tts bodhan` implies `--voice-engine indic-speak`. Edge is removed. Detail in [docs/STAGES.md](docs/STAGES.md).

### Editing a finished dub

```
$PY -m backend.edit_agent -j <job_id> "speed up 0:30 to 0:45 by 1.2x"
$PY -m backend.edit_agent -j <job_id> --list-ops
```

Dry-run by default; `--apply` writes, `--run` re-renders. Edits are reversible (`--revert`). Model: Gemma 3 12B via OpenRouter. JSON mode is OFF (model emits `{".ops":[...]}` under it). See [docs/EDIT_AGENT.md](docs/EDIT_AGENT.md).

### LLM tracing

Langfuse wraps `backend/llm.py`. Both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` or no tracing. `--check-langfuse` verifies the pair. `DUBBING_LANGFUSE_CAPTURE_CONTENT=0` drops prompt/completion text, keeps timings/tokens/cost. `protobuf==6.32.0` must be pinned. Detail in [docs/STAGES.md](docs/STAGES.md).

---

## 3. Project skills

Three in `.claude/skills/`:

- **`clip-run`** — running a stage or lip-sync model on a time window: spec pinning, VRAM/ETA budgets, artifact paths.
- **`long-run`** — launching long GPU jobs detached in tmux `dubsync`, status from disk, resume after session death.
- **`youtube-media-fetch`** — yt-dlp: video, captions, per-language dub tracks, dub provenance, 429 and unmerged-part-file traps.

---

## 4. Known rough edges

See [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md). Highlights to be aware of:

- `extra_margin` not plumbed through MuseTalk adapter (only `bbox_shift`).
- `data/jobs/` was emptied by the 2026-10-11 archive.
- 9c2 is on for every UI run; CLI still needs `--condense-overlong`.
- Lip-sync adapters do not choose the SPEAKING face — fix via `run_lipsync(..., coord_list=…)` + stage 6 scores npz, now wired at both call sites.
- Bodhan outages return `internal_error` 5xx — no automatic fallback.
- ASD scores produced before 2026-10-11 are worthless (float vs int16 MFCC bug).

---

## 5. Keeping the docs honest

Six files, six different owners.

| File | Owned by | Covers |
|---|---|---|
| **README.md** — generated block | `scripts/repo_checkup.py` | structure: layout, modules, registries, stages, scripts, skills |
| **README.md** — prose outside it | you | what the project is, how a run is organised, why |
| **CHANGES-PENDING.md** | script writes, you clear | changes awaiting a narrative |
| **CLAUDE.md** (this file) | you | host behaviour, conventions, traps — the one-page summary |
| **docs/HOST.md** | you | host internals (long form) |
| **docs/STAGES.md** | you | pipeline stage internals (long form) |
| **docs/ROUGH_EDGES.md** | you | known bugs, incident history |
| **docs/EDIT_AGENT.md** | you | edit agent ops, scopes, refusals |
| **docs/PROMPTS.md** | you | every LLM prompt verbatim |
| **docs/UI_RUN.md** | you | what a UI submission sets per stage |

### The scheduled checkup

A cron job runs every 5 hours:

```
17 */5 * * *  →  scripts/repo_checkup.py  →  logs/repo_checkup.log
```

Snapshots structure, diffs against `.repo-state.json`, rewrites README's generated block, appends to `CHANGES-PENDING.md` when something moves. Silent when nothing moved.

```bash
$PY scripts/repo_checkup.py --dry-run   # report drift, write nothing
$PY scripts/repo_checkup.py             # update README + pending buffer
$PY scripts/repo_checkup.py --clear     # empty the buffer once folded in
```

### When CHANGES-PENDING.md is non-empty

The generated block is already correct. What the buffer is asking for is the part a diff can't produce: *why* it changed, and what it means for someone using the repo. Fold what deserves it into README prose, HOST.md, STAGES.md, ROUGH_EDGES.md, or this file, then `--clear`. A rename or deleted scratch file can be cleared unexplained — the buffer is a prompt, not a backlog.

### When this file changes

- adding, rebuilding or moving a conda env or interpreter
- changing hardware, drivers, CUDA, or an ffmpeg build
- adding an env var or secret to `.env.example`
- changing a convention — job layout, where artifacts go, how models are registered
- adding or materially changing a skill in `.claude/skills/`
- hitting new behaviour worth recording

Edits go in the affected section. A line checked against the live system (registry, `nvidia-smi`, log) before being written down. A contradicted line is worse than a missing one, so a line that stops being true gets deleted.

Deeper detail (stage internals, incident history) goes in the pointed-to docs, not here.
