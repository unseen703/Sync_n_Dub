# CLAUDE.md — Dubbing Studio

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Never read .env files directly, remind user if ever prompted to do so.
I'd like you to communicate in a natural, conversational, human way. Use contractions,
vary your enthusiasm, and skip the formal corporate speak.
Avoid AI slop, invented jargon and filler words; keep it short, simple and precise wherever possible.

First suggest what changes need to be made, or what the implications of a change will be.
The author wants to review everything before it's edited.

Before making an edit, ask the user which repository to look in and which files to change. If a reference goes out of scope, prompt the user before reading those files. Don't read logs or unrelated modules unless explicitly asked to.

Don't run anything; ask for permission before running, and provide the sh command I will run.
For every run list a link to its output , use tabular foramt if there are multiple runs. Output should be reachable by user, not in the temp folder.
---

Indic video dubbing pipeline: take a video (YouTube or local), transcribe it, translate it, synthesize a
dubbed track in an Indian language, mix it, and lip-sync the speaker's mouth to the new audio.

Repo: `git@github.com:unseen703/Sync_n_Dub.git` · branch `main`.

**Structure lives in [README.md](README.md)** (the Structure section). This file holds the rules most tasks
need. Long form:

- [docs/HOST.md](docs/HOST.md) — envs, GPU busy policy, onnxruntime preload, ffmpeg builds, Node, secrets
- [docs/STAGES.md](docs/STAGES.md) — stage internals, UI, speed modes, casting, tracing
- [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md) — known bugs, incident history
- [docs/EDIT_AGENT.md](docs/EDIT_AGENT.md) — edit agent ops, scopes, refusals
- [docs/PROMPTS.md](docs/PROMPTS.md) — every LLM prompt verbatim
- [docs/UI_RUN.md](docs/UI_RUN.md) — what a UI submission sets per stage
- [data/DATA_LAYOUT.md](data/DATA_LAYOUT.md) — job folder layout

---

## 1. The host

**Python — there's no `python` on PATH.**

```
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
```

Aux envs `musetalk`, `latentsync`, `lipgan` exist for upstream pin conflicts; adapters shell into them via
`<MODEL>_PYTHON` (all in `.env.example`). Node (`dubui` env) is for the UI build only; bare `npm` is broken.

**libstdc++ preload.** Anything touching faces/ASD needs this, or `import insightface` fails on
`CXXABI_1.3.15`:

```
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
```

**GPUs.** 2 × NVIDIA RTX PRO 2000 Blackwell, 16311 MiB each (sm_120). 16 GiB rules out LatentSync at 512
and leaves MuseTalk ~100 MiB at `batch_size=20`. Pin with `CUDA_VISIBLE_DEVICES` or `DUBBING_GPUS=0,1`.
A card is "free" when idle (`backend/gpu.py::free_devices`: util under 10% AND foreign memory under 10%);
`DUBBING_GPU_ALLOW_SHARED=1` overrides. onnxruntime silently falls back to CPU unless
`backend.gpu.ensure_onnxruntime_cuda()` runs first.

**ffmpeg is filter-specific.** `envs/dubbing/bin/ffmpeg` (7.1) has `atempo`, not `rubberband`; bundled
`imageio_ffmpeg` (7.0.2) has both. `media_io._ffmpeg_bin()` / `_ffprobe_bin()` look in the running
interpreter's dir, and `BaseLipSync._subprocess_env()` puts it on PATH, because vendored upstreams call bare
`ffmpeg` and exit 0 on failure.

**Secrets.** `.env` (gitignored): `HF_TOKEN`, `BODHAN_{ASR,MT,TTS,TRANSLIT}_{URL,KEY,MODEL}`,
`OPENROUTER_{API_KEY,URL,MODEL}`, Langfuse keys, checkpoint paths. See [.env.example](.env.example).

---

## 2. Conventions

- **One run, one job folder.** Everything goes under `data/jobs/<job_id>/` with a `manifest.json`.
  Nothing is written outside it.
- **Merged JSON is versioned** at `mt/versions/<stem>.v<NNN>.<producer>.json`; the live file is rewritten
  in place by 9b/9b2/9c/9c2/9e and every edit. **Trap:** `edit_agent._find_merged_json` takes the newest
  `mt/*.mt.json` by mtime, so a snapshot placed in `mt/` or named `.mt.json` silently becomes the document.
  `tests/test_mt_versions.py` guards it.
- **Downloads keep uploader metadata** (`<stem>.meta.json`, 18 fields via `downloader.video_metadata()`).
  `categories` drives the register classifier (9b1). `description`, `title`, `tags` are untrusted: pass
  them as delimited data with enum answers.
- **Adding a model = one adapter + one registry line** in `models/<stage>/__init__.py`. Worked example:
  `models/lipsync/`.
- **No clip-window flag.** Cut with ffmpeg first (`clip-run` skill). Exception:
  `run_lipsync(..., window=(start, end))`, overshoot one frame.
- **Speed mode `segment` is the default** (2026-10-22): per-cue fit both ways, `--speed-cap` 1.8,
  `--speed-floor` 0.75.
- **Stage 9c2** rewrites a line when `est > target * --condense-speed-cap` (1.5). On in the UI; the CLI
  needs `--condense-overlong`.
- **Edit replays read the run's own `request.json`**, not argparse defaults. A new `SpeedSetup` field also
  goes in `plan_config`; `tests/test_ui_backend_sync.py` greps both.
- **Bodhan is the only TTS engine.** `--tts bodhan` implies `--voice-engine indic-speak`.

### Canonical implementations — find these, don't rewrite them

A new caller imports the canonical one. A new adapter inherits. A case that doesn't fit widens it.

| Thing | Lives in |
|---|---|
| adapter registry: lazy load, lister, kwarg filter, `get_*` | `models/_registry.py::make_registry` |
| Bodhan auth, env resolution, pacing floor, `Retry-After` backoff | `models/_bodhan.py::BodhanClientMixin` |
| ffprobe → `(fps, duration)` | `backend/media_io.py::probe_video` |
| media duration (ffprobe → ffmpeg banner) | `backend/media_io.py::probe_duration` |
| `--speed-mode segment` knobs | `backend/speed.py` |
| best-effort JSON read of a job file | `ui/_io.py::read_json` |
| lip-sync model ids the edit agent advertises | `models.lipsync.available_models()`, via `edits._lipsync_model_ids` |
| ASD face crops, int16-scale MFCC, VRAM windowing | `models/asd/_mfcc_av.py` |
| where a vendored upstream lives, either side of the move | `models/_vendor.py::resolve_vendor_repo` |

Don't break these: `_REGISTRY_SPEC` stays a module-level dict literal (`repo_checkup.py` parses it), registry
error strings are pinned by `scripts/ab_refactor.py`, Bodhan retry loops are deliberately not shared, and
`backend/speed.py` stays dependency-free. Why: [docs/STAGES.md](docs/STAGES.md).

---

## 3. Running things

**UI.** `ui/server.py` (FastAPI, loopback only, port 7861) serves `ui/web-dist/`. Jobs launch detached via
`scripts/ui_run_job.sh`, the one launcher that sets the libstdc++ preload. The progress bar is 8 pills:
7 stage groups plus `Finished`.

**Edit agent.**

```
$PY -m backend.edit_agent -j <job_id> "speed up 0:30 to 0:45 by 1.2x"
$PY -m backend.edit_agent -j <job_id> --list-ops
```

Dry run by default; `--apply` writes, `--run` re-renders, `--revert` undoes. Gemma 3 12B via OpenRouter,
JSON mode off (the model emits `{".ops":[...]}` under it).

**Langfuse tracing** wraps `backend/llm.py`. Needs both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`;
`--check-langfuse` verifies them. `DUBBING_LANGFUSE_CAPTURE_CONTENT=0` drops prompt text and keeps
timings/tokens/cost. Pin `protobuf==6.32.0`.

**Skills** in `.claude/skills/`:

- `clip-run` — a stage or lip-sync model on a time window: spec pinning, VRAM/ETA, artifact paths
- `long-run` — long GPU jobs detached in tmux `dubsync`, status from disk, resume
- `youtube-media-fetch` — yt-dlp video, captions, dub tracks, 429 and part-file traps

---

## 4. Known rough edges

Full list in [docs/ROUGH_EDGES.md](docs/ROUGH_EDGES.md). The ones that bite most:

- `extra_margin` isn't plumbed through the MuseTalk adapter (only `bbox_shift`).
- `data/jobs/` was emptied by the 2026-10-11 archive.
- Lip-sync adapters don't pick the speaking face themselves; `run_lipsync(..., coord_list=…)` plus stage 6's
  scores npz does, wired at both call sites.
- Bodhan outages return `internal_error` 5xx, with no automatic fallback.
- ASD scores from before 2026-10-11 are worthless (float vs int16 MFCC bug).

---

## 5. Keeping the docs honest
README, this file, `plan.md` and `docs/*`.  When a file is added, moved or removed, update the README's Structure section.

Update this file when an env, interpreter, GPU, driver, CUDA or ffmpeg build changes; when an env var is
added to `.env.example`; when a convention or skill changes; or when you hit behaviour worth recording.
Check a line against the live system before writing it, delete a line that stops being true, and put long
detail in `docs/`.
