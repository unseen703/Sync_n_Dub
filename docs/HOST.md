# Host internals

Moved out of CLAUDE.md on 2026-10-24. CLAUDE.md keeps the one-page summary; this
file holds the long form.

---

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

`ui/web/` (the operator UI) is a Vite app. Its toolchain lives in a separate
conda env so it never reaches the Python server's PATH:

```
conda create -n dubui nodejs=20     # 20.20.2 from pkgs/main
conda run -n dubui npm --prefix ui/web install
conda run -n dubui npm --prefix ui/web run build     # -> ui/web-dist/
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

**A card is "free" when it looks IDLE, not when it is empty.** Changed
2026-10-22. `backend/gpu.py::free_devices` used to refuse any card carrying a
foreign compute process at all, which refused a run for a neighbour holding
1117 MiB of **resident but idle** CUDA context at 0% utilisation. It is now a
load test: utilisation under `GPU_BUSY_UTIL_PCT` (10%) **and** foreign memory
under `GPU_BUSY_MEM_PCT` (10% — ~1631 MiB of a 16311 MiB card), overridable
with `DUBBING_GPU_BUSY_{UTIL,MEM}_PCT`. `DUBBING_GPU_ALLOW_SHARED=1` remains
the harder override that skips the test entirely.

Three things make it work, and each is the reason an earlier attempt failed:

1. **It measures FOREIGN memory, not `memory.used`.** A plain VRAM floor was
   tried and removed before, because this process holds 1.9–4.7 GiB of its
   own context from face detection and ASD — and
   `scripts/lipsync_asd_gated.py` scores TalkNet on the GPU in the same
   process *before* asking whether a card is free. A threshold over
   `memory.used` reads our own context as a stranger and refuses every run
   that did any GPU work first. `gpu_load()` sums `--query-compute-apps`
   rows, excluding `_own_pids()`.
2. **`utilization.gpu` is an instantaneous sample**, so a job between batches
   reads 0% and one read would call a busy card free. It is sampled
   `GPU_UTIL_SAMPLES` (3) times 200 ms apart and the **max** is taken —
   averaging would let 0/0/90 pass as 30.
3. **Unknown is busy.** A device missing from `gpu_load()` is skipped, not
   admitted: failing open puts an unpinned worker on whichever card the
   driver picks, which is the collision the whole gate exists to prevent.

This is a real loosening and the risk is the operator's: a neighbour that
grows takes one of the two jobs with it, and lip-sync is the likelier victim
because MuseTalk answers an OOM by exiting 0 having written nothing.

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

