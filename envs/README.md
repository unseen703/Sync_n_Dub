# Environment specs

One spec per conda env this project actually uses at runtime, captured from
the live envs on the host this project was built on — **not** redesigned or
guessed at. Each `<name>.yml` is the conda-installed layer (taken verbatim
from that env's own `conda-meta/history` where one exists); each
`<name>.pip.txt` is a `pip list --format=freeze` of everything pip put
there. Together they're a from-scratch recipe for that env, line for line.

| Env | Spec | Why it's separate |
|---|---|---|
| `dubbing` | `dubbing.yml` + `dubbing.pip.txt` | The main pipeline + UI env — everything in `backend/`, `models/` (except the three below) and `ui/` runs here. |
| `musetalk` | `musetalk.yml` + `musetalk.pip.txt` + `build_mmcv_musetalk.sh` | diffusers/mmcv/mmdet/mmpose pins conflict with `dubbing`. |
| `latentsync` | `latentsync.yml` + `latentsync.pip.txt` | diffusers/transformers/insightface pins conflict with `dubbing`. |
| `restore` (on disk: `wav2liphd`) | `restore.yml` + `restore.pip.txt` | `basicsr`/`gfpgan` need a dead-import patch `dubbing` doesn't carry — see `restore.yml` for whether this needs its own env at all. |
| `dubui` | `dubui.yml` | Node, for building `ui/web2/` — never a runtime dependency of the server. |

**Deliberately excluded**: `lipgan` (vestigial — its adapter was removed
2026-10-23, see `tests/test_lipsync_env_python.py`) and
`.venv-diarization` (CLAUDE.md records it as the "wrong interpreter" trap,
not a real deploy target). `chemformer`, `chemformer-gpu` and `vllm` on this
host are unrelated projects.

## What these fix

Before this, the only spec in the repo was `requirements.txt` (loose
`>=` floors) and `setup.sh` (one `.venv`, torch from the **cu121** index).
Neither describes `envs/dubbing` as it actually runs, and cu121 has no
kernels for this project's own target hardware — Blackwell, sm_120 /
compute capability 12.0 (`nvidia-smi --query-gpu=compute_cap` on this host:
`12.0`). Every env here instead runs **torch 2.8.0+cu128** (driver
580.178.04), which is what every `.yml`'s install instructions pull from.

## Recreating an env

Each `.yml`'s header has the exact commands. The short version, for any of
the four Python envs:

```bash
conda env create -f envs/<name>.yml -n <name>
<conda prefix>/envs/<name>/bin/python -m pip install \
    -r envs/<name>.pip.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

`musetalk` additionally needs `envs/build_mmcv_musetalk.sh` (mmcv has no
prebuilt wheel for this torch/CUDA/GPU-arch combination, so it's compiled
locally — budget ~20+ minutes for that one step). `restore` is a clone of
`dubbing` plus four packages, not a fresh env (see `restore.yml`).

These specs are not yet wired into `setup.sh` — that's the next stage. For
now they're the accurate, reviewable record of what each env needs; a doctor
script comes later and will check each one is actually importable, the same
way `scripts/setup_face_restore.py` already smoke-tests the restore env.

## Keeping these honest

There's no automation yet that re-captures these when an env changes (adding
a package, bumping torch). Until there is: after installing something new
into any of these envs, regenerate its freeze —

```bash
<env>/bin/python -m pip list --format=freeze > envs/<name>.pip.txt
```

— and if the change was conda-level (not pip), add the command to that
env's `.yml` header the same way the existing history lines are recorded,
so the next deploy doesn't silently diverge from what's actually running.
