# Environment specs

One spec per conda env this project actually uses at runtime, captured from
the live envs on the host this project was built on — **not** redesigned or
guessed at. Each `<name>.yml` is the conda-installed layer (taken verbatim
from that env's own `conda-meta/history` where one exists); each
`<name>.pip.txt` is a `pip list --format=freeze` of everything pip put
there. Together they're a from-scratch recipe for that env, line for line.

| Env | Spec | Why it's separate |
|---|---|---|
| `dubbing` | `dubbing.yml` + `dubbing.pip.txt` | The main pipeline + UI env — everything in `backend/`, `models/` (except the two sidecars below) and `frontend/` runs here. Face restoration (basicsr / gfpgan / facexlib / realesrgan) was folded in on 2026-10-30. |
| `musetalk` | `musetalk.yml` + `musetalk.pip.txt` + `build_mmcv_musetalk.sh` | diffusers/mmcv/mmdet/mmpose pins conflict with `dubbing`. |
| `latentsync` | `latentsync.yml` + `latentsync.pip.txt` | diffusers/transformers/insightface pins conflict with `dubbing`. |
| `dubui` | `dubfrontend.yml` | Node, for building `frontend/web/` — never a runtime dependency of the server. |

**Deliberately excluded**: `lipgan` (vestigial — its adapter was removed
2026-10-23 and its last display row on 2026-10-28, see
`tests/test_lipsync_env_python.py`); `restore` / `wav2liphd` (merged into
`dubbing` on 2026-10-30 — the four packages are now in
`envs/dubbing.pip.txt`, basicsr's dead-import patch is applied inline by
`scripts/install.sh`, and face-restore weights are placed into site-packages
by `scripts/fetch_assets.py`); and `.venv-diarization` (CLAUDE.md records
it as the "wrong interpreter" trap, not a real deploy target). `chemformer`,
`chemformer-gpu` and `vllm` on this host are unrelated projects.

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
the three Python envs:

```bash
conda env create -f envs/<name>.yml -n <name>
<conda prefix>/envs/<name>/bin/python -m pip install \
    -r envs/<name>.pip.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

`musetalk` additionally needs `envs/build_mmcv_musetalk.sh` (mmcv has no
prebuilt wheel for this torch/CUDA/GPU-arch combination, so it's compiled
locally — budget ~20+ minutes for that one step). `scripts/install.sh`
already runs the dead-import patch for basicsr /
`torchvision.transforms.functional_tensor` inline after the `dubbing` env
is built, and `scripts/fetch_assets.py` places the facexlib / gfpgan /
realesrgan weights into site-packages whenever the `restore` feature is
fetched.

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
