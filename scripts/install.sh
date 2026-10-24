#!/usr/bin/env bash
# One-command install: the four conda envs (envs/*.yml), the vendored
# lip-sync/ASD repos and their weights (models/assets.lock.json via
# scripts/fetch_assets.py), the UI build, then scripts/doctor.py to report
# what's left.
#
#   scripts/install.sh                  # everything
#   scripts/install.sh --skip-envs      # assume the conda envs already exist
#   scripts/install.sh --skip-assets    # assume the vendor repos/weights are
#                                        # already in place (e.g. rsynced in)
#   scripts/install.sh --skip-ui        # skip the npm build
#   scripts/install.sh --include-optional   # also fetch optional weights
#                                            # (musetalk v1, syncnet -- +~6GB)
#
# Idempotent: each step checks whether its target already exists before
# doing anything, so re-running after a partial or failed install only
# does the remaining work. This is the single entry point CLAUDE.md asks
# `setup.sh` to be; `setup.sh` now just execs this.
#
# What it does NOT do, on purpose:
#   - write or touch .env (copy .env.example yourself and fill in keys --
#     see .env.example's own comments for what each one needs)
#   - accept the pyannote gate on your behalf (that's a one-time click on
#     the model card at huggingface.co, tied to your HF_TOKEN's account)
#   - start the UI server (scripts/start_ui.sh does that, once this finishes)

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

SKIP_ENVS=0
SKIP_ASSETS=0
SKIP_UI=0
INCLUDE_OPTIONAL=0
for arg in "$@"; do
  case "$arg" in
    --skip-envs) SKIP_ENVS=1 ;;
    --skip-assets) SKIP_ASSETS=1 ;;
    --skip-ui) SKIP_UI=1 ;;
    --include-optional) INCLUDE_OPTIONAL=1 ;;
    --help|-h)
      sed -n '2,25p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg (see --help)" ;;
  esac
done

command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
CONDA_BASE="$(conda info --base)"
PY_DUBBING="$CONDA_BASE/envs/dubbing/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# ---------------------------------------------------------------------------
# 1. conda envs
# ---------------------------------------------------------------------------
if [ "$SKIP_ENVS" -eq 0 ]; then
  say "conda envs (envs/*.yml)"

  if env_exists dubbing; then
    say "dubbing: env already exists, skipping creation"
  else
    conda env create -f envs/dubbing.yml -n dubbing
  fi
  "$CONDA_BASE/envs/dubbing/bin/python" -c "import torch" 2>/dev/null \
    || "$CONDA_BASE/envs/dubbing/bin/python" -m pip install -r envs/dubbing.pip.txt \
         --extra-index-url "$TORCH_INDEX"

  if env_exists musetalk; then
    say "musetalk: env already exists, skipping creation"
  else
    conda env create -f envs/musetalk.yml -n musetalk
  fi
  "$CONDA_BASE/envs/musetalk/bin/python" -c "import torch" 2>/dev/null \
    || "$CONDA_BASE/envs/musetalk/bin/python" -m pip install --no-deps \
         -r envs/musetalk.pip.txt --extra-index-url "$TORCH_INDEX"
  "$CONDA_BASE/envs/musetalk/bin/python" -c "import mmcv" 2>/dev/null \
    || { say "musetalk: building mmcv (this is the ~20+ minute step)"; \
         envs/build_mmcv_musetalk.sh "$CONDA_BASE/envs/musetalk/bin/python"; }

  if env_exists latentsync; then
    say "latentsync: env already exists, skipping creation"
  else
    conda env create -f envs/latentsync.yml -n latentsync
  fi
  "$CONDA_BASE/envs/latentsync/bin/python" -c "import torch" 2>/dev/null \
    || "$CONDA_BASE/envs/latentsync/bin/python" -m pip install \
         -r envs/latentsync.pip.txt --extra-index-url "$TORCH_INDEX"

  if env_exists restore; then
    say "restore: env already exists, skipping creation"
  elif env_exists wav2liphd; then
    say "restore: found as 'wav2liphd' (the name this host used), leaving it"
  else
    say "restore: cloning dubbing (see envs/restore.yml for why -- a smaller"
    say "  install that adds the 4 packages straight into dubbing instead"
    say "  has not been tried on a live install, so this is the safe default)"
    conda create -y -n restore --clone dubbing
  fi
  RESTORE_ENV=restore; env_exists restore || RESTORE_ENV=wav2liphd
  "$CONDA_BASE/envs/$RESTORE_ENV/bin/python" -c "import basicsr" 2>/dev/null \
    || "$CONDA_BASE/envs/$RESTORE_ENV/bin/python" -m pip install -r envs/restore.pip.txt
  "$PY_DUBBING" scripts/setup_face_restore.py 2>&1 | sed 's/^/  /' || warn "setup_face_restore.py reported a problem -- see above"

  if env_exists dubui; then
    say "dubui: env already exists, skipping creation"
  else
    conda env create -f envs/dubui.yml -n dubui
  fi
else
  say "skipping conda env creation (--skip-envs)"
fi

# ---------------------------------------------------------------------------
# 2. vendor repos + weights
# ---------------------------------------------------------------------------
if [ "$SKIP_ASSETS" -eq 0 ]; then
  say "vendor repos + weights (models/assets.lock.json)"
  OPT_FLAG=""
  [ "$INCLUDE_OPTIONAL" -eq 1 ] && OPT_FLAG="--include-optional"
  "$PY_DUBBING" scripts/fetch_assets.py --all $OPT_FLAG
else
  say "skipping asset fetch (--skip-assets)"
fi

# ---------------------------------------------------------------------------
# 3. UI build
# ---------------------------------------------------------------------------
if [ "$SKIP_UI" -eq 0 ]; then
  say "UI build (ui/web2 -> ui/web-dist)"
  conda run -n dubui npm --prefix ui/web2 ci
  conda run -n dubui npm --prefix ui/web2 run build
else
  say "skipping UI build (--skip-ui)"
fi

# ---------------------------------------------------------------------------
# 4. doctor
# ---------------------------------------------------------------------------
say "doctor"
if [ ! -f .env ]; then
  warn ".env does not exist yet -- copy .env.example to .env and fill in keys, then re-run the doctor:"
  warn "  cp .env.example .env && chmod 600 .env"
fi
"$PY_DUBBING" scripts/doctor.py || true

say "done. Fix anything FAIL/WARN above, then: scripts/start_ui.sh"
