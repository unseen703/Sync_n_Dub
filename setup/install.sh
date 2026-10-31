#!/usr/bin/env bash
# One-command install: the four conda envs (envs/*.yml), the vendored
# lip-sync/ASD repos and their weights (setup/assets.lock.json via
# setup/fetch_assets.py), the UI build, then setup/doctor.py to report
# what's left.
#
#   ./install.sh                  # everything
#   ./install.sh --skip-envs      # assume the conda envs already exist
#   ./install.sh --skip-assets    # assume the vendor repos/weights are
#                                        # already in place (e.g. rsynced in)
#   ./install.sh --skip-ui        # skip the npm build
#   ./install.sh --include-optional   # also fetch optional weights
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
#   - start the UI server (./start_frontend.sh does that, once this finishes)

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
  # Patch the dead `torchvision.transforms.functional_tensor` import that
  # basicsr / facexlib / gfpgan / realesrgan still carry (moved to
  # `torchvision.transforms.functional` in torchvision 0.17). Pinning
  # torchvision < 0.17 is not an option on Blackwell sm_120 cards, so the
  # import is rewritten in place. Idempotent; re-run after upgrading any of
  # the four packages. Face-restore WEIGHTS are placed by fetch_assets.py.
  "$PY_DUBBING" - <<'PY' 2>&1 | sed 's/^/  /' \
    || warn "torchvision import shim reported a problem -- see above"
import sysconfig
from pathlib import Path
sp = Path(sysconfig.get_paths()["purelib"])
old = "torchvision.transforms.functional_tensor"
new = "torchvision.transforms.functional"
fixed = 0
for pkg in ("basicsr", "facexlib", "gfpgan", "realesrgan"):
    root = sp / pkg
    if not root.is_dir():
        continue
    for py in root.rglob("*.py"):
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if old not in text:
            continue
        py.write_text(text.replace(old, new))
        print(f"shimmed {py.relative_to(sp)}")
        fixed += 1
print(f"{fixed} file(s) changed")
PY

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

  if env_exists dubui; then
    say "dubui: env already exists, skipping creation"
  else
    conda env create -f envs/dubfrontend.yml -n dubui
  fi
else
  say "skipping conda env creation (--skip-envs)"
fi

# ---------------------------------------------------------------------------
# 2. vendor repos + weights
# ---------------------------------------------------------------------------
if [ "$SKIP_ASSETS" -eq 0 ]; then
  say "vendor repos + weights (setup/assets.lock.json)"
  OPT_FLAG=""
  [ "$INCLUDE_OPTIONAL" -eq 1 ] && OPT_FLAG="--include-optional"
  "$PY_DUBBING" setup/fetch_assets.py --all $OPT_FLAG
else
  say "skipping asset fetch (--skip-assets)"
fi

# ---------------------------------------------------------------------------
# 3. UI build
# ---------------------------------------------------------------------------
if [ "$SKIP_UI" -eq 0 ]; then
  say "UI build (frontend/ -> web-dist)"
  conda run -n dubui npm --prefix frontend/ ci
  conda run -n dubui npm --prefix frontend/ run build
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
"$PY_DUBBING" setup/doctor.py || true

say "done. Fix anything FAIL/WARN above, then: setup/start_ui.sh"
