#!/usr/bin/env bash
# Dubbing Studio — one-shot setup.
#
#   ./setup.sh                 # venv + deps + fetch models + dry run on 60s clip
#   ./setup.sh --check-only    # report only, change nothing
#   ./setup.sh --no-dry-run    # set up but skip the 60s pipeline run
#   ./setup.sh --help          # pass-through to scripts/setup.py
#
# Env overrides:
#   VENV=.venv-foo  PY=python3.10  ./setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="${VENV:-.venv}"
PY="${PY:-python3}"

say() { printf '\033[36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

say "repo: $REPO_ROOT"
say "venv: $VENV   (python: $PY)"

# 1. system deps -----------------------------------------------------------
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not on PATH — apt install ffmpeg / brew install ffmpeg"
command -v git    >/dev/null 2>&1 || die "git not on PATH"
"$PY" --version >/dev/null 2>&1    || die "$PY not found"

# 2. venv ------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  say "creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

# 3. torch (CUDA if nvidia-smi is present, else default wheel) -------------
if ! python -c "import torch" 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    say "installing torch (CUDA 12.1 build)"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    say "installing torch (CPU build)"
    pip install torch torchaudio
  fi
fi

# 4. project requirements --------------------------------------------------
say "installing requirements.txt"
pip install -r requirements.txt

# 5. doctor + model fetch + dry run ---------------------------------------
exec python scripts/setup.py "$@"
