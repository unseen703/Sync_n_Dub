#!/usr/bin/env bash
# Dubbing Studio — install entry point.
#
#   ./setup.sh                 # conda envs, vendor repos+weights, UI build, doctor
#   ./setup.sh --check-only    # just run the doctor; install/fetch nothing
#   ./setup.sh --help          # pass-through to scripts/install.sh
#
# This used to build a single `.venv` and install torch from the cu121
# index -- wrong on two counts for this project: the pipeline resolves
# FOUR separate conda envs (CLAUDE.md), and this host's GPUs are Blackwell
# (sm_120), which cu121 has no kernels for. The real logic now lives in
# scripts/install.sh, built from envs/*.yml (pinned to what each env here
# actually runs) and models/assets.lock.json (the vendored repos and
# weights pip/conda can't provide). This file stays as the name every doc
# and habit already points at.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "--check-only" ]; then
  shift
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  PY="${CONDA_BASE:+$CONDA_BASE/envs/dubbing/bin/python}"
  if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "dubbing env not found yet -- run ./setup.sh (without --check-only) first" >&2
    exit 1
  fi
  exec "$PY" scripts/doctor.py "$@"
fi

exec scripts/install.sh "$@"
