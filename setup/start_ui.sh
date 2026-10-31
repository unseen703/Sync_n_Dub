#!/usr/bin/env bash
# Start the operator UI, portably -- resolves its own interpreter and
# libstdc++ preload instead of needing either done by hand.
#
#   ./start_frontend.sh                    # 127.0.0.1:7861
#   ./start_frontend.sh --port 7900
#
# There is NO auth (removed 2026-10-25) -- `--host 0.0.0.0` exposes every
# route, including the ones that start GPU jobs. Use the SSH tunnel instead.
#
# Replaces "started by hand" (README/CLAUDE.md's old `conda activate dubbing;
# python server/server.py`), which is how this host once ran the server from
# `.venv-diarization` and spent a day on "insightface not found" before
# realising it was the wrong interpreter (CLAUDE.md). PY resolution here
# mirrors ./ui_run_job.sh's: DUBBING_PY wins if it points at something
# real, then the sibling 'dubbing' conda env, then refuse.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_py() {
  if [ -n "${DUBBING_PY:-}" ] && [ -x "${DUBBING_PY}" ]; then
    printf '%s\n' "$DUBBING_PY"; return 0
  fi
  local base=""
  if [ -n "${CONDA_PREFIX:-}" ]; then
    base="${CONDA_PREFIX%/envs/*}"
  elif command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null)"
  fi
  if [ -n "$base" ] && [ -x "$base/envs/dubbing/bin/python" ]; then
    printf '%s\n' "$base/envs/dubbing/bin/python"; return 0
  fi
  return 1
}

if ! PY="$(resolve_py)"; then
  echo "no usable python interpreter found: set DUBBING_PY to the dubbing" \
       "env's python (e.g. <conda prefix>/envs/dubbing/bin/python)" >&2
  exit 1
fi
export DUBBING_PY="$PY"   # so the UI's own job launcher (server.jobqueue) inherits it too

ENV_LIB="$(dirname "$PY")/../lib"
LIBSTDCXX="$ENV_LIB/libstdc++.so.6"
if [ -f "$LIBSTDCXX" ]; then
  export LD_PRELOAD="$LIBSTDCXX"
fi
export PYTHONUNBUFFERED=1

cd "$REPO" || exit 1

WEB_DIST="$REPO/web-dist/index.html"
if [ ! -f "$WEB_DIST" ]; then
  echo "warning: $WEB_DIST missing -- server will be API-only" >&2
  echo "  build it: conda run -n dubui npm --prefix frontend/ ci && conda run -n dubui npm --prefix frontend/ run build" >&2
fi

if [ -f "$REPO/setup/doctor.py" ]; then
  echo "--- quick doctor check (setup/doctor.py --quick) ---"
  "$PY" "$REPO/setup/doctor.py" --quick || true
  echo "--- starting the server regardless of the above; fix FAILs for full functionality ---"
fi

exec "$PY" -m server "$@"
