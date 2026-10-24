#!/usr/bin/env bash
# Start the operator UI, portably -- resolves its own interpreter and
# libstdc++ preload instead of needing either done by hand.
#
#   scripts/start_ui.sh                    # 127.0.0.1:7861
#   scripts/start_ui.sh --host 0.0.0.0     # needs DUBBING_UI_USER/PASS in .env
#   scripts/start_ui.sh --port 7900
#
# Replaces "started by hand" (README/CLAUDE.md's old `conda activate dubbing;
# python ui/server.py`), which is how this host once ran the server from
# `.venv-diarization` and spent a day on "insightface not found" before
# realising it was the wrong interpreter (CLAUDE.md). PY resolution here
# mirrors scripts/ui_run_job.sh's: DUBBING_PY wins if it points at something
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
export DUBBING_PY="$PY"   # so the UI's own job launcher (ui.jobqueue) inherits it too

ENV_LIB="$(dirname "$PY")/../lib"
LIBSTDCXX="$ENV_LIB/libstdc++.so.6"
if [ -f "$LIBSTDCXX" ]; then
  export LD_PRELOAD="$LIBSTDCXX"
fi
export PYTHONUNBUFFERED=1

cd "$REPO" || exit 1

WEB_DIST="$REPO/ui/web-dist/index.html"
if [ ! -f "$WEB_DIST" ]; then
  echo "warning: $WEB_DIST missing -- serving the read-only legacy inspector" >&2
  echo "  build it: conda run -n dubui npm --prefix ui/web2 ci && conda run -n dubui npm --prefix ui/web2 run build" >&2
fi

if [ -f "$REPO/scripts/doctor.py" ]; then
  echo "--- quick doctor check (scripts/doctor.py --quick) ---"
  "$PY" "$REPO/scripts/doctor.py" --quick || true
  echo "--- starting the server regardless of the above; fix FAILs for full functionality ---"
fi

exec "$PY" -m ui.server "$@"
