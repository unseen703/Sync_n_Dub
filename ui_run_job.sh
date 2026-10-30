#!/usr/bin/env bash
# Launch one UI-submitted dubbing job, detached.
#
#   ./ui_run_job.sh <job_id> [--rerun]
#
# The job id is the ONLY argument. The URL, the language, the window and the
# resolved pipeline kwargs all live in data/jobs/<job_id>/request.json, so no
# user-supplied string ever reaches this shell, `ps`, or a tmux window title.
#
# Everything this script does that is not obvious is explained in CLAUDE.md:
#
#  * There is no `python` on PATH. The env interpreter is named outright
#    rather than activating conda, because a subprocess of an activated env
#    and a subprocess of a named interpreter differ in ways that have cost
#    runs here (CLAUDE.md section 1).
#  * LD_PRELOAD of the conda libstdc++ is not optional. cv2's bundled
#    libtbb.so.12 needs CXXABI_1.3.15, which the system libstdc++ does not
#    have; whichever library loads first decides which one the process gets,
#    so `import insightface` fails or succeeds depending on import order. The
#    face probe imports both. Without this the failure looks intermittent and
#    package-related, and is neither.
#  * Output goes to <job>/run.log, inside the job folder, because nothing is
#    written outside data/jobs/<job_id>/ (CLAUDE.md section 2) and the UI
#    already serves that path as the raw log tail.

set -u

# REPO is this script's own location, not a literal, so the same file works
# from any checkout. PY mirrors models/lipsync/base.py::resolve_env_python's
# order: an explicit DUBBING_PY wins -- `frontend.server.main()` sets it to its own
# `sys.executable` before spawning a job (CLAUDE.md), so a detached run
# normally gets the exact interpreter already proven to work -- then the
# sibling `dubbing` conda env next to whatever interpreter this shell can
# find, then refuse rather than guess. A silently wrong interpreter here is
# the cv2/insightface import-order trap in CLAUDE.md section 1, and it would
# otherwise only surface after ASR/MT/TTS/mix have already run.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_py() {
  if [ -n "${DUBBING_PY:-}" ] && [ -x "${DUBBING_PY}" ]; then
    printf '%s\n' "$DUBBING_PY"
    return 0
  fi
  local base=""
  if [ -n "${CONDA_PREFIX:-}" ]; then
    base="${CONDA_PREFIX%/envs/*}"
  elif command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null)"
  fi
  if [ -n "$base" ] && [ -x "$base/envs/dubbing/bin/python" ]; then
    printf '%s\n' "$base/envs/dubbing/bin/python"
    return 0
  fi
  return 1
}

if ! PY="$(resolve_py)"; then
  echo "no usable python interpreter found: set DUBBING_PY to the dubbing" \
       "env's python (e.g. <conda prefix>/envs/dubbing/bin/python)" >&2
  exit 1
fi
ENV_LIB="$(dirname "$PY")/../lib"

if [ $# -lt 1 ]; then
  echo "usage: $0 <job_id> [--rerun]" >&2
  exit 2
fi
JOB_ID=$1
shift

JOB_DIR="${DUBBING_UI_DATA_DIR:-${DUBBING_DATA_DIR:-$REPO/data}}/jobs/$JOB_ID"
if [ ! -d "$JOB_DIR" ]; then
  echo "no such job folder: $JOB_DIR" >&2
  exit 2
fi

mkdir -p "$JOB_DIR"
exec >> "$JOB_DIR/run.log" 2>&1

# LD_PRELOAD only if the conda libstdc++ is actually there -- a venv install
# has none at this derived path, and exporting a path to nothing makes the
# loader print "object ... cannot be preloaded" on every exec instead of
# silently doing without it.
LIBSTDCXX="$ENV_LIB/libstdc++.so.6"
if [ -f "$LIBSTDCXX" ]; then
  export LD_PRELOAD="$LIBSTDCXX"
fi
export PYTHONUNBUFFERED=1

# `--speed-mode warp` (the UI default since 2026-10-26) calls out to the
# `rubberband` CLI through pyrubberband. It is installed in its own conda
# env rather than in `dubbing` because pulling it into `dubbing` created a
# libstdc++ fight with cv2's bundled libtbb -- the preload above resolves
# that for `dubbing`, but the rubberband install would have moved it again.
# Prepending its bin keeps the two envs apart and gives pyrubberband a
# `rubberband` to shell out to; if the env is absent, the warp producer
# falls back to pytsmod WSOLA.
if [ -x "/data1/dipak_data/miniconda3/envs/rubberband/bin/rubberband" ]; then
  export PATH="/data1/dipak_data/miniconda3/envs/rubberband/bin:$PATH"
fi

echo "=================================================================="
echo " ui_run_job   : $JOB_ID"
echo " started      : $(date -Is)"
echo " interpreter  : $PY"
echo " LD_PRELOAD   : ${LD_PRELOAD:-<not set -- $LIBSTDCXX not found>}"
echo " GPUs         : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "=================================================================="

cd "$REPO" || exit 1
exec "$PY" -m frontend.runner "$JOB_ID" "$@"
