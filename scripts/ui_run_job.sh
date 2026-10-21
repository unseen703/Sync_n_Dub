#!/usr/bin/env bash
# Launch one UI-submitted dubbing job, detached.
#
#   scripts/ui_run_job.sh <job_id> [--rerun]
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

REPO=/data1/dipak_data/dubbingstudio
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
ENV_LIB=/data1/dipak_data/miniconda3/envs/dubbing/lib

if [ $# -lt 1 ]; then
  echo "usage: $0 <job_id> [--rerun]" >&2
  exit 2
fi
JOB_ID=$1
shift

JOB_DIR="${DUBBING_UI_DATA_DIR:-$REPO/data}/jobs/$JOB_ID"
if [ ! -d "$JOB_DIR" ]; then
  echo "no such job folder: $JOB_DIR" >&2
  exit 2
fi

mkdir -p "$JOB_DIR"
exec >> "$JOB_DIR/run.log" 2>&1

export LD_PRELOAD="$ENV_LIB/libstdc++.so.6"
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo " ui_run_job   : $JOB_ID"
echo " started      : $(date -Is)"
echo " interpreter  : $PY"
echo " LD_PRELOAD   : $LD_PRELOAD"
echo " GPUs         : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "=================================================================="

cd "$REPO" || exit 1
exec "$PY" -m ui.runner "$JOB_ID" "$@"
