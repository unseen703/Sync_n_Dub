#!/usr/bin/env bash
# Run ONE lip-sync adapter on an already-dubbed mix, waiting for a free GPU.
#
#   scripts/run_lipsync.sh <wav2lip|musetalk|latentsync|lipgan> <stem>
#
# Replaces retry_wav2lip.sh / retry_latentsync.sh / run_musetalk.sh, which were
# this same script three times over: each pinned one model, one GPU and the
# stem 2vxQaT0Fy-c_1min, whose media no longer exists. Everything that actually
# differed between them is an argument or an env var here.
#
#   GPU=0|1       which card to pin                     (default 0)
#   WAIT_MIB=200  wait until every GPU is under this    (0 disables the wait)
#   TIMEOUT=3600  seconds before the run is killed
#
# Launch detached per the long-run skill:
#   tmux new-window -t dubsync -n w2l "bash scripts/run_lipsync.sh wav2lip MY_STEM"
#
# NOTE ON OUTPUT PATH: run_lipsync_only.py still writes to the flat
# data/lipsync/ + data/mix/ layout, not to data/jobs/<job_id>/ as CLAUDE.md §2
# requires. This wrapper does not fix that -- scripts/regroup_lipsync_into_jobs.py
# exists solely to clean up after it. Fixing the writer is the real repair.
set -u

REPO=/data1/dipak_data/dubbingstudio
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
cd "$REPO"

if [ $# -lt 2 ]; then
  echo "usage: $0 <wav2lip|musetalk|latentsync|lipgan> <stem> [extra args...]" >&2
  exit 2
fi
MODEL=$1; STEM=$2; shift 2

GPU=${GPU:-0}
WAIT_MIB=${WAIT_MIB:-200}
TIMEOUT=${TIMEOUT:-3600}

STAMP=$(date +%Y%m%dT%H%M%S)
mkdir -p logs
LOG=${LOG:-$REPO/logs/lipsync_${MODEL}_${STEM}_${STAMP}.log}
exec > "$LOG" 2>&1

status() { echo "[$(date +%H:%M:%S)] $*"; }
status "=== $MODEL on $STEM (gpu=$GPU, timeout=${TIMEOUT}s) ==="

# MuseTalk reads config.json from whichever checkpoint dir it loads; upstream
# ships neither, so both point at the same vendored file.
if [ "$MODEL" = musetalk ]; then
  MT_REPO=$REPO/third_party/MuseTalk
  ln -sf musetalk.json "$MT_REPO/models/musetalk/config.json"    2>/dev/null || true
  ln -sf musetalk.json "$MT_REPO/models/musetalkV15/config.json" 2>/dev/null || true
fi

# Wait for the cards to drain rather than competing with a run already in
# flight -- 16 GiB leaves MuseTalk under 100 MiB of headroom at batch_size=20,
# so a second tenant is an OOM, not a slowdown (CLAUDE.md §1).
if [ "$WAIT_MIB" -gt 0 ]; then
  status "waiting for both GPUs below ${WAIT_MIB} MiB..."
  while true; do
    U0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    if [ "${U0:-99999}" -lt "$WAIT_MIB" ] && [ "${U1:-99999}" -lt "$WAIT_MIB" ]; then
      status "GPUs free (${U0}, ${U1} MiB)"; break
    fi
    sleep 30
  done
fi

CUDA_VISIBLE_DEVICES=$GPU timeout "$TIMEOUT" \
  "$PY" "$REPO/scripts/run_lipsync_only.py" --model "$MODEL" --stem "$STEM" "$@"
RC=$?

status "$MODEL exit=$RC"
ls -la "$REPO/data/lipsync/${STEM}.${MODEL}."*.mp4 2>/dev/null \
  || status "no output file for $MODEL"
exit $RC
