#!/usr/bin/env bash
# A/B the MuseTalk lazy-frame change on a 2-minute IIT professor clip.
#
#   bash scripts/run_ab_lazy_frames.sh [seconds] [gpu]
#
# Three renders: a warm-up to populate the coord cache (discarded), then
# eager, then lazy. ~15-20 min at 120s per arm.
set -uo pipefail
cd /data1/dipak_data/dubbingstudio

SECONDS_ARG="${1:-120}"
GPU="${2:-1}"
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python

# cv2's bundled libtbb needs the conda libstdc++, and whichever library
# loads first decides for the whole process (CLAUDE.md section 1).
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

# The guard would pause mid-batch and distort the wall-time measurement.
# This A/B is about memory, and it is bounded by the clip length; the
# guard is not what is being tested.
export DUBBING_MEM_GUARD_PCT=0

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)"
if [ -z "$used" ]; then echo "could not read gpu $GPU"; exit 2; fi
if [ "$used" -gt 500 ]; then
  echo "gpu $GPU already holds ${used} MiB -- the A/B needs a card to itself"
  echo "other card:  bash $0 $SECONDS_ARG $((1 - GPU))"
  exit 3
fi

exec "$PY" scripts/ab_lazy_frames.py --seconds "$SECONDS_ARG" --gpu "$GPU"
