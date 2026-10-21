#!/usr/bin/env bash
# All remaining 30-second segments, strictly ONE AT A TIME.
#
#   bash scripts/run_seg30_sequential.sh <wait-pids-csv> <gpu> <tag:glob> [tag:glob ...]
#
# `run_seg30_lane.sh` is already sequential WITHIN a lane; what this adds is
# that no two lanes overlap either. Two lanes on two cards finish sooner, but
# they also make the host's behaviour harder to read: a 30s segment that OOMs
# or slows down cannot be blamed on its own content while another render is
# competing for PCIe, RAM bandwidth and the Bodhan rate limiter. One at a
# time, every segment is measured under the same conditions.
#
# It waits on PIDs rather than on free VRAM: another account holds ~4 GiB on
# GPU 0, so `backend.gpu.free_devices()` reports nothing free and a
# VRAM-threshold gate would never open.
set -uo pipefail
cd /data1/dipak_data/dubbingstudio

WAIT_CSV="${1:?comma-separated pids to wait for (or - for none)}"; shift
GPU="${1:?gpu index}"; shift

ts() { date +%H:%M:%S; }

if [ "$WAIT_CSV" != "-" ]; then
  IFS=',' read -ra PIDS <<< "$WAIT_CSV"
  echo "[$(ts)] holding until these finish: ${PIDS[*]}"
  while :; do
    alive=0
    for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    sleep 60
  done
  echo "[$(ts)] all clear -- starting"
fi

for spec in "$@"; do
  TAG="${spec%%:*}"
  GLOB="${spec#*:}"
  echo "[$(ts)] ===== lane $TAG (gpu $GPU) ====="
  bash scripts/run_seg30_lane.sh "$TAG" "$GPU" "$GLOB"
  echo "[$(ts)] ===== lane $TAG finished (rc=$?) ====="
done
echo "[$(ts)] ALL LANES DONE"
