#!/usr/bin/env bash
# Wait for a lane already on this GPU to finish, then start another.
#
#   bash scripts/chain_seg30_lane.sh <wait-pid> <lane-tag> <gpu> <clip-glob>
#
# One MuseTalk render is ~10 GiB at 1080p/batch 10, so two lanes cannot
# share a card -- the second would OOM at stage 9h after paying for ASR, MT
# and TTS. Rather than gate on free VRAM (which also sees other accounts'
# jobs and would never open), this waits on the specific lane that owns the
# card and hands over the moment it exits.
set -uo pipefail
WAIT_PID="${1:?pid to wait for}"; shift
cd /data1/dipak_data/dubbingstudio
echo "[$(date +%H:%M:%S)] waiting for pid $WAIT_PID to finish before lane '$1' on gpu $2"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] pid $WAIT_PID gone -- starting lane '$1'"
exec bash scripts/run_seg30_lane.sh "$@"
