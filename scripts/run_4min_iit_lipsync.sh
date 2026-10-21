#!/usr/bin/env bash
# Lip-sync the 4-minute IIT professor clip against the dub that ALREADY
# EXISTS. ASR, translation and TTS are NOT re-run.
#
#   bash scripts/run_4min_iit_lipsync.sh [n-gpus]      # default 2
#
# The dub came from the eight 30-second `iit_*` segments, joined back into
# one 4-minute file. Verified against the source before being trusted:
# 1920x1080, 25 fps, 6000 frames, 240.000s on BOTH sides, and the audio
# correlates -0.0048 with the English original, i.e. fully replaced.
#
# No CUDA_VISIBLE_DEVICES here, deliberately. The adapter pins each shard
# itself from the devices `pick_devices()` hands it, and setting the variable
# outside as well does NOT compose: the parent then sees one card as index 0
# and passes "0" to the child, which reads that as PHYSICAL gpu 0. Measured
# 2026-10-20 -- a run exported CUDA_VISIBLE_DEVICES=1 and its MuseTalk child
# started on gpu 0, next to another account's job.
set -uo pipefail
cd /data1/dipak_data/dubbingstudio

NGPU="${1:-2}"
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
SRC=data/clips/iit_1080p_4min.mp4
DUB=data/clips/seg30_2026-10-20/joined/iit_dub_joined.mp4
JOB="iit4min_lipsync_$(date +%Y%m%dT%H%M%S)"

# cv2's bundled libtbb needs the conda libstdc++, and whichever library loads
# first decides for the whole process (CLAUDE.md section 1).
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

for f in "$SRC" "$DUB"; do
  [ -s "$f" ] || { echo "missing: $f"; exit 2; }
done

# Check the cards BEFORE the run. `run_lipsync` refuses a card another
# process is using, and on a full pipeline that refusal arrives only after
# the dub is paid for. Here there is nothing upstream to waste -- but a
# clear message now still beats a stack trace in ten minutes.
#
# Memory is read PER INDEX. Matching `--query-compute-apps` rows back to a
# card through gpu_uuid is easy to get wrong, and a sloppy version of this
# check reported every process on gpu 0.
CLEAR=0; BUSY=""
for g in 0 1; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)"
  [ -z "$used" ] && { echo "could not read gpu $g"; exit 2; }
  # ~100 MiB is idle driver overhead; anything above 500 is a real workload.
  if [ "$used" -gt 500 ]; then BUSY="$BUSY gpu$g(${used}MiB)"; else CLEAR=$((CLEAR+1)); fi
done

if [ "$CLEAR" -lt "$NGPU" ]; then
  echo "asked for $NGPU card(s); only $CLEAR clear.  busy:$BUSY"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
    | while IFS=, read -r pp mm; do
        echo "   pid$pp $mm $(ps -o user= -p "${pp// /}" 2>/dev/null)"
      done
  echo
  echo "Another account's process counts: lip-sync will not share a card."
  [ "$CLEAR" -ge 1 ] && echo "Run on the clear one instead:  bash $0 1"
  exit 3
fi

mkdir -p logs
echo "job   $JOB   (${NGPU} gpu, lip-sync only -- no ASR/MT/TTS)"
nohup "$PY" scripts/lipsync_from_dub.py \
    --source "$SRC" --dub "$DUB" \
    --job-id "$JOB" --model musetalk --gpus "$NGPU" \
    > "logs/${JOB}.out" 2>&1 &
echo "pid   $!"
echo "watch tail -f logs/${JOB}.log"
echo "out   data/jobs/${JOB}/lipsync/"
