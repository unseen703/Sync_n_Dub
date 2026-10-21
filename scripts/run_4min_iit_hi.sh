#!/usr/bin/env bash
# One 4-minute Hindi dub + MuseTalk lip-sync of the IIT professor clip.
#
#   bash scripts/run_4min_iit_hi.sh [gpu-index]
#
# Defaults to GPU 1. Pass 0 to use the other card.
#
# Pinned to ONE card with --lipsync-num-gpus 1, not sharded across both:
# this box is shared, and `backend/lipsync.py` refuses to start lip-sync on a
# card another process is using. Asking for two cards means the run is
# refused unless BOTH are clear -- and that refusal lands at stage 9h, after
# ASR, MT and TTS have already been paid for. One card is the bet that
# actually pays off here.
set -uo pipefail
cd /data1/dipak_data/dubbingstudio

GPU="${1:-1}"
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
CLIP=data/clips/iit_1080p_4min.mp4
TS="$(date +%Y%m%dT%H%M%S)"
JOB="hi4min_iit_${TS}"
LOG="logs/musetalk_4min_iit_${TS}.log"

# cv2's bundled libtbb needs the conda libstdc++; whichever library loads
# first decides, so stage 9h's face work fails without this (CLAUDE.md §1).
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES="$GPU"
# -13% wall, -25% peak VRAM, face crop unchanged (34.45 vs 34.46 dB).
export MUSETALK_FP16=1

[ -s "$CLIP" ] || { echo "missing clip: $CLIP"; exit 2; }

# Refuse EARLY rather than at stage 9h. The guard inside run_lipsync fires
# only once the dub is made; this one costs nothing.
#
# Read memory PER INDEX. Matching `--query-compute-apps` rows back to a card
# through gpu_uuid is easy to get wrong -- a sloppy grep reported every
# process on GPU 0 and would have sent this run at the one card that is
# genuinely occupied.
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)"
if [ -z "$USED" ]; then
  echo "could not read GPU $GPU"; exit 2
fi
# ~100 MiB is idle driver overhead; anything above it is somebody's workload.
if [ "$USED" -gt 500 ]; then
  echo "GPU $GPU already holds ${USED} MiB -- lip-sync needs a card to itself."
  echo "Other card:  bash $0 $((1 - GPU))"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  gpu /'
  exit 3
fi

mkdir -p logs
echo "job $JOB -> $LOG (gpu $GPU)"
nohup "$PY" -m backend.cli --input "$CLIP" \
    --job-id "$JOB" \
    --model pyannote --resegment-turns --refine-speaker-labels \
    --face-detector retinaface --asd talknet \
    --asr bodhan --align mms_fa \
    --mt bodhan --mt-target-language hi --mt-target-script codemix \
    --tts bodhan --voice-engine indic-speak --tts-style auto \
    --mix --lipsync musetalk --lipsync-num-gpus 1 \
    > "$LOG" 2>&1 &
echo "pid $!"
echo "watch:  grep -E '=== stage' $LOG | tail -5"
echo "out:    data/jobs/$JOB/lipsync/"
