#!/usr/bin/env bash
# One LANE of the 2026-10-20 30-second segment run.
#
#   bash scripts/run_seg30_lane.sh <lane-tag> <gpu-index> <clip-glob>
#
# Each lane owns one GPU and walks its clips SEQUENTIALLY — the request was
# "run each 30s segment separately", so every segment is its own job folder
# with its own manifest, not one run over four minutes.
#
# Two lanes in parallel, one per card, rather than `--lipsync-num-gpus 2`:
# sharding one clip across both cards makes each shard do its own face
# detection smoothing, and the seam shows as a one-frame mouth jump
# (`clip-run` skill). Separate jobs on separate cards have no seam.
#
# Launch (per the `long-run` skill, so it outlives the session):
#   tmux new-window -t dubsync -n seg30-dk  "bash scripts/run_seg30_lane.sh dk 0 'data/clips/seg30_2026-10-20/L3oOldViIgY_*.mp4'"
#   tmux new-window -t dubsync -n seg30-w4  "bash scripts/run_seg30_lane.sh w4l11 1 'data/clips/seg30_2026-10-20/IBfVkrvqAI_*.mp4'"
set -uo pipefail

LANE="${1:?lane tag}"
GPU="${2:?gpu index}"
GLOB="${3:?clip glob}"

REPO=/data1/dipak_data/dubbingstudio
PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
cd "$REPO"

STAMP="$(date +%Y%m%dT%H%M%S)"
LOG="logs/seg30_${LANE}_${STAMP}.log"
STATUS="logs/seg30_${LANE}.STATUS"
mkdir -p logs
exec > "$LOG" 2>&1

# cv2/insightface decide which libstdc++ the process gets by load order; the
# UI launcher is the only other place that exports this, and without it stage
# 9h walks into the import-order trap in CLAUDE.md §1.
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES="$GPU"
# No --use-float16 on backend/cli.py; MUSETALK_FP16 is the only route from the
# pipeline. Measured on this host: -13% wall, -25% peak VRAM, face crop
# unchanged (34.45 vs 34.46 dB).
export MUSETALK_FP16=1

ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*"; }
mark() { echo "$(date -Is) $LANE $*" > "$STATUS"; }

mapfile -t CLIPS < <(ls -1 $GLOB 2>/dev/null | sort -t_ -k2 -n)

cat <<EOF
=========================================================================
 30-second segment run — lane "$LANE"
 launched      $(date -Is)
 command       bash scripts/run_seg30_lane.sh $LANE $GPU '$GLOB'
 log           $LOG
 status file   $STATUS
-------------------------------------------------------------------------
 gpu           CUDA_VISIBLE_DEVICES=$GPU
 clips         ${#CLIPS[@]} x 30s, run SEPARATELY (one job folder each)
 stack         asr=bodhan  mt=bodhan -> hi  tts=bodhan  mix  lipsync=musetalk
 musetalk      fp16=on  batch_size=10 (adapter default; 99 MiB headroom at
               batch 20 on 1080p, so 10 is the safe size at 30s)
 resolution    1920x1080 for both sources — the Dark Knight 4K was
               downscaled to match the existing 120s benchmarks
=========================================================================
EOF
for c in "${CLIPS[@]}"; do echo "   clip: $c"; done
echo

OK=0; FAIL=0
for i in "${!CLIPS[@]}"; do
  CLIP="${CLIPS[$i]}"
  STEM="$(basename "$CLIP" .mp4)"
  JOB="seg30_${LANE}_${STEM}"
  N=$((i + 1))

  say "=== [$N/${#CLIPS[@]}] $STEM -> data/jobs/$JOB"
  mark "running $N/${#CLIPS[@]} $STEM"
  T0=$(date +%s)

  "$PY" -m backend.cli \
      --input "$CLIP" \
      --job-id "$JOB" \
      --asr bodhan \
      --mt bodhan --mt-target-language hi \
      --tts bodhan \
      --mix \
      --lipsync musetalk
  RC=$?

  DT=$(( $(date +%s) - T0 ))
  # The FILE is the arbiter, not the exit code: a lip-sync adapter can exit 0
  # having written nothing (CLAUDE.md §4), so check for the render itself.
  OUTV="$(ls -S data/jobs/$JOB/lipsync/*.mp4 2>/dev/null | head -1)"
  if [ $RC -eq 0 ] && [ -s "$OUTV" ]; then
    OK=$((OK + 1))
    say "    done rc=0 in ${DT}s -> $OUTV ($(du -h "$OUTV" | cut -f1))"
  else
    FAIL=$((FAIL + 1))
    say "    FAILED rc=$RC in ${DT}s; lipsync mp4 missing or empty"
    say "    mix: $(ls data/jobs/$JOB/mix/*.mp4 2>/dev/null | head -1 || echo none)"
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/    gpu /'
done

say "LANE DONE: $OK ok, $FAIL failed, of ${#CLIPS[@]}"
mark "finished $OK ok / $FAIL failed"
