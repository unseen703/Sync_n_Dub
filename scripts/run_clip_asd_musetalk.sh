#!/bin/bash
# ASD-gated MuseTalk clip run, launched detached so it survives the session.
#
#   tmux new-window -t dubsync -n clip2320 \
#     "bash /data1/dipak_data/dubbingstudio/scripts/run_clip_asd_musetalk.sh"
#
# Override any of START / DURATION / JOB_ID / DUB_LANG / GPU in the environment.
# Log:    logs/musetalk_asdgated_<tag>_<stamp>.log
# Status: data/jobs/<JOB_ID>/STATUS.json   (read this, not the log, for state)

set -u
REPO=/data1/dipak_data/dubbingstudio
cd "$REPO"

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
# cv2's bundled libtbb needs CXXABI_1.3.15, which the system libstdc++ lacks.
# Without this, `import insightface` before anything else fails outright --
# the failure is import-ORDER dependent, so it looks intermittent.
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

ASSET="$REPO/data/videos/iit_profs_quantum_image_disaster_svwrTOm55y4"
SRC="$ASSET/svwrTOm55y4.mp4"
EN_AUDIO="$ASSET/svwrTOm55y4.en.44k.stereo.wav"

START=${START:-1400}
DURATION=${DURATION:-20}
# NB: never name this LANG -- that is the shell locale var and is always set.
DUB_LANG=${DUB_LANG:-hi}
GPU=${GPU:-0}
BATCH=${BATCH:-8}
FP16=${FP16:-0}      # 1 -> pass --use-float16 to MuseTalk
# 1 -> also render the diagnostic box/ASD overlays. Nothing downstream reads
# them and they cost several minutes a run, so they are opt-in.
BBOX_PREVIEW=${BBOX_PREVIEW:-0}
TAG=${TAG:-clip${START}to$((START + DURATION))}
STAMP=$(date +%Y%m%dT%H%M%S)
JOB_ID=${JOB_ID:-${STAMP}__svwrTOm55y4_${START}to$((START + DURATION))__musetalk_asdgated_${DUB_LANG}}
DUB_AUDIO="$ASSET/svwrTOm55y4.${DUB_LANG}.webm"

mkdir -p "$REPO/logs"
LOG=${LOG:-$REPO/logs/musetalk_asdgated_${TAG}_${STAMP}.log}
exec > "$LOG" 2>&1

echo "=== ASD-gated MuseTalk clip run ==="
echo "started      : $(date -Is)"
echo "host         : $(hostname)"
echo "job_id       : $JOB_ID"
echo "job_dir      : data/jobs/$JOB_ID"
echo "status file  : data/jobs/$JOB_ID/STATUS.json"
echo "source       : $SRC"
echo "window       : ${START}s -> $((START + DURATION))s (${DURATION}s)"
echo "dub audio    : $DUB_AUDIO  (downloaded YouTube track; MT+TTS SKIPPED)"
echo "asd audio    : $EN_AUDIO   (ORIGINAL English -- the dub is not lip-synchronous)"
echo "lipsync      : musetalk v15, batch=$BATCH, gpu=$GPU, bbox_shift=0, precision=$([ "$FP16" = 1 ] && echo fp16 || echo fp32)"
echo "face policy  : ASD-gated -- only single-speaking-face shots get lip-synced"
echo "bbox preview : $([ "$BBOX_PREVIEW" = 1 ] && echo "ON -- overlays rendered" || echo "off (BBOX_PREVIEW=1 to render overlays)")"
echo "working res  : 1920x1080 (source is 3840x2160; downscaled for cost)"
echo
echo "relaunch with:"
echo "  START=$START DURATION=$DURATION DUB_LANG=$DUB_LANG JOB_ID=$JOB_ID \\"
echo "    bash $REPO/scripts/run_clip_asd_musetalk.sh"
echo "=================================="
echo

CUDA_VISIBLE_DEVICES=$GPU $PY "$REPO/scripts/clip_asd_musetalk_run.py" \
  --src "$SRC" \
  --dub-audio "$DUB_AUDIO" \
  --en-audio "$EN_AUDIO" \
  --start "$START" \
  --duration "$DURATION" \
  --lang "$DUB_LANG" \
  --job-id "$JOB_ID" \
  --batch-size "$BATCH" \
  $([ "$FP16" = 1 ] && echo --use-float16) \
  $([ "$BBOX_PREVIEW" = 1 ] && echo --bbox-preview) \
  --gpu 0
RC=$?

echo
echo "=== exit=$RC at $(date -Is) ==="
cat "$REPO/data/jobs/$JOB_ID/STATUS.json" 2>/dev/null
exit $RC
