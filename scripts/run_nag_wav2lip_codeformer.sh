#!/usr/bin/env bash
# NAG lecture, 1:30-2:00, Wav2Lip + CodeFormer face restoration.
#
#   video : data/videos/nag_nesterov_accelerated_gradient_descent_dIYDPtHWBNA/dIYDPtHWBNA.mp4
#   window: 90 -> 120 s (30 s), 1280x720 @ 25 fps
#   audio : the DOWNLOADED YouTube Hindi dub track (dub_kind=dubbed-auto,
#           i.e. YouTube machine dubbing) -- MT and TTS are skipped entirely
#   model : wav2lip (resize_factor=1, native 720p) then codeformer (-w 0.7 -s 1)
#   gpu   : CUDA_VISIBLE_DEVICES=1
set -u
cd /data1/dipak_data/dubbingstudio

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
FF=/data1/dipak_data/miniconda3/envs/dubbing/bin/ffmpeg
STEM=NAG_dIYDPtHWBNA_90to120_rawhi
SRC_DIR=data/videos/nag_nesterov_accelerated_gradient_descent_dIYDPtHWBNA
JOB=data/jobs/20261012T2230__dIYDPtHWBNA_90to120__wav2lip_codeformer_rawhi
START=90; END=120

mkdir -p "$JOB" data/mix logs
LOG="logs/wav2lip_codeformer_${STEM}_$(date +%Y%m%dT%H%M%S).log"
exec > "$LOG" 2>&1

export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=1

echo "=================================================================="
echo " launch   : bash scripts/run_nag_wav2lip_codeformer.sh"
echo " job      : $JOB"
echo " video    : $SRC_DIR/dIYDPtHWBNA.mp4  (L35: Nesterov accelarated gradient descent)"
echo " window   : ${START}s -> ${END}s  (1:30-2:00, 30 s, 750 frames @ 25 fps)"
echo " audio    : $SRC_DIR/dIYDPtHWBNA.hi.webm  (downloaded YT hi dub, dubbed-auto)"
echo " path     : downloaded-dub -- MT/TTS/voice-assignment NOT run"
echo " lipsync  : wav2lip  resize_factor=1 batch=8 fd_batch=4 (native 1280x720)"
echo " restore  : codeformer -w 0.7 -s 1 bg_upsampler=None (face region only)"
echo " gpu      : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo " note     : speaker is a small picture-in-picture inset (~70 px face)"
echo "=================================================================="

echo "[$(date +%T)] cutting video window"
"$FF" -y -hide_banner -loglevel error -ss $START -to $END \
      -i "$SRC_DIR/dIYDPtHWBNA.mp4" \
      -c:v libx264 -crf 17 -preset medium -an \
      "$JOB/${STEM}.video.mp4" || { echo "video cut FAILED"; exit 1; }

echo "[$(date +%T)] cutting hindi dub window"
"$FF" -y -hide_banner -loglevel error -ss $START -to $END \
      -i "$SRC_DIR/dIYDPtHWBNA.hi.webm" \
      -ar 48000 -ac 2 "data/mix/${STEM}.dubbed.wav" \
      || { echo "audio cut FAILED"; exit 1; }

echo "[$(date +%T)] muxing dubbed audio onto the window"
"$FF" -y -hide_banner -loglevel error \
      -i "$JOB/${STEM}.video.mp4" -i "data/mix/${STEM}.dubbed.wav" \
      -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -ar 48000 -ac 2 -b:a 192k \
      -shortest "data/mix/${STEM}.dubbed.mp4" \
      || { echo "mux FAILED"; exit 1; }

echo "[$(date +%T)] cut durations (video / audio must match):"
for f in "data/mix/${STEM}.dubbed.mp4" "data/mix/${STEM}.dubbed.wav"; do
  echo -n "   $f  "
  /data1/dipak_data/miniconda3/envs/dubbing/bin/ffprobe -v error \
    -show_entries format=duration -of default=nw=1:nk=1 "$f"
done

echo "[$(date +%T)] starting wav2lip + codeformer"
WAV2LIP_RESIZE_FACTOR=1 WAV2LIP_BATCH=8 WAV2LIP_FD_BATCH=4 \
"$PY" scripts/run_lipsync_only.py \
    --model wav2lip --stem "$STEM" \
    --out-dir "$JOB" \
    --restore codeformer \
    --restore-fidelity 0.7 --restore-upscale 1
RC=$?
echo "[$(date +%T)] pipeline rc=$RC"
ls -la "$JOB"
echo "[$(date +%T)] DONE"
