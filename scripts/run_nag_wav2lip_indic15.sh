#!/bin/bash
# NAG deep-learning lecture, 4:00-4:30, 15 Indic languages, Wav2Lip (+CodeFormer).
# Launch detached:
#   tmux new-window -t dubsync -n nag15 \
#     "bash /data1/dipak_data/dubbingstudio/scripts/run_nag_wav2lip_indic15.sh"
set -u
cd /data1/dipak_data/dubbingstudio

PY=/data1/dipak_data/miniconda3/envs/dubbing/bin/python
FF=/data1/dipak_data/miniconda3/envs/dubbing/bin/ffmpeg
export PATH=/data1/dipak_data/miniconda3/envs/dubbing/bin:$PATH
export LD_PRELOAD=/data1/dipak_data/miniconda3/envs/dubbing/lib/libstdc++.so.6

# The 54x65 px face is already below Wav2Lip's 96x96 crop. resize_factor=2
# (the script's default) would halve the frame and take it to ~27 px, so the
# model would be inpainting a mouth from almost nothing. Full resolution is
# mandatory here, not a quality preference.
export WAV2LIP_RESIZE_FACTOR=1
export WAV2LIP_BATCH=4
export WAV2LIP_FD_BATCH=4
# GPU 0 holds 10.8 GiB belonging to another user (pid 2726547, rxnpred env).
# Everything here is pinned to GPU 1, which is empty.
export CUDA_VISIBLE_DEVICES=1

STEM_BASE=NAG_dIYDPtHWBNA_240to270
CLIP=data/clips/${STEM_BASE}.mp4
CUES=data/clips/${STEM_BASE}.cues.en.json
LANGS="${LANGS:-as bn gu hi kn ml mr ne or pa sa sd ta te ur}"

TS=$(date +%Y%m%dT%H%M%S)
LOG="logs/wav2lip_nag_indic15_${TS}.log"
STATUS="logs/wav2lip_nag_indic15_${TS}.status"
exec > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] ================ CONFIG ================"
echo "  video        : NAG 'L35: Nesterov accelarated gradient descent' (dIYDPtHWBNA)"
echo "  window       : 240.0-270.0 s (4:00-4:30), 30 s, 25 fps, 1280x720"
echo "  clip         : $CLIP"
echo "  face         : 1 face in every frame; 54x65 px (PiP inset, head tilted down)"
echo "                 -> BELOW wav2lip's 96x96 crop; mouth detail is source-limited"
echo "  text source  : YouTube auto-caption dIYDPtHWBNA.en-orig.vtt, de-rolled via"
echo "                 inline word timestamps (plain cue text has 0.621 duplication)"
echo "  translation  : Bodhan indic-translate, target_script=CODEMIX (default)"
echo "                 English technical terms stay in Latin inside the Indic"
echo "                 sentence ('weight सही है'). native mode re-scripts"
echo "                 inconsistently -- same term 3 ways in one 30s clip --"
echo "                 and TTS does not reliably say a bare Latin remnant."
echo "                 NOT YouTube's translated captions: timedtext is 429-throttled."
echo "  tts          : Bodhan indic-speak, male lead voice per language"
echo "  audio fit    : per-cue, rubberband, stretch cap +/-30%"
echo "  lipsync      : wav2lip, resize_factor=1, batch=4, fp32"
echo "  restore      : codeformer, fidelity 0.7, upscale 1  (BOTH renders kept)"
echo "  device       : GPU 1 (GPU 0 is in use by another user, 10.8 GiB)"
echo "  languages    : $LANGS"
echo "  est. wall    : ~10.5 min/lang (237 s wav2lip + 386 s codeformer + tts)"
echo "                 ~2.6 h for all 15"
echo "  log          : $LOG"
echo "  status       : $STATUS"
echo "[$(date +%H:%M:%S)] ========================================"

echo "state=starting langs_done=0/15 ts=$(date -Is)" > "$STATUS"
mkdir -p data/mix data/lipsync

i=0
for LG in $LANGS; do
  i=$((i+1))
  STEM="${STEM_BASE}_${LG}"
  echo "[$(date +%H:%M:%S)] ===== [$i/15] $LG : START ====="
  T0=$(date +%s)

  echo "[$(date +%H:%M:%S)] [$LG] P1 dub audio (MT + TTS + fit)"
  $PY -I scripts/clip_dub_audio.py \
      --cues "$CUES" --lang "$LG" \
      --out-wav "data/mix/${STEM}.dubbed.wav" \
      --work-dir "data/mix/_work/${STEM}" || { echo "[$LG] AUDIO FAILED, skipping"; continue; }

  echo "[$(date +%H:%M:%S)] [$LG] P2 mux dub onto clip"
  "$FF" -y -v error -i "$CLIP" -i "data/mix/${STEM}.dubbed.wav" \
        -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest \
        "data/mix/${STEM}.dubbed.mp4" || { echo "[$LG] MUX FAILED"; continue; }

  echo "[$(date +%H:%M:%S)] [$LG] P3 wav2lip + codeformer"
  $PY -I scripts/run_lipsync_only.py --model wav2lip --stem "$STEM" \
      --restore codeformer --restore-fidelity 0.7 --restore-upscale 1
  RC=$?

  T1=$(date +%s)
  echo "[$(date +%H:%M:%S)] ===== [$i/15] $LG : DONE rc=$RC in $((T1-T0))s ====="
  echo "state=running langs_done=$i/15 last=$LG rc=$RC ts=$(date -Is)" > "$STATUS"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
done

# run_lipsync_only.py writes flat into data/lipsync/ (its line 67), which
# predates CLAUDE.md §2 -- one run, one job folder, nothing written outside it.
# Regroup so this sweep matches the MuseTalk half on disk.
echo "[$(date +%H:%M:%S)] ===== REGROUP into data/jobs/ ====="
cat > "/tmp/nag_cfg_${TS}.json" <<CFG
{
  "src": "$CLIP",
  "title": "L35: Nesterov accelarated gradient descent",
  "window_abs_s": [240.0, 270.0], "window_hhmmss": ["00:04:00", "00:04:30"],
  "resolution": "1280x720", "fps": 25,
  "face_note": "single face every frame, 54x65 px PiP inset; below wav2lip 96x96 crop",
  "text_source": "YouTube en-orig auto-caption, de-rolled via inline word timings",
  "mt": "bodhan indic-translate", "mt_target_script": "codemix",
  "tts": "bodhan indic-speak, male lead voice",
  "audio_fit": "per-cue rubberband, cap +/-30%",
  "wav2lip_resize_factor": 1, "restore_fidelity": 0.7, "gpu": 1
}
CFG
$PY -I scripts/regroup_lipsync_into_jobs.py \
    --stem-base "$STEM_BASE" \
    --job-prefix "${TS}__dIYDPtHWBNA_240to270__wav2lip_codeformer" \
    --config-json "/tmp/nag_cfg_${TS}.json"

echo "[$(date +%H:%M:%S)] ================ DONE ================"
echo "state=done langs_done=$i/15 ts=$(date -Is)" > "$STATUS"
ls -d data/jobs/${TS}__dIYDPtHWBNA_240to270__wav2lip_codeformer_*/ 2>/dev/null
